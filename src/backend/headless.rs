//! Headless backend for tests and headless operation.
//!
//! Without a renderer (`add_renderer`), this is the minimal test backend used by the test suite.
//! With a renderer, it also provides dmabuf imports for clients and a timer-driven frame cadence,
//! which is enough to run real clients and capture them through screencopy.

use std::mem;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Context as _;
use calloop::timer::{TimeoutAction, Timer};
use niri_config::OutputName;
use smithay::backend::allocator::dmabuf::Dmabuf;
use smithay::backend::drm::DrmNode;
use smithay::backend::egl::native::EGLSurfacelessDisplay;
use smithay::backend::egl::{EGLContext, EGLDevice, EGLDisplay};
use smithay::backend::renderer::element::RenderElementStates;
use smithay::backend::renderer::gles::GlesRenderer;
use smithay::backend::renderer::ImportDma;
use smithay::output::{Mode, Output, PhysicalProperties, Subpixel};
use smithay::reexports::wayland_protocols::wp::presentation_time::server::wp_presentation_feedback;
use smithay::utils::Size;
use smithay::wayland::dmabuf::{DmabufFeedbackBuilder, DmabufGlobal};
use smithay::wayland::presentation::Refresh;

use super::{IpcOutputMap, OutputId, RenderResult};
use crate::niri::{Niri, RedrawState, State};
use crate::render_helpers::{resources, shaders};
use crate::utils::{get_monotonic_time, logical_output};

/// Default refresh rate of headless outputs, in millihertz.
pub const DEFAULT_REFRESH_MHZ: u32 = 60_000;

pub struct Headless {
    renderer: Option<GlesRenderer>,
    /// DRM render node of the renderer's EGL device, if it has one (llvmpipe does not).
    render_node: Option<DrmNode>,
    dmabuf_global: Option<DmabufGlobal>,
    ipc_outputs: Arc<Mutex<IpcOutputMap>>,
}

impl Headless {
    pub fn new() -> Self {
        Self {
            renderer: None,
            render_node: None,
            dmabuf_global: None,
            ipc_outputs: Default::default(),
        }
    }

    pub fn init(&mut self, _niri: &mut Niri) {}

    pub fn add_renderer(&mut self) -> anyhow::Result<()> {
        if self.renderer.is_some() {
            error!("add_renderer: renderer must not already exist");
            return Ok(());
        }

        let mut renderer = unsafe {
            let display =
                EGLDisplay::new(EGLSurfacelessDisplay).context("error creating EGL display")?;
            let context = EGLContext::new(&display).context("error creating EGL context")?;
            GlesRenderer::new(context).context("error creating renderer")?
        };

        resources::init(&mut renderer);
        shaders::init(&mut renderer);

        // The surfaceless EGL display is created once, so its render node cannot change.
        self.render_node = match fetch_render_node(&renderer) {
            Ok(node) => {
                if let Some(path) = node.dev_path() {
                    debug!("using as the render node: {path:?}");
                } else {
                    debug!("using as the render node: {node}");
                }
                Some(node)
            }
            Err(err) => {
                debug!("failed querying render node: {err:?}");
                None
            }
        };

        self.renderer = Some(renderer);
        Ok(())
    }

    /// Advertises dmabuf support to clients, importing through the surfaceless EGL renderer.
    ///
    /// Requires `add_renderer` to have succeeded first.
    pub fn create_dmabuf_global(&mut self, niri: &mut Niri) -> anyhow::Result<()> {
        let renderer = self
            .renderer
            .as_mut()
            .context("create_dmabuf_global: no renderer")?;

        // On software renderers (llvmpipe on a GPU-less box) there may be nothing to import
        // with. Advertising a dmabuf global that fails every import just wastes client
        // round-trips; let clients use wl_shm from the start instead.
        if renderer.dmabuf_formats().iter().next().is_none() {
            anyhow::bail!("renderer supports no dmabuf formats");
        }

        let default_feedback = || {
            let node = self
                .render_node
                .as_ref()
                .context("no render node available")?;
            let primary_formats = renderer.dmabuf_formats();
            DmabufFeedbackBuilder::new(node.dev_id(), primary_formats)
                .build()
                .context("error building dmabuf feedback")
        };

        // Fall back to dmabuf v3 if we failed to build feedback.
        let dmabuf_global = match default_feedback() {
            Ok(feedback) => niri
                .dmabuf_state
                .create_global_with_default_feedback::<State>(&niri.display_handle, &feedback),
            Err(err) => {
                debug!("failed building default dmabuf feedback, falling back to v3: {err:?}");
                let primary_formats = renderer.dmabuf_formats();
                niri.dmabuf_state
                    .create_global::<State>(&niri.display_handle, primary_formats)
            }
        };
        assert!(self.dmabuf_global.replace(dmabuf_global).is_none());
        Ok(())
    }

    /// Adds a headless output.
    ///
    /// `refresh_mhz` is in millihertz and must correspond to a refresh interval strictly below
    /// one second (i.e. above 1000 mHz), which the frame clock requires.
    pub fn add_output(&mut self, niri: &mut Niri, n: u8, size: (u16, u16), refresh_mhz: u32) {
        assert!(refresh_mhz > 1000, "refresh rate must be above 1 Hz");

        let connector = format!("headless-{n}");
        let make = "niri".to_string();
        let model = "headless".to_string();
        let serial = n.to_string();

        let output = Output::new(
            connector.clone(),
            PhysicalProperties {
                size: (0, 0).into(),
                subpixel: Subpixel::Unknown,
                make: make.clone(),
                model: model.clone(),
                serial_number: serial.clone(),
            },
        );

        let mode = Mode {
            size: Size::from((i32::from(size.0), i32::from(size.1))),
            refresh: refresh_mhz as i32,
        };
        output.change_current_state(Some(mode), None, None, None);
        output.set_preferred(mode);

        output.user_data().insert_if_missing(|| OutputName {
            connector,
            make: Some(make),
            model: Some(model),
            serial: Some(serial),
        });

        let physical_properties = output.physical_properties();
        self.ipc_outputs.lock().unwrap().insert(
            OutputId::next(),
            niri_ipc::Output {
                name: output.name(),
                make: physical_properties.make,
                model: physical_properties.model,
                serial: None,
                physical_size: None,
                modes: vec![niri_ipc::Mode {
                    width: size.0,
                    height: size.1,
                    refresh_rate: refresh_mhz,
                    is_preferred: true,
                }],
                current_mode: Some(0),
                is_custom_mode: true,
                vrr_supported: false,
                vrr_enabled: false,
                logical: Some(logical_output(&output)),
                max_bpc: None,
            },
        );

        // Give the frame clock the real refresh interval so it ticks on a proper vblank grid
        // rather than the drifting "now + fallback" path.
        let refresh_interval = Duration::from_nanos(1_000_000_000_000 / u64::from(refresh_mhz));
        niri.add_output(output, Some(refresh_interval), false);
    }

    pub fn seat_name(&self) -> String {
        "headless".to_owned()
    }

    pub fn with_primary_renderer<T>(
        &mut self,
        f: impl FnOnce(&mut GlesRenderer) -> T,
    ) -> Option<T> {
        self.renderer.as_mut().map(f)
    }

    pub fn primary_render_node(&mut self) -> Option<DrmNode> {
        self.render_node
    }

    pub fn render(
        &mut self,
        niri: &mut Niri,
        output: &Output,
        target_presentation_time: Duration,
    ) -> RenderResult {
        let refresh = niri
            .output_state
            .get(output)
            .and_then(|state| state.frame_clock.refresh_interval())
            .map_or(Refresh::Unknown, Refresh::Fixed);

        let states = RenderElementStates::default();
        let mut presentation_feedbacks = niri.take_presentation_feedbacks(output, &states);
        presentation_feedbacks.presented::<_, smithay::utils::Monotonic>(
            target_presentation_time,
            refresh,
            0,
            wp_presentation_feedback::Kind::Vsync,
        );

        let output_state = niri.output_state.get_mut(output).unwrap();
        output_state.frame_callback_sequence = output_state.frame_callback_sequence.wrapping_add(1);

        // In real headless operation (a renderer exists), simulate the display: enter
        // WaitingForVBlank, which gates further redraws, and fire a timer at the target
        // presentation time in place of the hardware vblank event. This is the same flow as the
        // TTY backend and is what actually throttles clients to the refresh rate.
        //
        // In renderer-less test mode, keep the old instant-redraw behavior: the test suite
        // drives time manually and does not expect timers here.
        if self.renderer.is_some() {
            let new_state = RedrawState::WaitingForVBlank {
                redraw_needed: false,
            };
            match mem::replace(&mut output_state.redraw_state, new_state) {
                RedrawState::Idle => unreachable!(),
                RedrawState::Queued => (),
                RedrawState::WaitingForVBlank { .. } => unreachable!(),
                RedrawState::WaitingForEstimatedVBlank(_) => unreachable!(),
                RedrawState::WaitingForEstimatedVBlankAndQueued(token) => {
                    niri.event_loop.remove(token);
                }
            }

            let duration = target_presentation_time.saturating_sub(get_monotonic_time());
            let timer = Timer::from_duration(duration);
            let output = output.clone();
            niri.event_loop
                .insert_source(timer, move |_, _, data| {
                    data.backend.headless().on_vblank_timer(
                        &mut data.niri,
                        output.clone(),
                        target_presentation_time,
                    );
                    TimeoutAction::Drop
                })
                .unwrap();
        } else {
            match mem::replace(&mut output_state.redraw_state, RedrawState::Idle) {
                RedrawState::Idle => unreachable!(),
                RedrawState::Queued => (),
                RedrawState::WaitingForVBlank { .. } => unreachable!(),
                RedrawState::WaitingForEstimatedVBlank(_) => unreachable!(),
                RedrawState::WaitingForEstimatedVBlankAndQueued(_) => unreachable!(),
            }
        }

        RenderResult::Submitted
    }

    /// Simulated vblank: the frame we "submitted" in `render()` has now been "presented".
    fn on_vblank_timer(&self, niri: &mut Niri, output: Output, presentation_time: Duration) {
        let Some(output_state) = niri.output_state.get_mut(&output) else {
            // The output was removed while the timer was pending.
            return;
        };

        let redraw_needed = match mem::replace(&mut output_state.redraw_state, RedrawState::Idle) {
            RedrawState::WaitingForVBlank { redraw_needed } => redraw_needed,
            RedrawState::Idle => unreachable!(),
            RedrawState::Queued => unreachable!(),
            RedrawState::WaitingForEstimatedVBlank(_) => unreachable!(),
            RedrawState::WaitingForEstimatedVBlankAndQueued(_) => unreachable!(),
        };

        output_state.frame_clock.presented(presentation_time);

        if redraw_needed || output_state.unfinished_animations_remain {
            niri.queue_redraw(&output);
        } else {
            niri.send_frame_callbacks(&output);
        }
    }

    pub fn import_dmabuf(&mut self, dmabuf: &Dmabuf) -> bool {
        let Some(renderer) = self.renderer.as_mut() else {
            return false;
        };
        match renderer.import_dmabuf(dmabuf, None) {
            Ok(_texture) => true,
            Err(err) => {
                debug!("error importing dmabuf: {err:?}");
                false
            }
        }
    }

    pub fn ipc_outputs(&self) -> Arc<Mutex<IpcOutputMap>> {
        self.ipc_outputs.clone()
    }
}

fn fetch_render_node(renderer: &GlesRenderer) -> anyhow::Result<DrmNode> {
    let display = renderer.egl_context().display();
    EGLDevice::device_for_display(display)
        .context("error getting EGL device")?
        .try_get_render_node()
        .context("error getting EGL device render node")?
        .context("failed to query EGL device render node")
}

impl Default for Headless {
    fn default() -> Self {
        Self::new()
    }
}
