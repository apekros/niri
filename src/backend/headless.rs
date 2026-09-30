//! Headless backend for tests and headless operation.
//!
//! Without a renderer (`add_renderer`), this is the minimal test backend used by the test suite.
//! With a renderer, it also provides dmabuf imports for clients and a timer-driven frame cadence,
//! which is enough to run real clients and capture them through screencopy.

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::mem;
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Context as _;
use calloop::timer::{TimeoutAction, Timer};
use niri_config::OutputName;
use smithay::backend::allocator::dmabuf::Dmabuf;
use smithay::backend::allocator::Fourcc;
use smithay::backend::drm::DrmNode;
use smithay::backend::egl::native::EGLSurfacelessDisplay;
use smithay::backend::egl::{EGLContext, EGLDevice, EGLDisplay};
use smithay::backend::renderer::damage::OutputDamageTracker;
use smithay::backend::renderer::element::RenderElementStates;
use smithay::backend::renderer::gles::{GlesRenderer, GlesTexture};
use smithay::backend::renderer::{Bind as _, ImportDma};
use smithay::output::{Mode, Output, PhysicalProperties, Subpixel};
use smithay::reexports::wayland_protocols::wp::presentation_time::server::wp_presentation_feedback;
use smithay::utils::Size;
use smithay::wayland::dmabuf::{DmabufFeedbackBuilder, DmabufGlobal};
use smithay::wayland::presentation::Refresh;

use super::{IpcOutputMap, OutputId, RenderResult};
use crate::niri::{Niri, RedrawState, State};
use crate::render_helpers::{create_texture, resources, shaders, RenderCtx, RenderTarget};
use crate::utils::{get_monotonic_time, logical_output};

/// Size and refresh rate of a headless output.
///
/// The refresh rate is always above 1 Hz, since the frame clock needs a refresh interval strictly
/// below one second, and at most 1000 Hz, since anything faster is a typo.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeadlessMode {
    size: (u16, u16),
    refresh_mhz: u32,
}

impl HeadlessMode {
    /// Refresh rate used when none is given, in millihertz.
    const DEFAULT_REFRESH_MHZ: u32 = 60_000;
    /// Exclusive lower bound on the refresh rate, in millihertz.
    const MIN_REFRESH_MHZ: u32 = 1_000;
    /// Inclusive upper bound on the refresh rate, in millihertz.
    const MAX_REFRESH_MHZ: u32 = 1_000_000;

    /// Mode of the given size at the default refresh rate of 60 Hz.
    pub fn new(size: (u16, u16)) -> Self {
        Self {
            size,
            refresh_mhz: Self::DEFAULT_REFRESH_MHZ,
        }
    }

    pub fn size(self) -> (u16, u16) {
        self.size
    }

    pub fn refresh_mhz(self) -> u32 {
        self.refresh_mhz
    }

    pub fn refresh_interval(self) -> Duration {
        Duration::from_nanos(1_000_000_000_000 / u64::from(self.refresh_mhz))
    }
}

impl FromStr for HeadlessMode {
    type Err = String;

    /// Parses `WIDTHxHEIGHT` with an optional `@HZ`, e.g. `1920x1080` or `1920x1080@119.88`.
    fn from_str(mode: &str) -> Result<Self, Self::Err> {
        let err = || {
            format!(
                "invalid headless mode {mode:?}, expected WIDTHxHEIGHT with an optional refresh \
                 rate, e.g. 1920x1080 or 1920x1080@120"
            )
        };

        let (size, refresh) = match mode.split_once('@') {
            Some((size, refresh)) => (size, Some(refresh)),
            None => (mode, None),
        };

        let (width, height) = size.split_once('x').ok_or_else(err)?;
        let width: u16 = width.parse().map_err(|_| err())?;
        let height: u16 = height.parse().map_err(|_| err())?;
        if width == 0 || height == 0 {
            return Err(err());
        }

        let mut this = Self::new((width, height));
        if let Some(refresh) = refresh {
            let hz: f64 = refresh.parse().map_err(|_| err())?;
            // Check the rounded value that is actually used, so that e.g. 1.0004 Hz cannot round
            // down to exactly 1 Hz. The negated comparison also rejects NaN.
            let mhz = (hz * 1000.).round();
            if !(mhz > f64::from(Self::MIN_REFRESH_MHZ) && mhz <= f64::from(Self::MAX_REFRESH_MHZ))
            {
                return Err(format!(
                    "invalid headless refresh rate {refresh:?}, expected a value above 1 and at \
                     most 1000 Hz"
                ));
            }
            this.refresh_mhz = mhz as u32;
        }

        Ok(this)
    }
}

pub struct Headless {
    renderer: Option<GlesRenderer>,
    /// DRM render node of the renderer's EGL device, if it has one (llvmpipe does not).
    render_node: Option<DrmNode>,
    dmabuf_global: Option<DmabufGlobal>,
    /// Offscreen render target and damage tracker per output.
    ///
    /// Rendering for real (rather than skipping it) is what marks surfaces as scanned out on the
    /// output, which in turn is what lets [`Niri::send_frame_callbacks`] send per-redraw frame
    /// callbacks. Without it, every surface looks invisible and clients only receive the ~1 Hz
    /// throttled fallback callbacks, stalling vsynced clients at ~1 fps.
    render_targets: HashMap<Output, RenderTargetState>,
    ipc_outputs: Arc<Mutex<IpcOutputMap>>,
}

/// Offscreen rendering state for one output.
struct RenderTargetState {
    texture: GlesTexture,
    damage_tracker: OutputDamageTracker,
}

impl Headless {
    pub fn new() -> Self {
        Self {
            renderer: None,
            render_node: None,
            dmabuf_global: None,
            render_targets: HashMap::new(),
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

    pub fn add_output(&mut self, niri: &mut Niri, n: u8, mode: HeadlessMode) {
        let size = mode.size();
        let refresh_mhz = mode.refresh_mhz();

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

        let output_mode = Mode {
            size: Size::from((i32::from(size.0), i32::from(size.1))),
            refresh: refresh_mhz as i32,
        };
        output.change_current_state(Some(output_mode), None, None, None);
        output.set_preferred(output_mode);

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
        niri.add_output(output, Some(mode.refresh_interval()), false);
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

        // Render for real when we have a renderer: the resulting element states mark
        // surfaces as scanned out, which drives per-redraw frame callbacks and correct
        // presentation feedback. See the `render_targets` field comment.
        let states = if self.renderer.is_some() {
            match self.render_offscreen(niri, output) {
                Ok(states) => {
                    niri.update_primary_scanout_output(output, &states);
                    states
                }
                Err(err) => {
                    warn!("error rendering headless output: {err:?}");
                    RenderElementStates::default()
                }
            }
        } else {
            RenderElementStates::default()
        };

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

    /// Renders the output to its offscreen target, returning the element states.
    ///
    /// Errors when there is no renderer (test mode).
    fn render_offscreen(
        &mut self,
        niri: &mut Niri,
        output: &Output,
    ) -> anyhow::Result<RenderElementStates> {
        let Self {
            renderer,
            render_targets,
            ..
        } = self;
        let renderer = renderer.as_mut().context("no renderer")?;

        let ctx = RenderCtx {
            renderer,
            target: RenderTarget::Output,
            xray: None,
        };
        let elements = niri.render_to_vec(ctx, output, true);

        let target = match render_targets.entry(output.clone()) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let mode = output.current_mode().context("output has no mode")?;
                let texture = create_texture(renderer, mode.size, Fourcc::Abgr8888)
                    .context("error creating the offscreen texture")?;
                entry.insert(RenderTargetState {
                    texture,
                    damage_tracker: OutputDamageTracker::from_output(output),
                })
            }
        };

        let mut framebuffer = renderer
            .bind(&mut target.texture)
            .context("error binding the offscreen texture")?;
        let res = target
            .damage_tracker
            .render_output(renderer, &mut framebuffer, 0, &elements, [0.; 4])
            .context("error rendering the output")?;
        Ok(res.states)
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

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::HeadlessMode;

    #[test]
    fn headless_mode_parses_size_and_refresh() {
        let mode: HeadlessMode = "1920x1080".parse().unwrap();
        assert_eq!(mode.size(), (1920, 1080));
        assert_eq!(mode.refresh_mhz(), 60_000);

        let mode: HeadlessMode = "640x480@120".parse().unwrap();
        assert_eq!(mode.size(), (640, 480));
        assert_eq!(mode.refresh_mhz(), 120_000);

        let mode: HeadlessMode = "640x480@59.94".parse().unwrap();
        assert_eq!(mode.refresh_mhz(), 59_940);

        assert_eq!(
            "1x1@1000".parse::<HeadlessMode>().unwrap().refresh_mhz(),
            1_000_000
        );
        assert_eq!(
            "1x1@1.001".parse::<HeadlessMode>().unwrap().refresh_mhz(),
            1_001
        );
    }

    #[test]
    fn headless_mode_rejects_invalid_values() {
        for bad in [
            "",
            "1920",
            "1920x",
            "x1080",
            "0x1080",
            "1920x0",
            "-1x1",
            "70000x1",
            "1920x1080@",
            "1920x1080@abc",
            "1920x1080@0",
            "1920x1080@1",
            "1920x1080@-60",
            "1920x1080@1000.1",
            "1920x1080@NaN",
            "1920x1080@inf",
            // Above 1 Hz, but rounds to exactly 1000 mHz, which the frame clock cannot use.
            "1920x1080@1.0004",
        ] {
            assert!(bad.parse::<HeadlessMode>().is_err(), "{bad:?} parsed");
        }
    }

    #[test]
    fn headless_mode_refresh_interval_is_below_one_second() {
        let slowest: HeadlessMode = "1x1@1.001".parse().unwrap();
        assert!(slowest.refresh_interval() < Duration::from_secs(1));
        assert_eq!(
            HeadlessMode::new((1, 1)).refresh_interval(),
            Duration::from_nanos(16_666_666)
        );
    }
}
