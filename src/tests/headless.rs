use niri_ipc::{ConfiguredMode, OutputAction};
use smithay::utils::Size;

use super::*;

#[test]
fn headless_output_takes_its_configured_mode_at_runtime() {
    let mut f = Fixture::new();
    f.add_output(1, (800, 600));

    let mode = |f: &Fixture| {
        let mode = f.niri_output(1).current_mode().unwrap();
        (mode.size.w, mode.size.h, mode.refresh)
    };
    let (_, _, refresh) = mode(&f);

    f.niri_state().apply_transient_output_config(
        "headless-1",
        OutputAction::CustomMode {
            mode: ConfiguredMode {
                width: 1000,
                height: 700,
                refresh: Some(30.),
            },
        },
    );

    // The size changes; the refresh rate stays what the frame clock ticks on.
    assert_eq!(mode(&f), (1000, 700, refresh));
    let output = f.niri_output(1);
    let monitor = f.niri().layout.monitor_for_output(&output).unwrap();
    assert_eq!(monitor.view_size(), Size::from((1000., 700.)));
}

#[test]
fn headless_output_takes_a_mode_from_output_management() {
    use crate::protocols::output_management::OutputManagementHandler as _;

    let mut f = Fixture::new();
    f.add_output(1, (800, 600));

    // wlr-output-management names an output it has no config for by what IPC says of it.
    let name = {
        let ipc = f.niri_state().backend.ipc_outputs();
        let ipc = ipc.lock().unwrap();
        let output = ipc.values().next().unwrap();
        niri_config::OutputName::from_ipc_output(output).format_make_model_serial_or_connector()
    };
    let config = niri_config::Output {
        name,
        mode: Some(niri_config::output::Mode {
            custom: true,
            mode: ConfiguredMode {
                width: 640,
                height: 400,
                refresh: Some(60.),
            },
        }),
        ..Default::default()
    };
    f.niri_state()
        .apply_output_config(niri_config::Outputs(vec![config]));

    let mode = f.niri_output(1).current_mode().unwrap();
    assert_eq!((mode.size.w, mode.size.h), (640, 400));
}
