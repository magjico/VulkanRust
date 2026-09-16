use anyhow::Result;

use winit::event_loop::{ControlFlow, EventLoop};

use graphic_env::app::AppManager;
use winit_input_helper::WinitInputHelper;

#[rustfmt::skip]
fn main() -> Result<()> {
    pretty_env_logger::init();

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop
        .run_app(
            &mut AppManager {
                input: WinitInputHelper::new(),
                ..Default::default()
            }
        ).unwrap(); 

    Ok(())
}