use anyhow::Result;

use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent, ElementState};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::event_loop::EventLoop;
use winit::window::WindowBuilder;

use graphic_env::app::App;

#[rustfmt::skip]
fn main() -> Result<()> {
    pretty_env_logger::init();

    // Window
    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title("Vulkan Tutorial (Rust)")
        .with_inner_size(LogicalSize::new(1024, 768))
        .build(&event_loop)?;

    // App
    let mut app = App::create(&window)?;
    let mut minimized = false;
    event_loop.run(move |event, elwt| {
        match event {
            // Request a redraw when all events were processed.
            Event::AboutToWait => window.request_redraw(),
            Event::WindowEvent { event, .. } => match event {
                // Render a frame if our Vulkan app is not being destroyed.
                WindowEvent::RedrawRequested if !elwt.exiting() && !minimized => {
                    app.render(&window).unwrap();
                },
                // Mark the window as having been resized.
                WindowEvent::Resized(size) => {
                    if size.width == 0 || size.height == 0 {
                        minimized = true;
                    } else {
                        minimized = false;
                        app.resized = true;
                    }
                }
                // Destroy our Vulkan app.
                WindowEvent::CloseRequested => {
                    elwt.exit();
                    unsafe { app.destroy(); }
                }
                // Handle keyboard events.
                WindowEvent::KeyboardInput { event, .. } => {
                    if event.state == ElementState::Pressed {
                        match event.physical_key {
                            PhysicalKey::Code(KeyCode::ArrowLeft) if app.models > 1 => app.models -= 1,
                            PhysicalKey::Code(KeyCode::ArrowRight) if app.models < 4 => app.models += 1,
                            _ => { }
                        }
                    }
                }
                _ => {}
            }
            _ => {}
        }
    })?;

    Ok(())
}