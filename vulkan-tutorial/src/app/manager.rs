use std::time::Instant;

use log::*;

use winit::window::{
	Window,
	WindowId,
};
use winit_input_helper::WinitInputHelper;
use winit::application::ApplicationHandler;
use winit::event_loop::ActiveEventLoop;
use winit::dpi::LogicalSize;
use winit::event::{
	WindowEvent,
	DeviceEvent,
	DeviceId,
	StartCause,
};

use super::app::App;
use crate::scene::{
	ECSContext,
	Time,
	CurrentFrame,
};

/// Manage [App] and [Window] event interaction.
#[derive(Default)]
pub struct AppManager {
    pub window: Option<Window>,
    pub app: Option<App>,
    pub input: WinitInputHelper,
	pub last_frame_time: Option<Instant>,

    pub fps_accumulator: f32,
    pub fps_frame_count: u32,
}


impl ApplicationHandler for AppManager {
    fn window_event(
        &mut self,
        elwt: &ActiveEventLoop,
        _: WindowId,
        event: WindowEvent,
    ) {
        if self.input.process_window_event(&event) {
            let Some(window) = self.window.as_mut() else { return };
            let Some(app) = self.app.as_mut() else { return };

            match event {
                WindowEvent::RedrawRequested if !window.is_minimized().unwrap_or(false) && !elwt.exiting() => {
					if let Err(e) = app.render(window) {
						error!("rendering error: {}", e);
						elwt.exit();
					}
                }
                WindowEvent::Resized(size) => if size.width != 0 && size.height != 0 {
                    app.window_resized = true;
                }
                _ => {}
            }
        }
    }

    fn device_event(
        &mut self,
        _: &ActiveEventLoop,
        _: DeviceId,
        event: DeviceEvent,
    ) {
        self.input.process_device_event(&event);
    }

    fn new_events(&mut self, _: &ActiveEventLoop, _: StartCause) {
        self.input.step();
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        self.input.end_step();

        if self.input.close_requested() || self.input.destroyed() {
            event_loop.exit();
            return;
        }

		if let Some(app) = self.app.as_mut() {
			let now = Instant::now();
			let delta_time = self.last_frame_time
				.map(|t| now.duration_since(t).as_secs_f32())
				.unwrap_or(0.0);

            self.fps_accumulator += delta_time;
            self.fps_frame_count += 1;

            if self.fps_accumulator >= 1.0 && self.fps_frame_count > 0 {
                let fps = self.fps_frame_count as f32 / self.fps_accumulator;
                let ms = 1000.0 * self.fps_accumulator / self.fps_frame_count as f32;
                info!("{:.1} fps ({:.2} ms/frame)", fps, ms);

                self.fps_accumulator = 0.0;
                self.fps_frame_count = 0;
            }

			self.last_frame_time = Some(now);

            let ECSContext { world, schedule, .. } = &mut app.ecs_context;

            // ECS - update time then execute the schedule
            if let Some(mut ecs_time) = world.get_resource_mut::<Time>() {
                ecs_time.0 = delta_time;
            }

            if let Some(mut c_frame) = world.get_resource_mut::<CurrentFrame>() {
                c_frame.0 = app.frame;
            }

            schedule.run(world);

            //  cameras
			for (keycode, action) in &app.input_bindings.camera_bindings {
				if self.input.key_held(*keycode) {
					app.data.persistent.camera.process_keyboard(*action, delta_time);
				}
			}

            // TODO: change this to adapt to the type of camera & app
            if self.input.mouse_held(winit::event::MouseButton::Right) {
                let (dx, dy) = self.input.mouse_diff();
                app.data.persistent.camera.process_mouse_movement(dx, dy, Some((-89.0, 89.0)));
            }

            let zoom = self.input.scroll_diff().1;
            if zoom != 0.0 {
                app.data.persistent.camera.process_mouse_scroll(zoom);
            }
		} 

        if let Some(window) = self.window.as_mut() {
            window.request_redraw();
        }
    }
    
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window =
                event_loop.create_window(
                    Window::default_attributes()
                        .with_title("Vulkan App")
                        .with_inner_size(LogicalSize::new(1024, 768))   
                ).unwrap();
            
            self.app = Some(App::new(&window).unwrap());
			self.last_frame_time = Some(Instant::now());
            self.window = Some(window);
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(app) = self.app.as_mut() {
            unsafe { app.destroy() }; 
        }
    }
}