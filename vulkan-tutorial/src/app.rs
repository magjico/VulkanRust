use anyhow::{Result, anyhow};
use log::*;

use std::time::Instant;
use std::ptr::copy_nonoverlapping as memcpy;

use winit::dpi::LogicalSize;
use winit::application::ApplicationHandler;
use winit::window::{Window, WindowId};
use winit::event_loop::ActiveEventLoop;
use winit::event::{DeviceEvent, WindowEvent, DeviceId, StartCause};
use winit_input_helper::WinitInputHelper;

use vulkanalia::loader::{LIBRARY, LibloadingLoader};
use vulkanalia::window as vk_window;
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::{
    ExtDebugUtilsExtensionInstanceCommands,
    KhrSurfaceExtensionInstanceCommands,
    KhrSwapchainExtensionDeviceCommands,
};

use crate::constants::*;
use crate::setup::*;
use crate::gpu::*;
use crate::input::InputBindings;
use crate::render::{CommandRecorder, DescriptorLayouts, Descriptors, Swapchain, TextureData, TexturesStorage, UniformBufferObject, ColorAttachment, DepthAttachment};
use crate::resources::{GeometryBuffer, UniformBuffers, FrameSync,
                        create_uniform_buffers, create_command_buffers};
use crate::scene::{Camera, CameraBuilder, CurrentFrame, ECSContext, InstanceBuffer, LightBuffer, ModelRegistry, ModelsStorage, SkinningBuffer, Time};
use crate::math::{Vec3, Vec4};

//===================================================
// App Manager
//===================================================

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
                WindowEvent::RedrawRequested if !window.is_minimized().unwrap() && !elwt.exiting() => {
                    app.render(&window).unwrap();
                }
                WindowEvent::Resized(size) => if size.width != 0 && size.height != 0 {
                    app.resized = true;
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
			for (keycode, action) in &app.input_binding.camera_bindings {
				if self.input.key_held(*keycode) {
					app.data.camera_data.process_keyboard(*action, delta_time);
				}
			}

            // TODO: change this to adapt to the type of camera & app
            if self.input.mouse_held(winit::event::MouseButton::Right) {
                let (dx, dy) = self.input.mouse_diff();
                app.data.camera_data.process_mouse_movement(dx, dy, Some((-89.0, 89.0)));
            }

            let zoom = self.input.scroll_diff().1;
            if zoom != 0.0 {
                app.data.camera_data.process_mouse_scroll(zoom);
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
            
            self.app = Some(App::create(&window).unwrap());
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

//===================================================
// App
//===================================================

/// Vulkan app.
pub struct App {
    pub entry: Entry,
    pub instance: Instance,
    pub data: AppData,
    pub device: Device,
    pub ecs_context: ECSContext,
    pub frame: usize,
    pub resized: bool,
    pub start: Instant,
    pub models: usize,
    pub input_binding: InputBindings,
}

impl App {
    pub fn create(window: &Window) -> Result<Self> {
        // TODO: move all params into a param file.
        let mandatory_feats = vk::PhysicalDeviceFeatures::builder()
            .sampler_anisotropy(true)
            .build();

        let optional_feats = vk::PhysicalDeviceFeatures::builder()
            .build();

        let mandatory_queue_flags = vk::QueueFlags::GRAPHICS;

        // I - entry
        let loader = unsafe { LibloadingLoader::new(LIBRARY)? };
        let entry = unsafe {Entry::new(loader)}.map_err(|b| anyhow!("{}", b))?;

        // II - instance
        let (instance, messenger) = create_instance(window, &entry)?;

        // III - app-data
        // 1. surface
        let surface = unsafe { vk_window::create_surface(&instance, &window, &window)? };
        // 2. device
        let gpu_requirements = GPURequirements {
            mandatory_features:     mandatory_feats,
            optional_features:      optional_feats,
            mandatory_extensions:   DEVICE_EXTENSIONS.to_vec(),
            optional_extensions:    (&[] as &[vk::ExtensionName]).to_vec(),
            queue_flags:            mandatory_queue_flags,
        };

        let (mut device_data, device) = GPUContext::create(
            &entry,
            &instance,
            surface,
            &gpu_requirements,
            false,
        )?;

        // 3. ECS Context
        let mut ecs_context = init_ecs_context(&device, &instance, device_data.physical_device)?;

        // 4. swapchain
        let swapchain_data = Swapchain::new(
            window,
            &instance,
            &device,
            device_data.physical_device,
            surface,
            &mut device_data.queue_family_indices
        )?;

        // 5. command
        let command_data = CommandRecorder::new(
            &device,
            &mut device_data.queue_family_indices,
            &swapchain_data.vk_images
        )?;

        // 6. color
        let color_data = ColorAttachment::new(
            &instance,
            &device,
            device_data.physical_device,
            swapchain_data.vk_extent.width,
            swapchain_data.vk_extent.height,
            device_data.max_msaa_samples,
            swapchain_data.vk_format
        )?;

        // 7. depth
        let depth_data = DepthAttachment::new(
            &instance,
            &device,
            device_data.physical_device,
            swapchain_data.vk_extent.width,
            swapchain_data.vk_extent.height,
            device_data.max_msaa_samples,
        )?;

        // 8. pipeline
        let descriptor_layout_data = DescriptorLayouts::new(&device)?;

        let pipeline_data = Pipeline::new(
            &device,
            &[
                (VERT, vk::ShaderStageFlags::VERTEX),
                (FRAG, vk::ShaderStageFlags::FRAGMENT)
            ],
            swapchain_data.vk_extent,
            swapchain_data.vk_format,
            depth_data.vk_format,
            device_data.max_msaa_samples,
            &[
                descriptor_layout_data.global_set_layout,
                descriptor_layout_data.material_set_layout,
                descriptor_layout_data.skin_set_layout,
                descriptor_layout_data.instance_set_layout,
            ],
        )?;
        
		// 8 - 9. load .glb models and textures
		let mut models = ModelsStorage::new();
		let mut textures = TexturesStorage::new();
		let mut model_registry = ModelRegistry::new();

		load_gltf_models(
			&device,
			&instance,
			device_data.physical_device,
			command_data.setup_command_buffer,
			device_data.graphics_queue,
			&mut models,
			&mut textures,
			&mut model_registry
		)?;

		let default_texture = create_default_texture(
			&instance,
			&device,
			device_data.physical_device,
			command_data.setup_command_buffer,
			device_data.graphics_queue
		)?;

        // 10. buffers
        let geometry_buffer = GeometryBuffer::new(
            &instance,
            &device,
            device_data.physical_device,
            &models,
            command_data.setup_command_buffer,
            device_data.graphics_queue,
            // swapchain_data.vk_images.len(),
        )?;

        let uniform_buffers = UniformBuffers::new(
            &instance,
            &device,
            device_data.physical_device,
            swapchain_data.vk_images.len()
        )?;

		// 11. lights
        // TODO: implement multiple type of light
        let lights = create_default_lightning(&instance, &device, device_data.physical_device)?;

        // 12. descriptor
        let descriptor_data = Descriptors::new(
            &device,
            &ecs_context,
            &descriptor_layout_data,
            &uniform_buffers.vk_buffers,
            &textures,
            &default_texture,
            &models,
            &lights,
            swapchain_data.vk_images.len(),
        )?;
        
		// 13. ECS - final init + instance spawn
		ecs_context.world.insert_resource(models);
		ecs_context.world.insert_resource(Time(0.0));
		spawn_from_cesium_man_instances(&mut ecs_context.world, &model_registry)?;
        spawn_from_brain_stem_instance(&mut ecs_context.world, &model_registry)?;

        // 14. sync
        let sync_data = FrameSync::new(
            &device,
            MAX_FRAMES_IN_FLIGHT,
            swapchain_data.vk_images.len(),
        )?;

        // 15. Camera
        // TODO: support multiple cameras
        let mut camera = CameraBuilder::new()
            .position(Vec3::new(0.0,8.0, 4.0))
            .movement_speed(10.0)
            .mouse_sensitivity(0.02)
            .build();

        camera.look_at(Vec3::new(0.0, 0.0, 0.75), None);


        let data = AppData {
            surface,
            device_data,
            swapchain_data,
            descriptor_layout_data,
            pipeline_data,
            geometry_buffer,
            uniform_buffers,
            descriptor_data,
            command_data,
            sync_data,
            textures_data: textures,
            depth_data,
            color_data,
            camera_data: camera,
            light_data: lights,
            messenger,
            default_texture
        };

        // IV - inputs
        // TODO: make InputBindings able to have None so we can have a default if we can't parse.
        let input_binding = InputBindings::bind_from_file(&INPUT_PATH)?;

        Ok( Self {
            entry,
            instance,
            data,
            device,
            ecs_context,
            frame: 0,
            resized: false,
            start: Instant::now(),
            models: 1,
            input_binding,
        })
    }

    /// Destroy Vulkan app.
    #[rustfmt::skip]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn destroy(&mut self) {
        self.device.device_wait_idle().unwrap();

        self.destroy_swapchain();

        // Destroy Appdata
		if let Some(skinning_buffer) = self.ecs_context.world.get_resource::<SkinningBuffer>() {
			skinning_buffer.destroy(&self.device);
		}
        if let Some(instance_buffer) = self.ecs_context.world.get_resource::<InstanceBuffer>() {
            instance_buffer.destroy(&self.device);
        }

        self.data.textures_data.destroy(&self.device);
        self.data.default_texture.destroy(&self.device);
        self.data.light_data.destroy(&self.device);
        
        self.device.destroy_descriptor_pool(self.data.descriptor_data.descriptor_pool, None);
        self.data.descriptor_layout_data.destroy(&self.device);
        self.data.sync_data.destroy(&self.device);
        self.data.geometry_buffer.destroy(&self.device);
        self.data.command_data.destroy(&self.device);

        // Destroy App
        self.device.destroy_device(None);
        self.instance.destroy_surface_khr(self.data.surface, None);
        if let Some(messenger) = self.data.messenger {
            self.instance.destroy_debug_utils_messenger_ext(messenger, None);
        }

        self.instance.destroy_instance(None);
    }

    /// Destroys the parts of our Vulkan app related to the swapchain.
    #[rustfmt::skip]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn destroy_swapchain(&mut self) {
        self.data.uniform_buffers.destroy(&self.device);
        
        self.data.depth_data.destroy(&self.device);
		self.data.color_data.destroy(&self.device);
        self.data.pipeline_data.destroy(&self.device);
        self.data.swapchain_data.destroy(&self.device);
    }

    /// Recreates the swapchain for our Vulkan app.
    #[rustfmt::skip]
    fn recreate_swapchain(&mut self, window: &Window) -> Result<()> {
        unsafe {
            self.device.device_wait_idle()?;
            self.destroy_swapchain();
        };

        self.data.swapchain_data = Swapchain::new(
            window,
            &self.instance,
            &self.device,
            self.data.device_data.physical_device,
            self.data.surface,
            &mut self.data.device_data.queue_family_indices,
        )?;

        self.data.color_data = ColorAttachment::new(
            &self.instance,
            &self.device,
            self.data.device_data.physical_device,
            self.data.swapchain_data.vk_extent.width,
            self.data.swapchain_data.vk_extent.height,
            self.data.device_data.max_msaa_samples,
            self.data.swapchain_data.vk_format,
        )?;

        self.data.depth_data = DepthAttachment::new(
            &self.instance,
            &self.device,
            self.data.device_data.physical_device,
            self.data.swapchain_data.vk_extent.width,
            self.data.swapchain_data.vk_extent.height,
            self.data.device_data.max_msaa_samples,
        )?;

        self.data.pipeline_data = Pipeline::new(
            &self.device,
            &[
                (VERT, vk::ShaderStageFlags::VERTEX),
                (FRAG, vk::ShaderStageFlags::FRAGMENT)
            ],
            self.data.swapchain_data.vk_extent,
            self.data.swapchain_data.vk_format,
            self.data.depth_data.vk_format,
            self.data.device_data.max_msaa_samples,
            &[
                self.data.descriptor_layout_data.global_set_layout,
                self.data.descriptor_layout_data.material_set_layout,
                self.data.descriptor_layout_data.skin_set_layout,
                self.data.descriptor_layout_data.instance_set_layout,
            ],
        )?;

        (self.data.uniform_buffers.vk_buffers, self.data.uniform_buffers.vk_buffers_memories) = create_uniform_buffers(
            &self.instance,
            &self.device,
            self.data.device_data.physical_device,
            self.data.swapchain_data.vk_images.len()
        )?;

        self.data.descriptor_data.update_global_descriptor_sets(
            &self.device,
            &self.data.uniform_buffers.vk_buffers
        );

        (self.data.command_data.primary_command_buffers, self.data.command_data.secondary_command_buffers) = create_command_buffers(
            &self.device,
            &self.data.command_data.frames_pools
        )?;

        self.data.sync_data.images_in_flight.resize(
            self.data.swapchain_data.vk_images.len(),
            vk::Fence::null()
        );

        Ok(())
    }

    /// Renders a frame for our Vulkan app.
    pub fn render(&mut self, window: &Window) -> Result<()> {
        let in_flight_fence = self.data.sync_data.in_flight_fences[self.frame];

        unsafe {self.device.wait_for_fences(&[in_flight_fence], true, u64::MAX)?};

        let result = unsafe { self.device.acquire_next_image_khr(
            self.data.swapchain_data.vk_swapchain,
            u64::MAX,
            self.data.sync_data.image_available_semaphores[self.frame],
            vk::Fence::null(),
        )};

        let image_index = match result {
            Ok((image_index, _)) => image_index as usize,
            Err(vk::ErrorCode::OUT_OF_DATE_KHR) => return self.recreate_swapchain(window),
            Err(e) => return Err(anyhow!(e)),
        };

        let image_in_flight = self.data.sync_data.images_in_flight[image_index];
        if !image_in_flight.is_null() {
            unsafe { self.device.wait_for_fences(&[image_in_flight], true, u64::MAX)? };
        }

        self.data.sync_data.images_in_flight[image_index] = in_flight_fence;

        // Update commands buffers
        self.data.command_data.record(
            &self.device,
			&mut self.ecs_context,
            &self.data.pipeline_data,
            &self.data.geometry_buffer,
            &self.data.swapchain_data,
            &self.data.color_data,
            &self.data.depth_data,
            &self.data.descriptor_data,
            self.data.device_data.max_msaa_samples,
            image_index,
            self.frame
        )?;

        // Update UBO
        self.update_uniform_buffer(image_index)?;

        let wait_semaphores = &[self.data.sync_data.image_available_semaphores[self.frame]];
        let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = &[self.data.command_data.primary_command_buffers[image_index]];
        let signal_semaphores = &[self.data.sync_data.render_finished_semaphores[self.frame]];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores);

        unsafe {
            self.device.reset_fences(&[in_flight_fence])?;

            self.device
                .queue_submit(self.data.device_data.graphics_queue, &[submit_info], in_flight_fence)?;
        }

        let swapchains = &[self.data.swapchain_data.vk_swapchain];
        let image_indices = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);

        let result = unsafe { self.device.queue_present_khr(self.data.device_data.present_queue, &present_info) };
        let changed = result == Ok(vk::SuccessCode::SUBOPTIMAL_KHR) || result == Err(vk::ErrorCode::OUT_OF_DATE_KHR);
        if self.resized || changed {
            self.resized = false;
            self.recreate_swapchain(window)?;
        } else if let Err(e) = result {
            return Err(anyhow!(e));
        }

        self.frame = (self.frame + 1) % MAX_FRAMES_IN_FLIGHT;

        Ok(())
    }

    fn update_uniform_buffer(&self, image_index: usize) -> Result<()> {
        let view = self.data.camera_data.get_view_matrix();

        let proj = CORRECTION * self.data.camera_data.get_projection_matrix(
            self.data.swapchain_data.vk_extent.width as f32 / self.data.swapchain_data.vk_extent.height as f32,
            Some(0.1),
            Some(1000.0)
        );

        let cam_pos = {
            let pos = self.data.camera_data.get_position();
            Vec4::new(pos.x, pos.y, pos.z, 1.0)
        };

        let ubo = UniformBufferObject {
            view,
            proj,
            cam_pos,
            exposure: 4.5,
            gamma: 2.2,
            prefiltered_cube_mip_levels: 1.0,
            scale_ibl_ambient: 1.0
        };

        unsafe {
            let memory = self.device.map_memory(
                self.data.uniform_buffers.vk_buffers_memories[image_index],
                0,
                size_of::<UniformBufferObject>() as u64,
                vk::MemoryMapFlags::empty()
            )?;


            memcpy(&ubo, memory.cast(), 1);
            self.device.unmap_memory(self.data.uniform_buffers.vk_buffers_memories[image_index]);
        }

        Ok(())
    }
}

//===================================================
// App data
//===================================================

/// The Vulkan handles and associated properties used by our Vulkan app.
#[derive(Debug)]
pub struct AppData {
    // Surface
    pub surface: vk::SurfaceKHR,
    // Physical Device / Logical Device
    pub device_data: GPUContext,
    // Swapchain
    pub swapchain_data: Swapchain,
    // Pipeline
    pub descriptor_layout_data: DescriptorLayouts,
    pub pipeline_data: Pipeline,
    // Textures
	pub textures_data: TexturesStorage,
    // Buffers
    pub geometry_buffer: GeometryBuffer,
    pub uniform_buffers: UniformBuffers,
    // Descriptor
    pub descriptor_data: Descriptors,
    // Command Buffers
    pub command_data: CommandRecorder,
    // Sync Objects
    pub sync_data: FrameSync,
    // Depth
    pub depth_data: DepthAttachment,
	// Render target (now only use for MSAA)
	pub color_data: ColorAttachment,
    // Camera
    pub camera_data: Camera,
    // Lights
    pub light_data: LightBuffer,
    // Debug
    pub messenger: Option<vk::DebugUtilsMessengerEXT>,

    // app-data const
    pub default_texture: TextureData,
}