use anyhow::{Result, anyhow};
use log::*;

use std::collections::HashMap;
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
use vulkanalia::vk::{ExtDebugUtilsExtensionInstanceCommands, KhrDynamicRenderingExtensionDeviceCommands,
                    KhrSurfaceExtensionInstanceCommands, KhrSwapchainExtensionDeviceCommands,
                    KhrSynchronization2ExtensionDeviceCommands};

use crate::constants::*;
use crate::setup::*;
use crate::gpu::*;
use crate::input::InputBindings;
use crate::type_safety::{MaterialId, MaterialSetId, MeshOffset, ModelId, NodeId};
use crate::render::{InstanceData, PushConstants, TextureData, TexturesStorage, UniformBufferObject, create_color_objects, create_depth_objects, create_swapchain, create_swapchain_image_views};
use crate::resources::{create_command_pool, create_command_pools, create_setup_command_buffer,
                        create_interleaved_buffer, create_uniform_buffers, create_command_buffers,
                        destroy_buffers};
use crate::scene::{Camera, CameraBuilder, CurrentFrame, ECSContext, InstanceBuffer, LightBuffer, Material, ModelRegistry, ModelsStorage, SkinningBuffer, Time};
use crate::math::{Vec3, Vec4, Mat4};

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
        let physical_device = pick_best_physical_device(
            &instance,
            surface,
            &mandatory_feats,
            &optional_feats,
            &DEVICE_EXTENSIONS,
            &[] as &[vk::ExtensionName],
            mandatory_queue_flags,
            false
        )?;
        let msaa_samples = get_max_msaa_samples(&instance, physical_device);
        let mut queue_family_indices = QueueFamilyIndices::create(
            &instance,
            physical_device,
            surface
        )?;
        let (device, graphics_queue, present_queue) = create_logical_device(
            &entry,
            &instance,
            physical_device,
            &mut queue_family_indices,
            VALIDATION_ENABLED,
            VALIDATION_LAYER,
            DEVICE_EXTENSIONS
        )?;

        let mut device_data= DeviceData::create(
            physical_device,
            graphics_queue,
            present_queue,
            msaa_samples,
            queue_family_indices
        );

        // 3. ECS Context
        let mut ecs_context = init_ecs_context(&device, &instance, physical_device)?;

        // 4. swapchain
        let swapchain_data = SwapchainData::create(
            window,
            &instance,
            &device,
            physical_device,
            surface,
            &mut device_data.queue_family_indices
        )?;

        // 5. command
        let command_data = CommandData::create(
            &device,
            &mut device_data.queue_family_indices,
            &swapchain_data.swapchain_images
        )?;

        // 6. color
        let color_data = ColorData::create(
            &instance,
            &device,
            physical_device,
            swapchain_data.swapchain_extent.width,
            swapchain_data.swapchain_extent.height,
            msaa_samples,
            swapchain_data.swapchain_format
        )?;

        // 7. depth
        let depth_data = DepthData::create(
            &instance,
            &device,
            physical_device,
            swapchain_data.swapchain_extent.width,
            swapchain_data.swapchain_extent.height,
            msaa_samples,
        )?;

        // 8. pipeline
        let descriptor_layout_data = DescriptorLayoutData::create(&device)?;

        let pipeline_data = Pipeline::new(
            &device,
            &[(&[VERT, FRAG], &[vk::ShaderStageFlags::VERTEX, vk::ShaderStageFlags::FRAGMENT])],
            swapchain_data.swapchain_extent,
            swapchain_data.swapchain_format,
            depth_data.depth_format,
            msaa_samples,
            &descriptor_layout_data,
        )?;
        
		// 8 - 9. load .glb models and textures
		let mut models = ModelsStorage::new();
		let mut textures = TexturesStorage::new();
		let mut model_registry = ModelRegistry::new();

		load_gltf_models(
			&device,
			&instance,
			physical_device,
			command_data.setup_command_buffer,
			graphics_queue,
			&mut models,
			&mut textures,
			&mut model_registry
		)?;

		let default_texture = create_default_texture(
			&instance,
			&device,
			physical_device,
			command_data.setup_command_buffer,
			graphics_queue
		)?;

        // 10. buffers
        let buffers_data = BuffersData::create(
            &instance,
            &device,
            physical_device,
            &models,
            command_data.setup_command_buffer,
            graphics_queue,
            swapchain_data.swapchain_images.len(),
        )?;

		// 11. lights
        // TODO: implement multiple type of light
        let lights = create_default_lightning(&instance, &device, physical_device)?;

        // 12. descriptor
        let descriptor_data = DescriptorData::create(
            &device,
            &ecs_context,
            &descriptor_layout_data,
            &buffers_data.uniform_buffers,
            &textures,
            &default_texture,
            &models,
            &lights,
            swapchain_data.swapchain_images.len(),
        )?;
        
		// 13. ECS - final init + instance spawn
		ecs_context.world.insert_resource(models);
		ecs_context.world.insert_resource(Time(0.0));
		spawn_from_cesium_man_instances(&mut ecs_context.world, &model_registry)?;
        spawn_from_brain_stem_instance(&mut ecs_context.world, &model_registry)?;

        // 14. sync
        let sync_data = SyncData::create(
            &device,
            MAX_FRAMES_IN_FLIGHT,
            swapchain_data.swapchain_images.len(),
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
            buffers_data,
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
        destroy_buffers(&self.device, &[self.data.buffers_data.interleaved_buffer], &[self.data.buffers_data.interleaved_buffer_memory]);
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
        // self.device.destroy_descriptor_pool(self.data.descriptor_data.descriptor_pool, None);
        destroy_buffers(
            &self.device,
            &self.data.buffers_data.uniform_buffers,
            &self.data.buffers_data.uniform_buffers_memory
        );

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

        self.data.swapchain_data = SwapchainData::create(
            window,
            &self.instance,
            &self.device,
            self.data.device_data.physical_device,
            self.data.surface,
            &mut self.data.device_data.queue_family_indices,
        )?;

        self.data.color_data = ColorData::create(
            &self.instance,
            &self.device,
            self.data.device_data.physical_device,
            self.data.swapchain_data.swapchain_extent.width,
            self.data.swapchain_data.swapchain_extent.height,
            self.data.device_data.msaa_samples,
            self.data.swapchain_data.swapchain_format,
        )?;

        self.data.depth_data = DepthData::create(
            &self.instance,
            &self.device,
            self.data.device_data.physical_device,
            self.data.swapchain_data.swapchain_extent.width,
            self.data.swapchain_data.swapchain_extent.height,
            self.data.device_data.msaa_samples,
        )?;

        self.data.pipeline_data = PipelineData::create(
            &self.device,
            self.data.swapchain_data.swapchain_format,
            self.data.swapchain_data.swapchain_extent,
            self.data.depth_data.depth_format,
            &self.data.descriptor_layout_data,
            self.data.device_data.msaa_samples,
            VERT,
            FRAG,
        )?;

        (self.data.buffers_data.uniform_buffers, self.data.buffers_data.uniform_buffers_memory) = create_uniform_buffers(
            &self.instance,
            &self.device,
            self.data.device_data.physical_device,
            self.data.swapchain_data.swapchain_images.len()
        )?;

        self.data.descriptor_data.update_global_descriptor_set(
            &self.device,
            &self.data.buffers_data.uniform_buffers
        )?;

        (self.data.command_data.command_buffers, self.data.command_data.secondary_command_buffers) = create_command_buffers(
            &self.device,
            &self.data.command_data.command_pools
        )?;

        self.data.sync_data.images_in_flight.resize(
            self.data.swapchain_data.swapchain_images.len(),
            vk::Fence::null()
        );

        Ok(())
    }

    /// Renders a frame for our Vulkan app.
    pub fn render(&mut self, window: &Window) -> Result<()> {
        let in_flight_fence = self.data.sync_data.in_flight_fences[self.frame];

        unsafe {self.device.wait_for_fences(&[in_flight_fence], true, u64::MAX)?};

        let result = unsafe { self.device.acquire_next_image_khr(
            self.data.swapchain_data.swapchain,
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
        self.data.command_data.update_command_buffer(
            &self.device,
			&mut self.ecs_context,
            &self.data.pipeline_data,
            &self.data.buffers_data,
            &self.data.swapchain_data,
            &self.data.color_data,
            &self.data.depth_data,
            &self.data.descriptor_data,
            self.data.device_data.msaa_samples,
            image_index,
            self.frame
        )?;

        // Update UBO
        self.update_uniform_buffer(image_index)?;

        let wait_semaphores = &[self.data.sync_data.image_available_semaphores[self.frame]];
        let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = &[self.data.command_data.command_buffers[image_index]];
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

        let swapchains = &[self.data.swapchain_data.swapchain];
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
            self.data.swapchain_data.swapchain_extent.width as f32 / self.data.swapchain_data.swapchain_extent.height as f32,
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
                self.data.buffers_data.uniform_buffers_memory[image_index],
                0,
                size_of::<UniformBufferObject>() as u64,
                vk::MemoryMapFlags::empty()
            )?;


            memcpy(&ubo, memory.cast(), 1);
            self.device.unmap_memory(self.data.buffers_data.uniform_buffers_memory[image_index]);
        }

        Ok(())
    }
}

//===================================================
// App data
//===================================================

/// The Vulkan handles and associated properties used by our Vulkan app.
#[derive(Clone, Debug)]
pub struct AppData {
    // Surface
    pub surface: vk::SurfaceKHR,
    // Physical Device / Logical Device
    pub device_data: DeviceData,
    // Swapchain
    pub swapchain_data: SwapchainData,
    // Pipeline
    pub descriptor_layout_data: DescriptorLayoutData,
    pub pipeline_data: PipelineData,
    // Textures
	pub textures_data: TexturesStorage,
    // Buffers
    pub buffers_data: BuffersData,
    // Descriptor
    pub descriptor_data: DescriptorData,
    // Command Buffers
    pub command_data: CommandData,
    // Sync Objects
    pub sync_data: SyncData,
    // Depth
    pub depth_data: DepthData,
	// Render target (now only use for MSAA)
	pub color_data: ColorData,
    // Camera
    pub camera_data: Camera,
    // Lights
    pub light_data: LightBuffer,
    // Debug
    pub messenger: Option<vk::DebugUtilsMessengerEXT>,

    // app-data const
    pub default_texture: TextureData,
}

#[derive(Clone, Debug)]
pub struct DeviceData {
    pub physical_device: vk::PhysicalDevice,
    pub graphics_queue: vk::Queue,
    pub present_queue: vk::Queue,
	pub msaa_samples: vk::SampleCountFlags,
    // QueueFamily
    pub queue_family_indices: QueueFamilyIndices,
}

impl DeviceData {
    pub fn create(
        physical_device: vk::PhysicalDevice,
        graphics_queue: vk::Queue,
        present_queue: vk::Queue,
        msaa_samples: vk::SampleCountFlags,
        // QueueFamily
        queue_family_indices: QueueFamilyIndices,
    ) -> Self {
        Self {
            physical_device,
            graphics_queue,
            present_queue,
            msaa_samples,
            queue_family_indices,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SwapchainData {
    pub swapchain_format: vk::Format,
    pub swapchain_extent: vk::Extent2D,
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_image_views: Vec<vk::ImageView>,
}

impl SwapchainData {
    pub fn create(
        window: &Window,
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
        queue_family_indices: &mut QueueFamilyIndices
    ) -> Result<Self> {
        let (swapchain, swapchain_format, swapchain_extent, swapchain_images) = create_swapchain(
            window,
            instance,
            device,
            physical_device,
            surface,
            queue_family_indices
        )?;

        let swapchain_image_views = create_swapchain_image_views(
            &device,
            &swapchain_images,
            swapchain_format
        )?;

        Ok(Self {
            swapchain_format,
            swapchain_extent,
            swapchain,
            swapchain_images,
            swapchain_image_views
        })
    }

    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn destroy(&self, device: &Device) {
        self.swapchain_image_views.iter().for_each(|v| device.destroy_image_view(*v, None));
        device.destroy_swapchain_khr(self.swapchain, None);
    }
}

#[derive(Clone, Debug)]
pub struct BuffersData {
    pub interleaved_buffer: vk::Buffer,
    pub interleaved_buffer_memory: vk::DeviceMemory,
    pub interleaved_offset: u64,
    pub mesh_offsets: HashMap<(ModelId, NodeId), MeshOffset>, // key: (model_id, node_id) -> value: (vert_offset, index_offset) 
    pub uniform_buffers: Vec<vk::Buffer>,
    pub uniform_buffers_memory: Vec<vk::DeviceMemory>,
}

impl BuffersData {
    pub fn create(
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        models: &ModelsStorage,
        setup_command_buffer: vk::CommandBuffer,
        graphics_queue: vk::Queue,
        images_count: usize,
    ) -> Result<Self> {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut mesh_offsets = HashMap::new();

        for (model_id, model) in models.iter().enumerate() {
            let model_id = ModelId(model_id);

            for (node_idx, node) in model.graph.iter().enumerate() {
                if let Some(mesh) = &node.value.mesh {
                    mesh_offsets.insert(
                        (model_id, NodeId(node_idx)),
                        MeshOffset {vertex_offset: vertices.len() as u32, first_index: indices.len() as u32 }
                    );

                    vertices.extend_from_slice(&mesh.vertices);
                    indices.extend_from_slice(&mesh.indices);
                }
            }
        }

        let (interleaved_buffer, interleaved_buffer_memory, interleaved_offset) = create_interleaved_buffer(
            instance,
            device,
            physical_device,
            &vertices,
            &indices,
            setup_command_buffer,
            graphics_queue,
        )?;

        let (uniform_buffers, uniform_buffers_memory) = create_uniform_buffers(
            instance,
            device,
            physical_device,
            images_count,
        )?;

        Ok(Self {
            interleaved_buffer,
            interleaved_buffer_memory,
            interleaved_offset,
            mesh_offsets,
            uniform_buffers,
            uniform_buffers_memory,
        })
    }
}

#[derive(Clone, Debug)]
pub struct DescriptorLayoutData {
    pub global_set_layout: vk::DescriptorSetLayout,
    pub material_set_layout: vk::DescriptorSetLayout,
    pub skin_set_layout: vk::DescriptorSetLayout,
    pub instance_set_layout: vk::DescriptorSetLayout,
}

impl DescriptorLayoutData {
    pub fn create(
        device: &Device,
    ) -> Result<Self> {
        let global_set_layout = create_global_descriptor_set_layout(device)?;
        let material_set_layout = create_material_descriptor_set_layout(device)?;
        let skin_set_layout = create_skinning_descriptor_set_layout(device)?;
        let instance_set_layout = create_instance_descriptor_set_layout(device)?;

        Ok(Self { global_set_layout, material_set_layout, skin_set_layout, instance_set_layout })
    }

    #[rustfmt::skip]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn destroy(&mut self, device: &Device) {
        device.destroy_descriptor_set_layout(self.global_set_layout, None);
        device.destroy_descriptor_set_layout(self.material_set_layout, None);
        device.destroy_descriptor_set_layout(self.skin_set_layout, None);
        device.destroy_descriptor_set_layout(self.instance_set_layout, None);
    }
}

#[derive(Clone, Debug)]
pub struct DescriptorData {
    pub descriptor_pool: vk::DescriptorPool,
    pub global_descriptor_sets: Vec<vk::DescriptorSet>,
    pub material_descriptor_sets: Vec<vk::DescriptorSet>,
    pub skinning_descriptor_set: vk::DescriptorSet,
	pub instance_descriptor_set: vk::DescriptorSet,
}

impl DescriptorData {
    pub fn  create(
        device: &Device,
        ecs_context: &ECSContext,
        descriptor_layout_data: &DescriptorLayoutData,
        uniform_buffers: &[vk::Buffer],
        textures: &TexturesStorage,
        default_texture: &TextureData,
		models: &ModelsStorage,
        light_buffer: &LightBuffer,
        images_count: usize,
    ) -> Result<Self> {
		let materials: Vec<&Material> = models.iter()
			.flat_map(|model| model.materials_iter())
			.collect();
        let materials_count = materials.len();

        let skinning_buffer = ecs_context.world.get_resource::<SkinningBuffer>()
            .ok_or_else(|| anyhow!("Skinning Buffer not found in ecs_context.world"))?;

		let instance_buffer = ecs_context.world.get_resource::<InstanceBuffer>()
			.ok_or_else(|| anyhow!("Instance Buffer not found in ecs_context.world"))?;

        let descriptor_pool = create_descriptor_pool(
            device,
            images_count		as u32,
            materials_count		as u32,
        )?;
        
        let global_descriptor_sets = create_global_descriptor_sets(
            device,
            images_count,
            descriptor_layout_data.global_set_layout,
            descriptor_pool,
            uniform_buffers,
            light_buffer,
        )?;

        let material_descriptor_sets = create_material_descriptor_sets(
            device,
            descriptor_layout_data.material_set_layout,
            descriptor_pool,
            &materials,
            textures,
            default_texture
        )?;

        debug!("descriptor points to buffer {:?}", skinning_buffer.buffer);
        let skinning_descriptor_set = create_skinning_descriptor_set(
            device,
            descriptor_layout_data.skin_set_layout,
            descriptor_pool,
            skinning_buffer,
        )?;

		let instance_descriptor_set = create_instance_descriptor_set(
			device,
			descriptor_layout_data.instance_set_layout,
			descriptor_pool,
			instance_buffer
		)?;

        Ok(Self { descriptor_pool, global_descriptor_sets, material_descriptor_sets, skinning_descriptor_set, instance_descriptor_set })
    }

    pub fn update_global_descriptor_set(
        &mut self,
        device: &Device,
        uniform_buffers: &[vk::Buffer],
    ) -> Result<()> {
        for i in 0..self.global_descriptor_sets.len() {
            let buffer_info = &[*vk::DescriptorBufferInfo::builder()
                .buffer(uniform_buffers[i])
                .offset(0)
                .range(size_of::<UniformBufferObject>() as u64)];

            let ubo_write = vk::WriteDescriptorSet::builder()
                .dst_set(self.global_descriptor_sets[i])
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(buffer_info);

            unsafe { device.update_descriptor_sets(&[ubo_write], &[] as &[vk::CopyDescriptorSet]); }
        }   

        Ok(())
    }
}

#[derive(Clone, Debug)]
struct EntityInstance {
    model_id:       ModelId,
    model:          Mat4,
    ssbo_offset:    u32,
}

#[derive(Clone, Debug)]
pub struct DrawItem {
    material_set_id:    MaterialSetId,
    model_id:           ModelId,
    material_id:        Option<MaterialId>,
    node_matrix:        Mat4,
    first_index:        u32,
    vertex_offset:      u32,
    index_count:        u32,
    instance_first:     u32,
    instance_count:     u32,
}

#[derive(Clone, Debug)]
pub struct CommandData {
    pub command_pool: vk::CommandPool,
    pub command_pools: Vec<vk::CommandPool>,
    pub command_buffers: Vec<vk::CommandBuffer>,
    pub secondary_command_buffers: Vec<vk::CommandBuffer>,
    pub setup_command_buffer: vk::CommandBuffer,

    draw_list:			Vec<DrawItem>,
    instance_data:		Vec<InstanceData>,
    sorted_entities:	Vec<EntityInstance>,
}

impl CommandData {
    pub fn create(
        device: &Device,
        queue_family_indices: &mut QueueFamilyIndices,
        swapchain_images: &[vk::Image]
    ) -> Result<Self> {
        let command_pool = create_command_pool(device, queue_family_indices)?;
        let command_pools = create_command_pools(
            device,
            queue_family_indices,
            swapchain_images.len(),
        )?;
        let setup_command_buffer = create_setup_command_buffer(device, command_pool)?;
        let (command_buffers, secondary_command_buffers) = create_command_buffers(
            device,
            &command_pools
        )?;

        Ok(Self {
            command_pool,
            command_pools,
            command_buffers,
            secondary_command_buffers,
            setup_command_buffer,

            draw_list:			Vec::new(),
            instance_data:		Vec::new(),
			sorted_entities:	Vec::new(),
        })
    }

    #[rustfmt::skip]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn destroy(&mut self, device: &Device) {
        self.command_pools.iter().for_each(|c| device.destroy_command_pool(*c, None));
        device.free_command_buffers(self.command_pool, &[self.setup_command_buffer]);
        device.destroy_command_pool(self.command_pool, None);
    }

    pub fn update_command_buffer(
        &mut self,
        device: &Device,
        ecs_context: &mut ECSContext,
        pipeline_data: &PipelineData,
        buffers_data: &BuffersData,
        swapchain_data: &SwapchainData,
        color_data: &ColorData,
        depth_data: &DepthData,
        descriptor_data: &DescriptorData,
        msaa_samples: vk::SampleCountFlags,
        image_index: usize,
        frame_index: usize,
    ) -> Result<()> {
        // Pool
        let command_pool = self.command_pools[image_index];
        unsafe { device.reset_command_pool(command_pool, vk::CommandPoolResetFlags::empty())? };

        // Commands
        let command_buffer = self.command_buffers[image_index];

        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe { device.begin_command_buffer(command_buffer, &info)? };

        let color_clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };

        let depth_clear_value = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        };

        let render_area = vk::Rect2D::builder()
            .offset(vk::Offset2D::default())
            .extent(swapchain_data.swapchain_extent);

        let color_attachment = vk::RenderingAttachmentInfo::builder()
            .image_view(color_data.color_image_view)
            .image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .resolve_mode(vk::ResolveModeFlags::AVERAGE)
            .resolve_image_view(swapchain_data.swapchain_image_views[image_index])
            .resolve_image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
            .clear_value(color_clear_value);

        let depth_attachment = vk::RenderingAttachmentInfo::builder()
            .image_view(depth_data.depth_image_view)
            .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(depth_clear_value);

        let rendering_info = vk::RenderingInfo::builder()
            .flags(vk::RenderingFlagsKHR::CONTENTS_SECONDARY_COMMAND_BUFFERS)
            .render_area(render_area)
            .layer_count(1)
            .color_attachments(std::slice::from_ref(&color_attachment))
            .depth_attachment(&depth_attachment);

        Self::transition_for_render(
            device,
            swapchain_data.swapchain_images[image_index],
            command_buffer
        );

        unsafe { device.cmd_begin_rendering_khr(command_buffer, &rendering_info); }

        let secondary_command_buffer = self.record_secondary_command_buffer(
            device,
            ecs_context,
            pipeline_data,
            buffers_data,
            &[swapchain_data.swapchain_format],
            depth_data,
            descriptor_data,
            msaa_samples,
            image_index,
            frame_index
        )?;

        unsafe { 
            device.cmd_execute_commands(command_buffer, &[secondary_command_buffer]);
            device.cmd_end_rendering_khr(command_buffer);
        };

        Self::transition_for_present(
            device,
            swapchain_data.swapchain_images[image_index],
            command_buffer
        );

        unsafe { device.end_command_buffer(command_buffer)? };

        Ok(())
    }
    
    /// record draws inside a unique secondary command buffer
    fn record_secondary_command_buffer(
        &mut self,
        device: &Device,
        ecs_context: &mut ECSContext,
        pipeline_data: &PipelineData,
        buffers_data: &BuffersData,
        swapchain_formats: &[vk::Format],
        depth_data: &DepthData,
        descriptor_data: &DescriptorData,
        msaa_samples: vk::SampleCountFlags,
        image_index: usize,
        frame_index: usize
    ) -> Result<vk::CommandBuffer> {
        // TODO: do a refactoring on the whole CommandData structure and functions
		// TODO: instance_data, draw_list and sorted_entities should be define inside record_secondary_command_buffer not inside the struct CommandData
        self.instance_data.clear();
        self.draw_list.clear();
		self.sorted_entities.clear();

		let models = ecs_context.world.get_resource::<ModelsStorage>()
            .ok_or_else(|| anyhow!("ModelsStorage not found"))?;

		// Step 1 - Draw Sorting
		// 1.a - gather entities, sorted by model
		self.sorted_entities.extend(
			ecs_context.cached_renderable_query
				.iter(&ecs_context.world)
				.map(|(global, mesh_handle, skeleton)| {
					let ssbo_offset = skeleton
						.map(|s| s.ssbo_offset + (frame_index * FRAME_STRIDE) as u32)
						.unwrap_or(PushConstants::NO_SKIN);

					EntityInstance {
						model_id: mesh_handle.model_id,
						model: global.0,
						ssbo_offset
					}
				})	
		);

		self.sorted_entities.sort_unstable_by_key(|e| e.model_id.0);

		// 1.b - instance data + model ranges
		let mut model_ranges: Vec<(ModelId, u32, u32)> = Vec::new();

		for entity in &self.sorted_entities {
			match model_ranges.last_mut() {
				Some((last_id, _, count)) if *last_id == entity.model_id => *count += 1,
				_ => model_ranges.push((entity.model_id, self.instance_data.len() as u32, 1)),
			}

			self.instance_data.push(
				InstanceData {
					model: entity.model,
					ssbo_offset: entity.ssbo_offset,
					_padding: [0; 3],
				}
			)
		}

		// 1.c - one DrawItem by (model, node, primitive)
		for (model_id, instance_first, instance_count) in &model_ranges {
			let model = models.get_model(*model_id);
			let material_offset = models.get_material_offset(*model_id);

			for (i, node) in model.graph.iter().enumerate() {
				let Some(mesh) = &node.value.mesh else { continue };

				let node_id = NodeId(i);
				let node_matrix = model.get_global_matrix_of(node_id);
				let mesh_offset = *buffers_data.mesh_offsets
					.get(&(*model_id, node_id))
					.unwrap_or_else(|| panic!("no mesh offset for ({:?}, {:?})", model_id, node_id));

				for primitive in &mesh.primitives {
					self.draw_list.push(DrawItem {
						material_set_id: primitive.material_id
							.map(|id| id.to_set_id(material_offset))
							.unwrap_or(MaterialSetId(0)),
						model_id: *model_id,
						material_id: primitive.material_id, 
						node_matrix,
						first_index: mesh_offset.first_index + primitive.first_index,
						vertex_offset: mesh_offset.vertex_offset,
						index_count: primitive.index_count,
						instance_first: *instance_first,
						instance_count: *instance_count
					});
				}
			}
		}

        self.draw_list.sort_unstable_by_key(|draw_item| draw_item.material_set_id);

		// 1.d - Upload instance data
		let instance_buffer = ecs_context.world.get_resource::<InstanceBuffer>()
			.ok_or_else(|| anyhow!("InstanceBuffer not found in world"))?;
		instance_buffer.write(frame_index, &self.instance_data);

        // Step 2 - Draw Binding with state tracking
        let command_buffer = self.secondary_command_buffers[image_index];

        let mut inheritance_rendering_info = vk::CommandBufferInheritanceRenderingInfo::builder()
            .color_attachment_formats(swapchain_formats)
            .depth_attachment_format(depth_data.depth_format)
            .rasterization_samples(msaa_samples);

        let inheritance_info = vk::CommandBufferInheritanceInfo::builder()
            .push_next(&mut inheritance_rendering_info);

        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE)
            .inheritance_info(&inheritance_info);


        unsafe {
            device.begin_command_buffer(command_buffer, &info)?;

            device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline_data.pipeline);
            device.cmd_bind_vertex_buffers(command_buffer, 0, &[buffers_data.interleaved_buffer], &[0]);
            device.cmd_bind_index_buffer(command_buffer, buffers_data.interleaved_buffer, buffers_data.interleaved_offset, vk::IndexType::UINT32);
        
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline_data.pipeline_layout,
                0,
                &[descriptor_data.global_descriptor_sets[image_index]],
                &[]
            );

			// skin binding
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline_data.pipeline_layout,
                2,
                &[descriptor_data.skinning_descriptor_set],
                &[]
            );

			// entity binding
			device.cmd_bind_descriptor_sets(
				command_buffer,
				vk::PipelineBindPoint::GRAPHICS,
				pipeline_data.pipeline_layout,
				3,
				&[descriptor_data.instance_descriptor_set],
				&[]
			);
        }

        let mut last_material: Option<MaterialSetId> = None;

        for item in &self.draw_list {
            // state tracking
            if last_material != Some(item.material_set_id) {
                last_material = Some(item.material_set_id);

                unsafe {
                    // materials binding
                    device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline_data.pipeline_layout,
                        1,
                        &[descriptor_data.material_descriptor_sets[item.material_set_id.0]],
                        &[]
                    );
                }
            }

            let material = {
                if let Some(material_id) = item.material_id {
                    Some(models.get_model(item.model_id).get_material(material_id))
                } else {
                    None
                }
            };

            let push_constant = PushConstants::new(item.node_matrix, material);

            unsafe {
                let push_bytes = std::slice::from_raw_parts(
                    &push_constant as *const PushConstants as *const u8,
                    size_of::<PushConstants>()
                );
                
                let frag_offset = PushConstants::get_frag_offset();

                device.cmd_push_constants(
                    command_buffer,
                    pipeline_data.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    &push_bytes[..frag_offset as usize]
                );
                device.cmd_push_constants(
                    command_buffer,
                    pipeline_data.pipeline_layout,
                    vk::ShaderStageFlags::FRAGMENT,
                    frag_offset,
                    &push_bytes[frag_offset as usize..],
                );

                device.cmd_draw_indexed(
                    command_buffer,
                    item.index_count,
                    item.instance_count,
                    item.first_index,
                    item.vertex_offset as i32,
                    item.instance_first
                );
            }
        } 

        unsafe { device.end_command_buffer(command_buffer)?; }
        Ok(command_buffer)
    }

    fn transition_for_render(
        device: &Device,
        swapchain_image: vk::Image,
        command_buffer: vk::CommandBuffer,
    ) {
        let barrier = vk::ImageMemoryBarrier2::builder()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .src_access_mask(vk::AccessFlags2::empty())
            .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
            .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE) // TODO: if blending then need to access READ aswell.
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .image(swapchain_image)
            .subresource_range(vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .base_array_layer(0)
                .layer_count(1)
                .level_count(1)
                .build()
            );

        let barriers = [barrier];
        let dependency_info = vk::DependencyInfo::builder()
            .image_memory_barriers(&barriers);

        unsafe { device.cmd_pipeline_barrier2_khr(command_buffer, &dependency_info) };
    }

    fn transition_for_present(
        device: &Device,
        swapchain_image: vk::Image,
        command_buffer: vk::CommandBuffer,
    ) {
        let barrier = vk::ImageMemoryBarrier2::builder()
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags2::empty())
            .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
            .image(swapchain_image)
            .subresource_range(vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .base_array_layer(0)
                .layer_count(1)
                .level_count(1)
                .build()
            );
        
        let barriers = [barrier];
        let dependency_info = vk::DependencyInfo::builder()
            .image_memory_barriers(&barriers);

        unsafe { device.cmd_pipeline_barrier2_khr(command_buffer, &dependency_info) };
    }
}

#[derive(Clone, Debug)]
pub struct SyncData {
    pub image_available_semaphores: Vec<vk::Semaphore>,
    pub render_finished_semaphores: Vec<vk::Semaphore>,
    pub in_flight_fences: Vec<vk::Fence>,
    pub images_in_flight: Vec<vk::Fence>,
}

impl SyncData {
    pub fn create(
        device: &Device,
        max_frame_in_flight: usize,
        images_count: usize
    ) -> Result<Self> {
        let (image_available_semaphores, render_finished_semaphores, in_flight_fences, images_in_flight) = create_sync_objects(
            device,
            max_frame_in_flight,
            images_count
        )?;

        Ok(Self {
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            images_in_flight
        })
    }

    #[rustfmt::skip]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn destroy(&mut self, device: &Device) {
        self.in_flight_fences.iter().for_each(|f| device.destroy_fence(*f, None)); // also free images_in_flight
        self.render_finished_semaphores.iter().for_each(|s| device.destroy_semaphore(*s, None));
        self.image_available_semaphores.iter().for_each(|s| device.destroy_semaphore(*s, None));
    }
}

#[derive(Clone, Debug)]
pub struct DepthData {
    pub depth_image: vk::Image,
    pub depth_image_memory: vk::DeviceMemory,
    pub depth_image_view: vk::ImageView,
    pub depth_format: vk::Format,
}

impl DepthData {
    pub fn create(
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        extent_width: u32,
        extent_height: u32,
        samples_count: vk::SampleCountFlags, 
    )-> Result<Self> {
        let (depth_image, depth_image_memory, depth_image_view, depth_format) = create_depth_objects(
            &instance,
            &device,
            physical_device,
            extent_width,
            extent_height,
            samples_count
        )?;

        Ok(Self {
            depth_image,
            depth_image_memory,
            depth_image_view,
            depth_format
        })
    }

    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn destroy(&self, device: &Device) {
        device.destroy_image_view(self.depth_image_view, None);
        device.free_memory(self.depth_image_memory, None);
        device.destroy_image(self.depth_image, None);
    }
}

#[derive(Clone, Debug)]
pub struct ColorData {
    pub color_image: vk::Image,
	pub color_image_memory: vk::DeviceMemory,
	pub color_image_view: vk::ImageView,
}

impl ColorData {
    pub fn create(
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        extent_width: u32,
        extent_height: u32,
        samples_count: vk::SampleCountFlags,
        swapchain_format: vk::Format,  
    ) -> Result<Self> {
        let (color_image, color_image_memory, color_image_view) = create_color_objects(
            instance,
            device,
            physical_device,
            extent_width,
            extent_height,
            samples_count,
            swapchain_format,
        )?;

        Ok(Self {
            color_image,
            color_image_memory,
            color_image_view
        })
    }

    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn destroy(&self, device: &Device) {
        device.destroy_image_view(self.color_image_view, None);
        device.free_memory(self.color_image_memory, None);
        device.destroy_image(self.color_image, None);
    }
}