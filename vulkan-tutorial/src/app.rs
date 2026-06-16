use anyhow::{Result, anyhow};
use std::collections::HashMap;
use std::time::Instant;
use std::ptr::copy_nonoverlapping as memcpy;

use log::*;

use winit::dpi::LogicalSize;
use winit::application::ApplicationHandler;
use winit::window::{Window, WindowId};
use winit::event_loop::ActiveEventLoop;
use winit::event::{DeviceEvent, WindowEvent, DeviceId, StartCause};
use winit_input_helper::WinitInputHelper;

use vulkanalia::loader::{LIBRARY, LibloadingLoader};
use vulkanalia::window as vk_window;
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::{ExtDebugUtilsExtensionInstanceCommands, KhrDynamicRenderingExtensionDeviceCommands, KhrSurfaceExtensionInstanceCommands, KhrSwapchainExtensionDeviceCommands, KhrSynchronization2ExtensionDeviceCommands};

use crate::constants::*;
use crate::gpu::{QueueFamilyIndices, create_instance, pick_best_physical_device,
                    get_max_msaa_samples, create_logical_device, create_global_descriptor_set_layout,
                    create_material_descriptor_set_layout, create_pipeline};
use crate::input::InputBindings;
use crate::render::{UniformBufferObject, TextureData, create_color_objects, create_depth_objects,
                    create_swapchain, create_swapchain_image_views};
use crate::resources::{create_command_pool, create_command_pools, create_setup_command_buffer,
                        create_interleaved_buffer, create_uniform_buffers, create_command_buffers,
                        destroy_buffers};
use crate::assets::{load_gltf_model};
use crate::scene::{Camera, CameraBuilder, Material, ModelGraph, Node};
use crate::setup::{create_descriptor_pool, create_global_descriptor_sets, create_material_descriptor_sets,
                    create_sync_objects, create_default_texture};
use crate::math::{Mat4, Vec3};

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
            if let Some(app) = self.app.as_mut() {
                unsafe { app.destroy() }; 
            }
            event_loop.exit();
            return;
        }

		if let Some(app) = self.app.as_mut() {
			let now = Instant::now();
			let delta_time = self.last_frame_time
				.map(|t| now.duration_since(t).as_secs_f32())
				.unwrap_or(0.0);
			self.last_frame_time = Some(now);

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
}

//===================================================
// App
//===================================================

/// Vulkan app.
#[derive(Clone)]
pub struct App {
    pub entry: Entry,
    pub instance: Instance,
    pub data: AppData,
    pub device: Device,
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

        // 3. swapchain
        let swapchain_data = SwapchainData::create(
            window,
            &instance,
            &device,
            physical_device,
            surface,
            &mut device_data.queue_family_indices
        )?;

        // 4. command
        let command_data = CommandData::create(
            &device,
            &mut device_data.queue_family_indices,
            &swapchain_data.swapchain_images
        )?;

        // 5. color
        let color_data = ColorData::create(
            &instance,
            &device,
            physical_device,
            swapchain_data.swapchain_extent.width,
            swapchain_data.swapchain_extent.height,
            msaa_samples,
            swapchain_data.swapchain_format
        )?;

        // 6. depth
        let depth_data = DepthData::create(
            &instance,
            &device,
            physical_device,
            swapchain_data.swapchain_extent.width,
            swapchain_data.swapchain_extent.height,
            msaa_samples,
        )?;

        // 7. pipeline
        let descriptor_layout_data = DescriptorLayoutData::create(&device)?;

        let pipeline_data = PipelineData::create(
            &device,
            swapchain_data.swapchain_format,
            swapchain_data.swapchain_extent,
            depth_data.depth_format,
            &descriptor_layout_data,
            msaa_samples,
            VERT,
            FRAG
        )?;
        
        // load .glb model
        let (helm_model, helm_textures) = load_gltf_model(
            &device,
            &instance,
            physical_device,
            FLIGHT_HELM_PATH,
            command_data.setup_command_buffer,
            graphics_queue
        )?;

        // 8. texture
        let textures_data = helm_textures;

        let default_texture = create_default_texture(
            &instance,
            &device,
            physical_device,
            command_data.setup_command_buffer,
            graphics_queue
        )?;

        // 9. model
        let models_data = helm_model;

        // 10. buffers
        let buffers_data = BuffersData::create(
            &instance,
            &device,
            physical_device,
            &models_data,
            command_data.setup_command_buffer,
            graphics_queue,
            swapchain_data.swapchain_images.len(),
        )?;

        // 11. descriptor
        let descriptor_data = DescriptorData::create(
            &device,
            &descriptor_layout_data,
            &buffers_data.uniform_buffers,
            &textures_data,
            &default_texture,
            &models_data.materials,
            swapchain_data.swapchain_images.len(),
        )?;
        
        // 12. sync
        let sync_data = SyncData::create(
            &device,
            MAX_FRAMES_IN_FLIGHT,
            swapchain_data.swapchain_images.len(),
        )?;

        // 13. Camera
        // TODO: support multiple cameras
        let mut camera = CameraBuilder::new()
            .movement_speed(30.0)
            .mouse_sensitivity(0.02)
            .build();

        camera.look_at(Vec3::new(0.0, 0.0, 0.0), None);

        let data = AppData {
            surface,
            device_data,
            swapchain_data,
            descriptor_layout_data,
            pipeline_data,
            models_data,
            buffers_data,
            descriptor_data,
            command_data,
            sync_data,
            textures_data,
            depth_data,
            color_data,
            camera_data: camera,
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

        // Destroy Appdata
        self.device.destroy_descriptor_pool(self.data.descriptor_data.descriptor_pool, None);
        self.destroy_swapchain();
        self.data.textures_data.iter_mut().for_each(|t| t.destroy(&self.device));
        self.data.default_texture.destroy(&self.device);
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
            &self.data.models_data,
            &self.data.pipeline_data,
            &self.data.buffers_data,
            &self.data.swapchain_data,
            &self.data.color_data,
            &self.data.depth_data,
            &self.data.descriptor_data,
            self.data.device_data.msaa_samples,
            image_index,
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
        debug!("camera info:\n- position: {:?}\n- front: {:?}\n- right {:?}\n- up: {:?}\n- zoom: {:?}°",
            self.data.camera_data.get_position(),
            self.data.camera_data.get_front(),
            self.data.camera_data.get_right(),
            self.data.camera_data.get_up(),
            self.data.camera_data.get_zoom()
        );

        let view = self.data.camera_data.get_view_matrix();

        let proj = CORRECTION * self.data.camera_data.get_projection_matrix(
            self.data.swapchain_data.swapchain_extent.width as f32 / self.data.swapchain_data.swapchain_extent.height as f32,
            Some(0.1),
            Some(1000.0)
        );

        let ubo = UniformBufferObject { view, proj };

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
    // Models
    pub models_data: ModelGraph,
    // Texture
	pub textures_data: Vec<TextureData>,
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
pub struct PipelineData {
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
}

impl PipelineData {
    pub fn create(
        device: &Device,
        swapchain_format: vk::Format,
        swapchain_extent: vk::Extent2D,
        depth_format: vk::Format,
        descriptor_layout_data: &DescriptorLayoutData,
        msaa_samples: vk::SampleCountFlags,
        vert: &[u8],
        frag: &[u8],
    ) -> Result<Self> {
        let (pipeline, pipeline_layout) = create_pipeline(
            &device,
            vert,
            frag,
            swapchain_extent,
            swapchain_format,
            depth_format,
            msaa_samples,
            descriptor_layout_data.global_set_layout,
            descriptor_layout_data.material_set_layout
        )?;

        Ok(Self {
            pipeline_layout,
            pipeline
        })
    }

    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn destroy(&self, device: &Device) {
        device.destroy_pipeline(self.pipeline, None);
        device.destroy_pipeline_layout(self.pipeline_layout, None);
    }
}

#[derive(Clone, Debug)]
pub struct BuffersData {
    pub interleaved_buffer: vk::Buffer,
    pub interleaved_buffer_memory: vk::DeviceMemory,
    pub interleaved_offset: u64,
    pub mesh_offsets: HashMap<usize, (u32, u32)>, // key: node idx -> value: (vert_offset, index_offset) 
    pub uniform_buffers: Vec<vk::Buffer>,
    pub uniform_buffers_memory: Vec<vk::DeviceMemory>,
}

impl BuffersData {
    pub fn create(
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        models_data: &ModelGraph,
        setup_command_buffer: vk::CommandBuffer,
        graphics_queue: vk::Queue,
        images_count: usize,
    ) -> Result<Self> {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut mesh_offsets = HashMap::new();

        for (i, w_node) in models_data.linear_nodes.iter().enumerate() {
            if let Some(ref_node) = w_node.upgrade() {
                match ref_node.try_borrow() {
                    Ok(node) => {
                        if let Some(mesh) = &node.mesh {
                            mesh_offsets.insert(i, (vertices.len() as u32, indices.len() as u32));

                            vertices.extend_from_slice(&mesh.vertices);
                            indices.extend_from_slice(&mesh.indices);
                        }
                    },
                    Err(e) => {
                        warn!("could not load a model node: {}", e);
                        continue;
                    },
                };
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
}

impl DescriptorLayoutData {
    pub fn create(
        device: &Device,
    ) -> Result<Self> {
        let global_set_layout = create_global_descriptor_set_layout(&device)?;
        let material_set_layout = create_material_descriptor_set_layout(&device)?;

        Ok(Self { global_set_layout, material_set_layout })
    }

    #[rustfmt::skip]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn destroy(&mut self, device: &Device) {
        device.destroy_descriptor_set_layout(self.global_set_layout, None);
        device.destroy_descriptor_set_layout(self.material_set_layout, None);
    }
}

#[derive(Clone, Debug)]
pub struct DescriptorData {
    pub descriptor_pool: vk::DescriptorPool,
    pub global_descriptor_sets: Vec<vk::DescriptorSet>,
    pub material_descriptor_sets: Vec<vk::DescriptorSet>,
}

impl DescriptorData {
    pub fn  create(
        device: &Device,
        descriptor_layout_data: &DescriptorLayoutData,
        uniform_buffers: &[vk::Buffer],
        textures_data: &[TextureData],
        default_texture: &TextureData,
        materials_data: &[Material],
        images_count: usize,
    ) -> Result<Self> {
        let materials_count = materials_data.len();
        info!("materials count: {}", materials_count);

        let descriptor_pool = create_descriptor_pool(
            device,
            images_count as u32,
            materials_count as u32,
        )?;
        
        let global_descriptor_sets = create_global_descriptor_sets(
            device,
            images_count,
            descriptor_layout_data.global_set_layout,
            descriptor_pool,
            uniform_buffers
        )?;

        let material_descriptor_sets = create_material_descriptor_sets(
            device,
            descriptor_layout_data.material_set_layout,
            descriptor_pool,
            materials_data,
            textures_data,
            default_texture
        )?;

        Ok(Self { descriptor_pool, global_descriptor_sets, material_descriptor_sets })
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
pub struct CommandData {
    pub command_pool: vk::CommandPool,
    pub command_pools: Vec<vk::CommandPool>,
    pub command_buffers: Vec<vk::CommandBuffer>,
    pub secondary_command_buffers: Vec<Vec<vk::CommandBuffer>>,
    pub setup_command_buffer: vk::CommandBuffer,
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
            setup_command_buffer
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
        models_data: &ModelGraph,
        pipeline_data: &PipelineData,
        buffers_data: &BuffersData,
        swapchain_data: &SwapchainData,
        color_data: &ColorData,
        depth_data: &DepthData,
        descriptor_data: &DescriptorData,
        msaa_samples: vk::SampleCountFlags,
        image_index: usize,
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

        let mut secondary_command_buffers = Vec::new();

        for (i, w_node) in models_data.linear_nodes.iter().enumerate() {
            let Some(ref_node) = w_node.upgrade() else { continue };
            if ref_node.borrow().mesh.is_none() { continue };
            
            let node = ref_node.borrow();
            secondary_command_buffers.push(self.update_secondary_command_buffers(
                device,
                &node,
                pipeline_data,
                buffers_data,
                &[swapchain_data.swapchain_format],
                depth_data,
                descriptor_data,
                msaa_samples,
                image_index,
                i,
            )?);
        }

        unsafe { 
            device.cmd_execute_commands(command_buffer, &secondary_command_buffers[..]);
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

    pub fn update_secondary_command_buffers(
        &mut self,
        device: &Device,
        model_node: &Node,
        pipeline_data: &PipelineData,
        buffers_data: &BuffersData,
        swapchain_formats: &[vk::Format],
        depth_data: &DepthData,
        descriptor_data: &DescriptorData,
        msaa_samples: vk::SampleCountFlags,
        image_index: usize,
        node_index: usize,
    ) -> Result<vk::CommandBuffer> {
        self.secondary_command_buffers.resize_with(image_index + 1, Vec::new);
        let command_buffers = &mut self.secondary_command_buffers[image_index];

        while node_index >= command_buffers.len() {
            let allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(self.command_pools[image_index])
                .level(vk::CommandBufferLevel::SECONDARY)
                .command_buffer_count(1);

            let command_buffer = unsafe { device.allocate_command_buffers(&allocate_info)?[0] };
            command_buffers.push(command_buffer);
        }

        let command_buffer = command_buffers[node_index];

        // push-constant model matrix
        // let model = model_data.instances[model_index].to_model_matrix();
        let mesh = model_node.mesh.as_ref().unwrap();
        let model = model_node.get_global_matrix();

        let model_bytes = unsafe {
            std::slice::from_raw_parts(
                &model as *const Mat4 as *const u8,
                size_of::<Mat4>()
            )
        };

        let opacity: f32 = 1.0;
        let opacity_bytes = &opacity.to_ne_bytes()[..];

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
            
            // UBO binding
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline_data.pipeline_layout,
                0,
                &[descriptor_data.global_descriptor_sets[image_index]],
                &[]
            );

            // materials binding
            let material_descriptor_index = model_node.mesh.as_ref()
                .map(|mesh| mesh.material_index.max(0) as usize)
                .unwrap_or(0);

            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline_data.pipeline_layout,
                1,
                &[descriptor_data.material_descriptor_sets[material_descriptor_index]],
                &[]
            );

            device.cmd_push_constants(
                command_buffer,
                pipeline_data.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                model_bytes
            );
            device.cmd_push_constants(
                command_buffer,
                pipeline_data.pipeline_layout,
                vk::ShaderStageFlags::FRAGMENT,
                64,
                opacity_bytes,
            );

            let (vertex_offset, first_index) = buffers_data.mesh_offsets.get(&node_index)
                .copied()
                .unwrap_or_default();

            device.cmd_draw_indexed(
                command_buffer,
                mesh.indices.len() as u32,
                1,
                first_index,
                vertex_offset as i32,
                0
            );
            device.end_command_buffer(command_buffer)?;
        }

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