#![allow(
    dead_code,
    unsafe_op_in_unsafe_fn,
    unused_variables,
    clippy::manual_slice_size_calculation,
    clippy::too_many_arguments,
    clippy::unnecessary_wraps
)]

use std::collections::HashSet;
use std::mem::size_of;
use std::ptr::copy_nonoverlapping as memcpy;
use std::time::Instant;

use anyhow::{Result, anyhow};
use cgmath::{
    vec3,
    point3,
    Deg,
};
use log::*;

use vulkanalia::loader::{LIBRARY, LibloadingLoader};
use vulkanalia::prelude::v1_0::*;
use vulkanalia::window as vk_window;
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent, ElementState};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::event_loop::EventLoop;
use winit::window::{Window, WindowBuilder};

use vulkanalia::vk::ExtDebugUtilsExtensionInstanceCommands;
use vulkanalia::vk::KhrSurfaceExtensionInstanceCommands;
use vulkanalia::vk::KhrSwapchainExtensionDeviceCommands;

use graphic_env::resources::*;
use graphic_env::render::*;
use graphic_env::assets::*;
use graphic_env::math::*;
use graphic_env::geometry::Vertex;
use graphic_env::gpu::*;
use graphic_env::setup::*;
use graphic_env::debug::debug_callback;

/// Whether the validation layers should be enabled.
const VALIDATION_ENABLED: bool = cfg!(debug_assertions);
/// The name of the validation layers.
const VALIDATION_LAYER: vk::ExtensionName = vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");

/// The required device extensions.
const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[vk::KHR_SWAPCHAIN_EXTENSION.name];

/// The maximum number of frames that can be processed concurrently.
const MAX_FRAMES_IN_FLIGHT: usize = 2;

/// The texture test path
const TEXTURE_PATH: &str = "resources/textures/viking_room.png";

/// The obj test path
const MESH_PATH: &str = "resources/3D_meshes/viking_room.obj";

/// Vert shader path
const VERT: &[u8] = include_bytes!("../shaders/hello_model/vert.spv");

/// Frag shader path
const FRAG: &[u8] = include_bytes!("../shaders/hello_model/frag.spv");

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

    let mut app = unsafe { App::create(&window)? };
    let mut minimized = false;
    event_loop.run(move |event, elwt| {
        match event {
            // Request a redraw when all events were processed.
            Event::AboutToWait => window.request_redraw(),
            Event::WindowEvent { event, .. } => match event {
                // Render a frame if our Vulkan app is not being destroyed.
                WindowEvent::RedrawRequested if !elwt.exiting() && !minimized => {
                    unsafe { app.render(&window) }.unwrap();
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

/// Our Vulkan app.
#[derive(Clone, Debug)]
struct App {
    entry: Entry,
    instance: Instance,
    data: AppData,
    device: Device,
    frame: usize,
    resized: bool,
    start: Instant,
    models: usize,
}

impl App {
    /// Creates our Vulkan app.
    unsafe fn create(window: &Window) -> Result<Self> {
        // TODO: move all params into a param file.
        let mandatory_feats = vk::PhysicalDeviceFeatures::builder()
            .sampler_anisotropy(true)
            .build();

        let optional_feats = vk::PhysicalDeviceFeatures::builder()
            .build();

        let mandatory_queue_flags = vk::QueueFlags::GRAPHICS;

        let loader = LibloadingLoader::new(LIBRARY)?;
        let entry = Entry::new(loader).map_err(|b| anyhow!("{}", b))?;
        let mut data = AppData::default();
        let instance = create_instance(window, &entry, &mut data)?;
        data.surface = vk_window::create_surface(&instance, &window, &window)?;
        data.physical_device = pick_best_physical_device(
            &instance,
            data.surface,
            &mandatory_feats,
            &optional_feats,
            &DEVICE_EXTENSIONS,
            &[] as &[vk::ExtensionName],
            mandatory_queue_flags,
            false
        )?;
        data.msaa_samples = get_max_msaa_samples(&instance, data.physical_device);
        data.queue_family_indices = QueueFamilyIndices::create(
            &instance,
            data.physical_device,
            data.surface
        )?;

        let (device, graphics_queue, present_queue) = create_logical_device(
            &entry,
            &instance,
            data.physical_device,
            &mut data.queue_family_indices,
            VALIDATION_ENABLED,
            VALIDATION_LAYER,
            DEVICE_EXTENSIONS
        )?;
        data.graphics_queue = graphics_queue;
        data.present_queue = present_queue;

        (data.swapchain, data.swapchain_format, data.swapchain_extent, data.swapchain_images) = create_swapchain(
            window,
            &instance,
            &device,
            data.physical_device,
            data.surface,
            &mut data.queue_family_indices
        )?;
        data.swapchain_image_views = create_swapchain_image_views(
            &device,
            &data.swapchain_images,
            data.swapchain_format
        )?;
        data.render_pass = create_render_pass(
            &instance,
            &device,
            data.physical_device,
            data.swapchain_format,
            data.msaa_samples
        )?;
        data.descriptor_set_layout = create_descriptor_set_layout(&device)?;
        (data.pipeline, data.pipeline_layout) = create_pipeline(
            &device,
            VERT,
            FRAG,
            data.render_pass,
            data.swapchain_extent,
            data.msaa_samples,
            data.descriptor_set_layout
        )?;
        data.command_pool = create_command_pool(&device, &mut data.queue_family_indices)?;
        data.command_pools = create_command_pools(
            &device,
            &mut data.queue_family_indices,
            data.swapchain_images.len(),
        )?;
		(data.color_image, data.color_image_memory, data.color_image_view) = create_color_objects(
            &instance,
            &device,
            data.physical_device,
            data.swapchain_extent.width,
            data.swapchain_extent.height,
            data.msaa_samples,
            data.swapchain_format,
        )?;
        (data.depth_image, data.depth_image_memory, data.depth_image_view) = create_depth_objects(
            &instance,
            &device,
            data.physical_device,
            data.swapchain_extent.width,
            data.swapchain_extent.height,
            data.msaa_samples
        )?;
        data.framebuffers = create_framebuffers(
            &device,
            data.render_pass,
            &data.swapchain_image_views,
            data.color_image_view,
            data.depth_image_view,
            data.swapchain_extent.width,
            data.swapchain_extent.height,
        )?;
        data.setup_command_buffer = create_setup_command_buffer(&device, data.command_pool)?;
        (data.texture_image, data.texture_image_memory, data.mip_levels) = create_texture_image(
            &instance,
            &device,
            data.physical_device,
            TEXTURE_PATH,
            data.setup_command_buffer,
            data.graphics_queue
        )?;
		data.texture_image_view = create_texture_image_view(&device, data.texture_image, data.mip_levels)?;
        data.texture_sampler = create_texture_sampler(&device, data.mip_levels as f32)?;

        (data.vertices, data.indices) = load_obj_model(MESH_PATH)?;
        (data.interleaved_buffer, data.interleaved_buffer_memory, data.index_offset) = create_interleaved_buffer(
            &instance,
            &device,
            data.physical_device,
            &data.vertices,
            &data.indices,
            data.setup_command_buffer,
            data.graphics_queue,
        )?;
        (data.uniform_buffers, data.uniform_buffers_memory) = create_uniform_buffers(
            &instance,
            &device,
            data.physical_device,
            data.swapchain_images.len(),
        )?;
        data.descriptor_pool = create_descriptor_pool(&device, data.swapchain_images.len() as u32)?;
        data.descriptor_sets = create_descriptor_sets(
            &device,
            data.swapchain_images.len(),
            data.descriptor_set_layout,
            data.descriptor_pool,
            &data.uniform_buffers,
            data.texture_image_view,
            data.texture_sampler
        )?;
        (data.command_buffers, data.secondary_command_buffers) = create_command_buffers(
            &device,
            &data.command_pools
        )?;
        (data.image_available_semaphores, data.render_finished_semaphores, data.in_flight_fences, data.images_in_flight) = create_sync_objects(
            &device,
            MAX_FRAMES_IN_FLIGHT,
            data.swapchain_images.len()
        )?;
        Ok(Self {
            entry,
            instance,
            data,
            device,
            frame: 0,
            resized: false,
            start: Instant::now(),
            models: 1
        })
    }

    /// Renders a frame for our Vulkan app.
    unsafe fn render(&mut self, window: &Window) -> Result<()> {
        let in_flight_fence = self.data.in_flight_fences[self.frame];

        self.device.wait_for_fences(&[in_flight_fence], true, u64::MAX)?;

        let result = self.device.acquire_next_image_khr(
            self.data.swapchain,
            u64::MAX,
            self.data.image_available_semaphores[self.frame],
            vk::Fence::null(),
        );

        let image_index = match result {
            Ok((image_index, _)) => image_index as usize,
            Err(vk::ErrorCode::OUT_OF_DATE_KHR) => return self.recreate_swapchain(window),
            Err(e) => return Err(anyhow!(e)),
        };

        let image_in_flight = self.data.images_in_flight[image_index];
        if !image_in_flight.is_null() {
            self.device.wait_for_fences(&[image_in_flight], true, u64::MAX)?;
        }

        self.data.images_in_flight[image_index] = in_flight_fence;

        self.update_command_buffer(image_index)?;
        self.update_uniform_buffer(image_index)?;

        let wait_semaphores = &[self.data.image_available_semaphores[self.frame]];
        let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = &[self.data.command_buffers[image_index]];
        let signal_semaphores = &[self.data.render_finished_semaphores[self.frame]];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores);

        self.device.reset_fences(&[in_flight_fence])?;

        self.device
            .queue_submit(self.data.graphics_queue, &[submit_info], in_flight_fence)?;

        let swapchains = &[self.data.swapchain];
        let image_indices = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);

        let result = self.device.queue_present_khr(self.data.present_queue, &present_info);
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

    unsafe fn update_uniform_buffer(&self, image_index: usize) -> Result<()> {
        let view = Mat4::look_at_rh(
            point3(6.0, 0.0, 2.0),
            point3(0.0, 0.0, 0.0),
            vec3(0.0, 0.0, 1.0)
        );

        // this matrix is use to correct the proj matrix from the OpenGL range to the Vulkan one.
        // warn: this is a column-major matrix.
        let correction = Mat4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, -1.0, 0.0, 0.0,
            0.0, 0.0, 1.0 / 2.0, 0.0,
            0.0, 0.0, 1.0 / 2.0, 1.0
        );

        let proj = correction * cgmath::perspective(
            Deg(45.0),
            self.data.swapchain_extent.width as f32 / self.data.swapchain_extent.height as f32,
            0.1,
            10.0
        );

        let ubo = UniformBufferObject { view, proj };

        let memory = self.device.map_memory(
            self.data.uniform_buffers_memory[image_index],
            0,
            size_of::<UniformBufferObject>() as u64,
            vk::MemoryMapFlags::empty()
        )?;

        memcpy(&ubo, memory.cast(), 1);
        self.device.unmap_memory(self.data.uniform_buffers_memory[image_index]);

        Ok(())
    }

    unsafe fn update_command_buffer(&mut self, image_index: usize) -> Result<()> {
        // Update the command buffer
        let command_pool = self.data.command_pools[image_index];
        self.device.reset_command_pool(command_pool, vk::CommandPoolResetFlags::empty())?;

        let command_buffer = self.data.command_buffers[image_index];

        // Commands
        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        self.device.begin_command_buffer(command_buffer, &info)?;

        let render_area = vk::Rect2D::builder()
            .offset(vk::Offset2D::default())
            .extent(self.data.swapchain_extent);

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

        let clear_values = &[color_clear_value, depth_clear_value];
        let info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.data.render_pass)
            .framebuffer(self.data.framebuffers[image_index])
            .render_area(render_area)
            .clear_values(clear_values);

        self.device.cmd_begin_render_pass(command_buffer, &info, vk::SubpassContents::SECONDARY_COMMAND_BUFFERS);

        let secondary_command_buffers = (0..self.models)
            .map(|i| self.update_secondary_command_buffers(image_index, i))
            .collect::<Result<Vec<_>, _>>()?;
            
        self.device.cmd_execute_commands(command_buffer, &secondary_command_buffers[..]);

        self.device.cmd_end_render_pass(command_buffer);

        self.device.end_command_buffer(command_buffer)?;

        Ok(())
    }

    unsafe fn update_secondary_command_buffers(&mut self, image_index: usize, model_index: usize) -> Result<vk::CommandBuffer> {
        self.data.secondary_command_buffers.resize_with(image_index + 1, Vec::new);

        let command_buffers = &mut self.data.secondary_command_buffers[image_index];
        while model_index >= command_buffers.len() {
            let allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(self.data.command_pools[image_index])
                .level(vk::CommandBufferLevel::SECONDARY)
                .command_buffer_count(1);

            let command_buffer = self.device.allocate_command_buffers(&allocate_info)?[0];
            command_buffers.push(command_buffer);
        }

        let command_buffer = command_buffers[model_index];

        // push-constant model matrix
        let y = (((model_index % 2) as f32) * 2.5) - 1.25;
        let z = (((model_index / 2) as f32) * -2.0) + 1.0;

        let time = self.start.elapsed().as_secs_f32(); // to make the model rotate

        let model = Mat4::from_translation(vec3(0.0, y, z)) 
            * Mat4::from_axis_angle(vec3(0.0, 0.0 ,1.0), Deg(90.0) * time);

        let model_bytes = std::slice::from_raw_parts(
            &model as *const Mat4 as *const u8,
            size_of::<Mat4>()
        );

        let opacity = (model_index + 1) as f32 * 0.25;
        let opacity_bytes = &opacity.to_ne_bytes()[..];

        let inheritance_info = vk::CommandBufferInheritanceInfo::builder()
            .render_pass(self.data.render_pass)
            .subpass(0)
            .framebuffer(self.data.framebuffers[image_index]);

        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE)
            .inheritance_info(&inheritance_info);

        self.device.begin_command_buffer(command_buffer, &info)?;

        self.device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.data.pipeline);
        self.device.cmd_bind_vertex_buffers(command_buffer, 0, &[self.data.interleaved_buffer], &[0]);
        self.device.cmd_bind_index_buffer(command_buffer, self.data.interleaved_buffer, self.data.index_offset, vk::IndexType::UINT32);

        self.device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.data.pipeline_layout,
            0,
            &[self.data.descriptor_sets[image_index]],
            &[]
        );

        self.device.cmd_push_constants(
            command_buffer,
            self.data.pipeline_layout,
            vk::ShaderStageFlags::VERTEX,
            0,
            model_bytes
        );
        self.device.cmd_push_constants(
            command_buffer,
            self.data.pipeline_layout,
            vk::ShaderStageFlags::FRAGMENT,
            64,
            opacity_bytes,
        );

        self.device.cmd_draw_indexed(command_buffer, self.data.indices.len() as u32, 1, 0, 0, 0);

        self.device.end_command_buffer(command_buffer)?;

        Ok(command_buffer)
    }

    /// Recreates the swapchain for our Vulkan app.
    #[rustfmt::skip]
    unsafe fn recreate_swapchain(&mut self, window: &Window) -> Result<()> {
        self.device.device_wait_idle()?;
        self.destroy_swapchain();
       (self.data.swapchain, self.data.swapchain_format, self.data.swapchain_extent, self.data.swapchain_images) = create_swapchain(
            window,
            &self.instance,
            &self.device,
            self.data.physical_device,
            self.data.surface,
            &mut self.data.queue_family_indices
        )?;
        self.data.swapchain_image_views = create_swapchain_image_views(
            &self.device,
            &self.data.swapchain_images,
            self.data.swapchain_format
        )?;
        self.data.render_pass = create_render_pass(
            &self.instance,
            &self.device,
            self.data.physical_device,
            self.data.swapchain_format,
            self.data.msaa_samples
        )?;
        (self.data.pipeline, self.data.pipeline_layout) = create_pipeline(
            &self.device,
            VERT,
            FRAG,
            self.data.render_pass,
            self.data.swapchain_extent,
            self.data.msaa_samples,
            self.data.descriptor_set_layout
        )?;
        (self.data.color_image, self.data.color_image_memory, self.data.color_image_view) = create_color_objects(
            &self.instance,
            &self.device,
            self.data.physical_device,
            self.data.swapchain_extent.width,
            self.data.swapchain_extent.height,
            self.data.msaa_samples,
            self.data.swapchain_format,
        )?;
        (self.data.depth_image, self.data.depth_image_memory, self.data.depth_image_view) = create_depth_objects(
            &self.instance,
            &self.device,
            self.data.physical_device,
            self.data.swapchain_extent.width,
            self.data.swapchain_extent.height,
            self.data.msaa_samples,
        )?;
        self.data.framebuffers = create_framebuffers(
            &self.device,
            self.data.render_pass,
            &self.data.swapchain_image_views,
            self.data.color_image_view,
            self.data.depth_image_view,
            self.data.swapchain_extent.width,
            self.data.swapchain_extent.height,
        )?;
        (self.data.uniform_buffers, self.data.uniform_buffers_memory) = create_uniform_buffers(
            &self.instance,
            &self.device,
            self.data.physical_device,
            self.data.swapchain_images.len()
        )?;
        self.data.descriptor_pool = create_descriptor_pool(&self.device, self.data.swapchain_images.len() as u32)?;
        self.data.descriptor_sets = create_descriptor_sets(
            &self.device,
            self.data.swapchain_images.len(),
            self.data.descriptor_set_layout,
            self.data.descriptor_pool,
            &self.data.uniform_buffers,
            self.data.texture_image_view,
            self.data.texture_sampler
        )?;
        (self.data.command_buffers, self.data.secondary_command_buffers) = create_command_buffers(
            &self.device,
            &self.data.command_pools
        )?;
        self.data.images_in_flight.resize(self.data.swapchain_images.len(), vk::Fence::null());
        Ok(())
    }

    /// Destroys our Vulkan app.
    #[rustfmt::skip]
    unsafe fn destroy(&mut self) {
        self.device.device_wait_idle().unwrap();

        self.destroy_swapchain();

        self.device.destroy_sampler(self.data.texture_sampler, None);
        self.device.destroy_image_view(self.data.texture_image_view, None);
		self.device.destroy_image(self.data.texture_image, None);
		self.device.free_memory(self.data.texture_image_memory, None);
        self.device.destroy_descriptor_set_layout(self.data.descriptor_set_layout, None);
        self.data.in_flight_fences.iter().for_each(|f| self.device.destroy_fence(*f, None));
        self.data.render_finished_semaphores.iter().for_each(|s| self.device.destroy_semaphore(*s, None));
        self.data.image_available_semaphores.iter().for_each(|s| self.device.destroy_semaphore(*s, None));
        destroy_buffers(&self.device, &[self.data.interleaved_buffer], &[self.data.interleaved_buffer_memory]);
        self.data.command_pools.iter().for_each(|c| self.device.destroy_command_pool(*c, None));
        self.device.free_command_buffers(self.data.command_pool, &[self.data.setup_command_buffer]);
        self.device.destroy_command_pool(self.data.command_pool, None);
        self.device.destroy_device(None);
        self.instance.destroy_surface_khr(self.data.surface, None);

        if VALIDATION_ENABLED {
            self.instance.destroy_debug_utils_messenger_ext(self.data.messenger, None);
        }

        self.instance.destroy_instance(None);
    }

    /// Destroys the parts of our Vulkan app related to the swapchain.
    #[rustfmt::skip]
    unsafe fn destroy_swapchain(&mut self) {
        self.device.destroy_descriptor_pool(self.data.descriptor_pool, None);
        destroy_buffers(&self.device, &self.data.uniform_buffers, &self.data.uniform_buffers_memory);
        self.device.destroy_image_view(self.data.depth_image_view, None);
        self.device.free_memory(self.data.depth_image_memory, None);
        self.device.destroy_image(self.data.depth_image, None);
		self.device.destroy_image_view(self.data.color_image_view, None);
		self.device.free_memory(self.data.color_image_memory, None);
		self.device.destroy_image(self.data.color_image, None);
        self.data.framebuffers.iter().for_each(|f| self.device.destroy_framebuffer(*f, None));
        self.device.destroy_pipeline(self.data.pipeline, None);
        self.device.destroy_pipeline_layout(self.data.pipeline_layout, None);
        self.device.destroy_render_pass(self.data.render_pass, None);
        self.data.swapchain_image_views.iter().for_each(|v| self.device.destroy_image_view(*v, None));
        self.device.destroy_swapchain_khr(self.data.swapchain, None);
    }
}

/// The Vulkan handles and associated properties used by our Vulkan app.
#[derive(Clone, Debug, Default)]
struct AppData {
    // Debug
    messenger: vk::DebugUtilsMessengerEXT,
    // Surface
    surface: vk::SurfaceKHR,
    // Physical Device / Logical Device
    physical_device: vk::PhysicalDevice,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
	msaa_samples: vk::SampleCountFlags,
    // QueueFamily
    queue_family_indices: QueueFamilyIndices,
    // Swapchain
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    // Pipeline
    render_pass: vk::RenderPass,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    // Framebuffers
    framebuffers: Vec<vk::Framebuffer>,
    // Command Pool
    command_pool: vk::CommandPool,
    // Meshes and Models
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    // Buffers
    interleaved_buffer: vk::Buffer,
    interleaved_buffer_memory: vk::DeviceMemory,
    index_offset: u64,
    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffers_memory: Vec<vk::DeviceMemory>,
    // Descriptor
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    // Command Buffers
    command_pools: Vec<vk::CommandPool>,
    command_buffers: Vec<vk::CommandBuffer>,
    secondary_command_buffers: Vec<Vec<vk::CommandBuffer>>,
    setup_command_buffer: vk::CommandBuffer,
    // Sync Objects
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    images_in_flight: Vec<vk::Fence>,
	// Texture
	texture_image: vk::Image,
	texture_image_memory: vk::DeviceMemory,
	texture_image_view: vk::ImageView,
    texture_sampler: vk::Sampler,
    mip_levels: u32,
    // Depth
    depth_image: vk::Image,
    depth_image_memory: vk::DeviceMemory,
    depth_image_view: vk::ImageView,
	// Render target (now only use for MSAA)
	color_image: vk::Image,
	color_image_memory: vk::DeviceMemory,
	color_image_view: vk::ImageView,
}

//================================================
// Instance
//================================================

unsafe fn create_instance(window: &Window, entry: &Entry, data: &mut AppData) -> Result<Instance> {
    // Application Info

    let application_info = vk::ApplicationInfo::builder()
        .application_name(b"Vulkan Tutorial (Rust)\0")
        .application_version(vk::make_version(1, 0, 0))
        .engine_name(b"No Engine\0")
        .engine_version(vk::make_version(1, 0, 0))
        .api_version(vk::make_version(1, 0, 0));

    // Layers

    let available_layers = entry
        .enumerate_instance_layer_properties()?
        .iter()
        .map(|l| l.layer_name)
        .collect::<HashSet<_>>();

    if VALIDATION_ENABLED && !available_layers.contains(&VALIDATION_LAYER) {
        return Err(anyhow!("Validation layer requested but not supported."));
    }

    let layers = if VALIDATION_ENABLED {
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        Vec::new()
    };

    // Extensions

    let mut extensions = vk_window::get_required_instance_extensions(window)
        .iter()
        .map(|e| e.as_ptr())
        .collect::<Vec<_>>();

    // Required by Vulkan SDK on macOS since 1.3.216.
    let flags = if cfg!(target_os = "macos") && entry.version()? >= PORTABILITY_MACOS_VERSION {
        info!("Enabling extensions for macOS portability.");
        extensions.push(vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_EXTENSION.name.as_ptr());
        extensions.push(vk::KHR_PORTABILITY_ENUMERATION_EXTENSION.name.as_ptr());
        vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
    } else {
        vk::InstanceCreateFlags::empty()
    };

    if VALIDATION_ENABLED {
        extensions.push(vk::EXT_DEBUG_UTILS_EXTENSION.name.as_ptr());
    }

    // Create

    let mut info = vk::InstanceCreateInfo::builder()
        .application_info(&application_info)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions)
        .flags(flags);

    let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .user_callback(Some(debug_callback));

    if VALIDATION_ENABLED {
        info = info.push_next(&mut debug_info);
    }

    let instance = entry.create_instance(&info, None)?;

    // Messenger

    if VALIDATION_ENABLED {
        data.messenger = instance.create_debug_utils_messenger_ext(&debug_info, None)?;
    }

    Ok(instance)
}