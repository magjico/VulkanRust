use anyhow::{Result, anyhow};
use cgmath::{Deg, vec3, point3};

use std::time::Instant;
use std::ptr::copy_nonoverlapping as memcpy;

use winit::window::Window;
use vulkanalia::loader::{LIBRARY, LibloadingLoader};
use vulkanalia::window as vk_window;
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::{KhrSurfaceExtensionInstanceCommands, KhrSwapchainExtensionDeviceCommands,
                    ExtDebugUtilsExtensionInstanceCommands};

use crate::constants::{CORRECTION, DEVICE_EXTENSIONS, FRAG, MAX_FRAMES_IN_FLIGHT, TEXTURE_PATH, VALIDATION_ENABLED, VALIDATION_LAYER, VERT};
use crate::gpu::{QueueFamilyIndices, create_instance, pick_best_physical_device,
                    get_max_msaa_samples, create_logical_device, create_render_pass,
                    create_descriptor_set_layout, create_pipeline};
use crate::render::{UniformBufferObject, create_swapchain, create_swapchain_image_views,
                    create_color_objects, create_depth_objects, create_texture_image,
                    create_texture_image_view, create_texture_sampler};
use crate::resources::{create_command_pool, create_command_pools, create_framebuffers,
                        create_setup_command_buffer, create_interleaved_buffer, create_uniform_buffers,
                        create_command_buffers, destroy_buffers};
use crate::scene::Model;
use crate::setup::{create_descriptor_pool, create_descriptor_sets, create_sync_objects, load_models};
use crate::math::Mat4;

//===================================================
// App
//===================================================

/// Vulkan app.
#[derive(Clone, Debug)]
pub struct App {
    pub entry: Entry,
    pub instance: Instance,
    pub data: AppData,
    pub device: Device,
    pub frame: usize,
    pub resized: bool,
    pub start: Instant,
    pub models: usize,
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

        // 4. pipeline
        let descriptor_set_layout = create_descriptor_set_layout(&device)?;

        let pipeline_data = PipelineData::create(
            &instance,
            &device,
            physical_device,
            swapchain_data.swapchain_format,
            swapchain_data.swapchain_extent,
            descriptor_set_layout,
            msaa_samples,
            VERT,
            FRAG
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

        // 8. framebuffers
        let framebuffers = create_framebuffers(
            &device,
            pipeline_data.render_pass,
            &swapchain_data.swapchain_image_views,
            color_data.color_image_view,
            depth_data.depth_image_view,
            swapchain_data.swapchain_extent.width,
            swapchain_data.swapchain_extent.height,
        )?;
        
        // 9. texture
        let texture_data = TextureData::create(
            &instance,
            &device,
            physical_device,
            command_data.setup_command_buffer,
            graphics_queue,
            TEXTURE_PATH
        )?;

        // 10. model
        let models = load_models()?;
        let models_data = ModelsData::create(models); 

        // 11. buffers
        let buffers_data = BuffersData::create(
            &instance,
            &device,
            physical_device,
            &models_data,
            command_data.setup_command_buffer,
            graphics_queue,
            swapchain_data.swapchain_images.len(),
        )?;

        // 12. descriptor
        let descriptor_data = DescriptorData::create(
            &device,
            descriptor_set_layout,
            &buffers_data.uniform_buffers,
            texture_data.texture_image_view,
            texture_data.texture_sampler,
            swapchain_data.swapchain_images.len()
        )?;
        
        // 13. sync
        let sync_data = SyncData::create(
            &device,
            MAX_FRAMES_IN_FLIGHT,
            swapchain_data.swapchain_images.len(),
        )?;

        let data = AppData {
            surface,
            device_data,
            swapchain_data,
            descriptor_set_layout,
            pipeline_data,
            framebuffers,
            models_data,
            buffers_data,
            descriptor_data,
            command_data,
            sync_data,
            texture_data,
            depth_data,
            color_data,
            messenger
        };

        Ok( Self {
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

    /// Destroys our Vulkan app.
    #[rustfmt::skip]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn destroy(&mut self) {
        self.device.device_wait_idle().unwrap();

        // Destroy Appdata
        self.destroy_swapchain();
        self.data.texture_data.destroy(&self.device);
        self.device.destroy_descriptor_set_layout(self.data.descriptor_set_layout, None);
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
        self.device.destroy_descriptor_pool(self.data.descriptor_data.descriptor_pool, None);

        destroy_buffers(
            &self.device,
            &self.data.buffers_data.uniform_buffers,
            &self.data.buffers_data.uniform_buffers_memory
        );

        self.data.depth_data.destroy(&self.device);
		self.data.color_data.destroy(&self.device);
        self.data.framebuffers.iter().for_each(|f| self.device.destroy_framebuffer(*f, None));
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

        self.data.pipeline_data = PipelineData::create(
            &self.instance,
            &self.device,
            self.data.device_data.physical_device,
            self.data.swapchain_data.swapchain_format,
            self.data.swapchain_data.swapchain_extent,
            self.data.descriptor_set_layout,
            self.data.device_data.msaa_samples,
            VERT,
            FRAG,
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

        self.data.framebuffers = create_framebuffers(
            &self.device,
            self.data.pipeline_data.render_pass,
            &self.data.swapchain_data.swapchain_image_views,
            self.data.color_data.color_image_view,
            self.data.depth_data.depth_image_view,
            self.data.swapchain_data.swapchain_extent.width,
            self.data.swapchain_data.swapchain_extent.height,
        )?;

        (self.data.buffers_data.uniform_buffers, self.data.buffers_data.uniform_buffers_memory) = create_uniform_buffers(
            &self.instance,
            &self.device,
            self.data.device_data.physical_device,
            self.data.swapchain_data.swapchain_images.len()
        )?;

        self.data.descriptor_data = DescriptorData::create(
            &self.device,
            self.data.descriptor_set_layout,
            &self.data.buffers_data.uniform_buffers,
            self.data.texture_data.texture_image_view,
            self.data.texture_data.texture_sampler,
            self.data.swapchain_data.swapchain_images.len()
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
            &self.data.framebuffers,
            &self.data.models_data,
            &self.data.pipeline_data,
            &self.data.buffers_data,
            &self.data.descriptor_data.descriptor_sets,
            self.data.swapchain_data.swapchain_extent,
            self.data.pipeline_data.render_pass,
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
        let view = Mat4::look_at_rh(
            point3(0.0, 0.0, 5.0),
            point3(0.0, 0.0, 0.0),
            vec3(0.0, 1.0, 0.0)
        );

        let proj = CORRECTION * cgmath::perspective(
            Deg(90.0),
            self.data.swapchain_data.swapchain_extent.width as f32 / self.data.swapchain_data.swapchain_extent.height as f32,
            0.1,
            20.0
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
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub pipeline_data: PipelineData,
    // Framebuffers
    pub framebuffers: Vec<vk::Framebuffer>,
    // Models
    pub models_data: ModelsData,
    // Buffers
    pub buffers_data: BuffersData,
    // Descriptor
    pub descriptor_data: DescriptorData,
    // Command Buffers
    pub command_data: CommandData,
    // Sync Objects
    pub sync_data: SyncData,
	// Texture
	pub texture_data: TextureData,
    // Depth
    pub depth_data: DepthData,
	// Render target (now only use for MSAA)
	pub color_data: ColorData,
    // Debug
    pub messenger: Option<vk::DebugUtilsMessengerEXT>,
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
    pub render_pass: vk::RenderPass,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
}

impl PipelineData {
    pub fn create(
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        swapchain_format: vk::Format,
        swapchain_extent: vk::Extent2D,
        descriptor_set_layout: vk::DescriptorSetLayout,
        msaa_samples: vk::SampleCountFlags,
        vert: &[u8],
        frag: &[u8],
    ) -> Result<Self> {
        let render_pass = create_render_pass(
            instance,
            device,
            physical_device,
            swapchain_format,
            msaa_samples
        )?;

        
        let (pipeline, pipeline_layout) = create_pipeline(
            &device,
            vert,
            frag,
            render_pass,
            swapchain_extent,
            msaa_samples,
            descriptor_set_layout
        )?;

        Ok(Self {
            render_pass,
            pipeline_layout,
            pipeline
        })
    }

    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn destroy(&self, device: &Device) {
        device.destroy_pipeline(self.pipeline, None);
        device.destroy_pipeline_layout(self.pipeline_layout, None);
        device.destroy_render_pass(self.render_pass, None);
    }
}

#[derive(Clone, Debug)]
pub struct ModelsData {
    // TODO: change this later for a models organisation architectures
    pub models: Vec<Model>
}

impl ModelsData {
    pub fn create(
        models: Vec<Model>,
    ) -> Self {
        Self {
            models,
        }
    }
}

#[derive(Clone, Debug)]
pub struct BuffersData {
    pub interleaved_buffer: vk::Buffer,
    pub interleaved_buffer_memory: vk::DeviceMemory,
    pub index_offset: u64,
    pub uniform_buffers: Vec<vk::Buffer>,
    pub uniform_buffers_memory: Vec<vk::DeviceMemory>,
}

impl BuffersData {
    pub fn create(
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        models_data: &ModelsData,
        setup_command_buffer: vk::CommandBuffer,
        graphics_queue: vk::Queue,
        images_count: usize,
    ) -> Result<Self> {
        // TODO: change this later to support multiple models
        let model = &models_data.models[0];

        let (interleaved_buffer, interleaved_buffer_memory, index_offset) = create_interleaved_buffer(
            instance,
            device,
            physical_device,
            &model.mesh.vertices,
            &model.mesh.indices,
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
            index_offset,
            uniform_buffers,
            uniform_buffers_memory,
        })
    }
}

#[derive(Clone, Debug)]
pub struct DescriptorData {
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
}

impl DescriptorData {
    pub fn create(
        device: &Device,
        descriptor_set_layout: vk::DescriptorSetLayout,
        uniform_buffers: &[vk::Buffer],
        texture_image_view: vk::ImageView,
        texture_sampler: vk::Sampler,
        images_count: usize,
    ) -> Result<Self> {
        let descriptor_pool = create_descriptor_pool(device, images_count as u32)?;
        let descriptor_sets = create_descriptor_sets(
            device,
            images_count,
            descriptor_set_layout,
            descriptor_pool,
            uniform_buffers,
            texture_image_view,
            texture_sampler
        )?;

        Ok(Self { descriptor_pool, descriptor_sets })
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
        framebuffers: &[vk::Framebuffer],
        models_data: &ModelsData,
        pipeline_data: &PipelineData,
        buffers_data: &BuffersData,
        descriptor_sets: &[vk::DescriptorSet],
        swapchain_extent: vk::Extent2D,
        render_pass: vk::RenderPass,
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

        let render_area = vk::Rect2D::builder()
            .offset(vk::Offset2D::default())
            .extent(swapchain_extent);

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
            .render_pass(render_pass)
            .framebuffer(framebuffers[image_index])
            .render_area(render_area)
            .clear_values(clear_values);

        unsafe { device.cmd_begin_render_pass(command_buffer, &info, vk::SubpassContents::SECONDARY_COMMAND_BUFFERS) };

        // TODO: change this to support multiple models*
        let model_data = &models_data.models[0];

        let secondary_command_buffers = (0..model_data.instances.len())
            .map(|i| self.update_secondary_command_buffers(
                device,
                framebuffers,
                model_data,
                pipeline_data,
                buffers_data,
                descriptor_sets,
                image_index,
                i,
            ))
            .collect::<Result<Vec<_>, _>>()?;

        unsafe { 
            device.cmd_execute_commands(command_buffer, &secondary_command_buffers[..]);
            device.cmd_end_render_pass(command_buffer);
            device.end_command_buffer(command_buffer)?;
        };

        Ok(())
    }

    pub fn update_secondary_command_buffers(
        &mut self,
        device: &Device,
        framebuffers: &[vk::Framebuffer],
        model_data: &Model,
        pipeline_data: &PipelineData,
        buffers_data: &BuffersData,
        descriptor_sets: &[vk::DescriptorSet],
        image_index: usize,
        model_index: usize,
    ) -> Result<vk::CommandBuffer> {
        self.secondary_command_buffers.resize_with(image_index + 1, Vec::new);
        let command_buffers = &mut self.secondary_command_buffers[image_index];

        while model_index >= command_buffers.len() {
            let allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(self.command_pools[image_index])
                .level(vk::CommandBufferLevel::SECONDARY)
                .command_buffer_count(1);

            let command_buffer = unsafe { device.allocate_command_buffers(&allocate_info)?[0] };
            command_buffers.push(command_buffer);
        }

        let command_buffer = command_buffers[model_index];

        // push-constant model matrix
        let model = model_data.instances[model_index].to_model_matrix();

        let model_bytes = unsafe {
            std::slice::from_raw_parts(
                &model as *const Mat4 as *const u8,
                size_of::<Mat4>()
            )
        };

        let opacity: f32 = 1.0;
        let opacity_bytes = &opacity.to_ne_bytes()[..];

        let inheritance_info = vk::CommandBufferInheritanceInfo::builder()
            .render_pass(pipeline_data.render_pass)
            .subpass(0)
            .framebuffer(framebuffers[image_index]);

        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE)
            .inheritance_info(&inheritance_info);

        unsafe {
            device.begin_command_buffer(command_buffer, &info)?;

            device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline_data.pipeline);
            device.cmd_bind_vertex_buffers(command_buffer, 0, &[buffers_data.interleaved_buffer], &[0]);
            device.cmd_bind_index_buffer(command_buffer, buffers_data.interleaved_buffer, buffers_data.index_offset, vk::IndexType::UINT32);
            
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline_data.pipeline_layout,
                0,
                &[descriptor_sets[image_index]],
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

            device.cmd_draw_indexed(command_buffer, model_data.mesh.indices.len() as u32, 1, 0, 0, 0);
            device.end_command_buffer(command_buffer)?;
        }

        Ok(command_buffer)
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
pub struct TextureData {
    pub texture_image: vk::Image,
	pub texture_image_memory: vk::DeviceMemory,
	pub texture_image_view: vk::ImageView,
    pub texture_sampler: vk::Sampler,
    pub mip_levels: u32,
}

impl TextureData {
    pub fn create(
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        setup_command_buffer: vk::CommandBuffer,
        graphics_queue: vk::Queue,
        texture_path: &str,
    ) -> Result<Self> {
        let (texture_image, texture_image_memory, mip_levels) = create_texture_image(
            instance,
            device,
            physical_device,
            texture_path,
            setup_command_buffer,
            graphics_queue
        )?;
		let texture_image_view = create_texture_image_view(&device, texture_image, mip_levels)?;
        let texture_sampler = create_texture_sampler(&device, mip_levels as f32)?;

        Ok(Self {
            texture_image,
            texture_image_memory,
            texture_image_view,
            texture_sampler,
            mip_levels
        })
    }

    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn destroy(&mut self, device: &Device) {
        device.destroy_sampler(self.texture_sampler, None);
        device.destroy_image_view(self.texture_image_view, None);
		device.destroy_image(self.texture_image, None);
		device.free_memory(self.texture_image_memory, None);
    }
}

#[derive(Clone, Debug)]
pub struct DepthData {
    pub depth_image: vk::Image,
    pub depth_image_memory: vk::DeviceMemory,
    pub depth_image_view: vk::ImageView,
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
        let (depth_image, depth_image_memory, depth_image_view) = create_depth_objects(
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
            depth_image_view
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