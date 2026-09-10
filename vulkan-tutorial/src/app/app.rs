use anyhow::{Result, anyhow};
use std::time::Instant;

use winit::window::Window;

use vulkanalia::prelude::v1_0::*;
use vulkanalia::window as vk_window;
use vulkanalia::loader::{LibloadingLoader, LIBRARY};
use vulkanalia::vk::{
	ExtDebugUtilsExtensionInstanceCommands,
	KhrSurfaceExtensionInstanceCommands,
	KhrSwapchainExtensionDeviceCommands
};

use super::resources::{
	Persistent,
	SwapchainDependant, 
};

use crate::resources::UniformBuffers;
use crate::input::InputBindings;
use crate::gpu::{
	GPURequirements,
	GPUContext,
	create_instance,
};
use crate::render::{
	DescriptorLayouts,
	UniformBufferObject,
};
use crate::scene::{
	ECSContext,
	SkinningBuffer,
	InstanceBuffer,
};
use crate::setup::{
	init_ecs_context,
};
use crate::constants::{
	DEVICE_EXTENSIONS,
	INPUT_PATH,
	MAX_FRAMES_IN_FLIGHT,
};


#[derive(Debug)]
pub struct AppData {
	pub transient:	SwapchainDependant,
	pub persistent:	Persistent,
}

/// - `messenger` ( Option<[vk::DebugUtilsMessengerEXT]> ) - Validation layer callback, present only when validation is enabled.
pub struct App {
	pub entry:				Entry,
	pub instance:			Instance,
	pub device:				Device,
	pub data:				AppData,
	pub ecs_context:		ECSContext,
	pub input_bindings:		InputBindings,
	pub surface:			vk::SurfaceKHR,

	pub frame:				usize,
	pub start:				Instant,
	pub window_resized:		bool,

	pub messenger:			Option<vk::DebugUtilsMessengerEXT>
}

impl App {
	pub fn new(window: &Window) -> Result<Self> {
		// region load engine parameters
		// TODO: move all the functions bellow into a config file and create a load_config function.
		let mandatory_feats = vk::PhysicalDeviceFeatures::builder()
            .sampler_anisotropy(true)
            .build();

        let optional_feats = vk::PhysicalDeviceFeatures::builder()
            .build();

        let mandatory_queue_flags = vk::QueueFlags::GRAPHICS;

		let gpu_requirements = GPURequirements {
            mandatory_features:     mandatory_feats,
            optional_features:      optional_feats,
            mandatory_extensions:   DEVICE_EXTENSIONS.to_vec(),
            optional_extensions:    (&[] as &[vk::ExtensionName]).to_vec(),
            queue_flags:            mandatory_queue_flags,
        };
		// endregion

        let loader = unsafe { LibloadingLoader::new(LIBRARY)? };
        let entry = unsafe {Entry::new(loader)}.map_err(|b| anyhow!("{}", b))?;
        let (instance, messenger) = create_instance(window, &entry)?;

		let surface = unsafe { vk_window::create_surface(&instance, &window, &window)? };
		let (mut gpu, device) = GPUContext::create(
            &entry,
            &instance,
            surface,
            &gpu_requirements,
            false,
        )?;

		let mut ecs_context = init_ecs_context(&device, &instance, gpu.physical_device)?;
		let layouts = DescriptorLayouts::new(&device)?;
		
		// region app-data
		let transient = SwapchainDependant::new(
			window,
			&instance,
			&device,
			&mut gpu,
			surface,
		)?;

		let persistent = Persistent::new(
			&instance,
			&device,
			&mut ecs_context,
			&transient,
			gpu,
			layouts,
		)?;

		let data = AppData {
			transient,
			persistent,
		};
		// endregion

		let input_bindings = InputBindings::bind_from_file(&INPUT_PATH)?;

		Ok(Self {
			entry,
			instance,
			device,
			data,
			ecs_context,
			input_bindings,
			surface,
			frame: 0,
			start: Instant::now(),
			window_resized: false,
			messenger
		})
	}

	#[rustfmt::skip]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn destroy(&mut self) {
		self.device.device_wait_idle().unwrap();

		// destroy swapchain
		self.data.transient.destroy(&self.device);

		// destroy ECS-system
		if let Some(skinning_buffer) = self.ecs_context.world.get_resource::<SkinningBuffer>() {
			skinning_buffer.destroy(&self.device);
		}
        if let Some(instance_buffer) = self.ecs_context.world.get_resource::<InstanceBuffer>() {
            instance_buffer.destroy(&self.device);
        }

		// destroy persistent-data
		self.data.persistent.destroy(&self.device);

		// destroy app
		self.device.destroy_device(None);
		self.instance.destroy_surface_khr(self.surface, None);
		if let Some(messenger) = self.messenger {
            self.instance.destroy_debug_utils_messenger_ext(messenger, None);
        }
        self.instance.destroy_instance(None);
	}

	fn recreate_swapchain(&mut self, window: &Window) -> Result<()> {
		unsafe {
			self.device.device_wait_idle()?;
			self.data.transient.recreate(
				window,
				&self.device,
				&self.instance,
				self.surface,
				&mut self.data.persistent.gpu
			)?;
		}

		// persistent data change only if the image count is not the same as before
		let image_count = self.data.transient.swapchain.vk_images.len();
		if image_count != self.data.persistent.uniform_buffers.vk_buffers.len() {
			unsafe { self.data.persistent.uniform_buffers.destroy(&self.device) };

			self.data.persistent.uniform_buffers = UniformBuffers::new(
				&self.instance,
				&self.device,
				self.data.persistent.gpu.physical_device,
				image_count,
			)?;

			self.data.persistent.descriptors.update_global_descriptor_sets(
				&self.device,
				&self.data.persistent.uniform_buffers.vk_buffers,
			);

			self.data.persistent.command_recorder.resize_frame_resources(
				&self.device,
				&self.data.persistent.gpu.queue_family_indices,
				image_count
			)?;

			self.data.persistent.sync.images_in_flight.resize(image_count, vk::Fence::null());
		}

		Ok(())
	}

	/// Renders a frame for our Vulkan app.
    pub fn render(&mut self, window: &Window) -> Result<()> {
        let in_flight_fence = self.data.persistent.sync.in_flight_fences[self.frame];

        unsafe {self.device.wait_for_fences(&[in_flight_fence], true, u64::MAX)?};

        let result = unsafe { self.device.acquire_next_image_khr(
            self.data.transient.swapchain.vk_swapchain,
            u64::MAX,
            self.data.persistent.sync.image_available_semaphores[self.frame],
            vk::Fence::null(),
        )};

        let image_index = match result {
            Ok((image_index, _)) => image_index as usize,
            Err(vk::ErrorCode::OUT_OF_DATE_KHR) => return self.recreate_swapchain(window),
            Err(e) => return Err(anyhow!(e)),
        };

        let image_in_flight = self.data.persistent.sync.images_in_flight[image_index];
        if !image_in_flight.is_null() {
            unsafe { self.device.wait_for_fences(&[image_in_flight], true, u64::MAX)? };
        }

        self.data.persistent.sync.images_in_flight[image_index] = in_flight_fence;

        // Update UBO
        let ubo = UniformBufferObject::from_camera(
			&self.data.persistent.camera,
			self.data.transient.swapchain.vk_extent.width as f32 / self.data.transient.swapchain.vk_extent.height as f32,
		);

		unsafe { self.data.persistent.uniform_buffers.write(&self.device, image_index, &ubo)? };

		// Update commands buffers
        self.data.persistent.command_recorder.record(
            &self.device,
			&mut self.ecs_context,
            &self.data.persistent.graphic_pipeline,
            &self.data.persistent.geometry_buffer,
            &self.data.transient.swapchain,
            &self.data.transient.color_attachment,
            &self.data.transient.depth_attachment,
            &self.data.persistent.descriptors,
            self.data.persistent.gpu.max_msaa_samples,
            image_index,
            self.frame
        )?;

        let wait_semaphores = &[self.data.persistent.sync.image_available_semaphores[self.frame]];
        let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = &[self.data.persistent.command_recorder.primary_command_buffers[image_index]];
        let signal_semaphores = &[self.data.persistent.sync.render_finished_semaphores[self.frame]];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores);

        unsafe {
            self.device.reset_fences(&[in_flight_fence])?;

            self.device
                .queue_submit(self.data.persistent.gpu.graphics_queue, &[submit_info], in_flight_fence)?;
        }

        let swapchains = &[self.data.transient.swapchain.vk_swapchain];
        let image_indices = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);

        let result = unsafe { self.device.queue_present_khr(self.data.persistent.gpu.present_queue, &present_info) };
        let changed = result == Ok(vk::SuccessCode::SUBOPTIMAL_KHR) || result == Err(vk::ErrorCode::OUT_OF_DATE_KHR);
        if self.window_resized || changed {
            self.window_resized = false;
            self.recreate_swapchain(window)?;
        } else if let Err(e) = result {
            return Err(anyhow!(e));
        }

        self.frame = (self.frame + 1) % MAX_FRAMES_IN_FLIGHT;

        Ok(())
    }
}