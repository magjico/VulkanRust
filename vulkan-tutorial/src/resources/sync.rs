use anyhow::Result;

use vulkanalia::prelude::v1_0::*;

/// Store the Vulkan **synchronization primitives**.
/// 
/// ## Fields
/// 
/// - `image_available_semaphores` ( Vec<[vk::Semaphore]> ) - sync GPU <=> GPU, between acquisition and rendering (for **synchronising the KHR image retrieval**).
/// - `render_finished_semaphores` ( Vec<[vk::Semaphore]> ) - sync GPU <=> GPU, between rendering and presentation (for **synchronising when an image finish to render**).
/// - `in_flight_fences` ( Vec<[vk::Fence]> ) - sync GPU <=> CPU, for each frame in flight (to **await the GPU operation on in flight frame before rendering them**).
/// - `images_in_flight` ( Vec<[vk::Fence]> ) - Doesn't add any synchronisation object, refer to in_flight_fences, for each swapchain-images (to **protect a swapchain image to be acquire by 2 differents frames in flight**).
#[derive(Debug)]
pub struct FrameSync {
	pub image_available_semaphores: Vec<vk::Semaphore>,
    pub render_finished_semaphores: Vec<vk::Semaphore>,
    pub in_flight_fences: Vec<vk::Fence>,
    pub images_in_flight: Vec<vk::Fence>,
}

impl FrameSync {
	/// ## Arguments
	/// 
	/// - `device` ( &[Device] ) - Vulkan device..
	/// - `max_frame_in_flight` ( `usize` ) - max supported frame in flight for assigning semaphores and fences.
	/// - `image_count` ( `usize` ) - max number of swapchain image (at the same time) for fences.
	pub fn new(
		device: &Device,
		max_frame_in_flight: usize,
		image_count: usize
	) -> Result<Self> {
		let semaphore_info = vk::SemaphoreCreateInfo::builder();
		let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

		let mut image_available_semaphores: Vec<vk::Semaphore> = Vec::new();
		let mut render_finished_semaphores: Vec<vk::Semaphore> = Vec::new();
		let mut in_flight_fences: Vec<vk::Fence> = Vec::new();
		let images_in_flight: Vec<vk::Fence>;

		unsafe {
			for _ in 0..max_frame_in_flight {
				image_available_semaphores
					.push(device.create_semaphore(&semaphore_info, None)?);
				render_finished_semaphores
					.push(device.create_semaphore(&semaphore_info, None)?);

				in_flight_fences.push(device.create_fence(&fence_info, None)?);
			}

			images_in_flight = (0..image_count).map(|_| vk::Fence::null()).collect();
		}

		Ok(Self {
			image_available_semaphores,
			render_finished_semaphores,
			in_flight_fences,
			images_in_flight
		})
	}

	#[rustfmt::skip]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn destroy(&self, device: &Device) {
        self.in_flight_fences.iter().for_each(|f| device.destroy_fence(*f, None)); // also free images_in_flight
        self.render_finished_semaphores.iter().for_each(|s| device.destroy_semaphore(*s, None));
        self.image_available_semaphores.iter().for_each(|s| device.destroy_semaphore(*s, None));
    }
}