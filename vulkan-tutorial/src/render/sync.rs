use anyhow::Result;

use vulkanalia::prelude::v1_0::*;

/// Create all the sync objects necessary for the render pipeline
/// 
/// ## Arguments
/// 
/// - `device` ( &[Device] ) - Vulkan device.
/// - `max_frame_in_flight` (`usize`) - max supported frame in flight for assigning semaphores and fences.
/// - `swapchain_images_count` (`usize`) - max number of swapchain image (at the same time) for fences.
/// 
/// # Returns
/// 
/// `Result<(Vec<vk::Semaphore>, Vec<vk::Semaphore>, Vec<vk::Fence>, Vec<vk::Fence>)>`:
/// - First a `Vec<vk::Semaphore>` for **synchronising the KHR image retrieval**.
/// - Second a `Vec<vk::Semaphore>` for **synchronising when an image finish to render (so we can use it for something else)**.
/// - Third a `Vec<vk::Fence>` to **await the GPU operation on in flight frame before rendering them**.
/// - Fourth a `Vec<vk::Fence>` to **protect a swapchain image to be acquire by 2 differents frames in flight**.
pub fn create_sync_objects(
    device: &Device,
    max_frame_in_flight: usize,
    swapchain_images_count: usize,
) -> Result<(Vec<vk::Semaphore>, Vec<vk::Semaphore>, Vec<vk::Fence>, Vec<vk::Fence>)> {
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

        images_in_flight = (0..swapchain_images_count).map(|_| vk::Fence::null()).collect();
    }

    Ok((
        image_available_semaphores,
        render_finished_semaphores,
        in_flight_fences,
        images_in_flight
    ))
}