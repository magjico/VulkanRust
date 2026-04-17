use anyhow::Result;

use vulkanalia::prelude::v1_0::*;

use crate::resources::{create_image, create_image_view};

/// Create 3 color objects:
/// - a color image.
/// - a color image memory.
/// - a color image view. 
/// 
/// # Arguments
/// 
/// - `instance` (&[`vk::Instance`]) - The Vulkan application's instance.
/// - `device` (&[`Device`]) - The Vulkan's device.
/// - `physical_device` ([`vk::PhysicalDevice`]) - Describe this parameter.
/// - `width` (`u32`) - wanted image width.
/// - `height` (`u32`) - wanted image height.
/// - `sample_count` ([`vk::SampleCountFlags`]) - wanted image sampling (AA).
/// - `format` ([`vk::Format`]) - wanted image format (for example B8G8R8A8_SRGB).
/// 
/// # Returns
/// 
/// - `Result<(vk::Image, vk::DeviceMemory, vk::ImageView)>` - image, memory, view.
/// ```
pub fn create_color_objects(
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    width: u32,
    height: u32,
    sample_count: vk::SampleCountFlags,
    format: vk::Format
) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView)> {
	let (color_image, color_image_memory) = create_image(
		instance,
		device,
		physical_device,
        width,
		height,
		1,
		sample_count, 
		format,
		vk::ImageTiling::OPTIMAL,
		vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
		vk::MemoryPropertyFlags::DEVICE_LOCAL
	)?;

	let color_image_view = create_image_view(
		device,
		color_image,
		format,
		vk::ImageAspectFlags::COLOR,
		1
	)?;

	Ok((color_image, color_image_memory, color_image_view))
}