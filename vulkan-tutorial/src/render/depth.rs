use anyhow::Result;

use vulkanalia::prelude::v1_0::*;

use crate::resources::{get_supported_format, create_image, create_image_view};

/// Create 4 depths objects:
/// - Depth image
/// - Depth image memory to link it to the swapchain
/// - Depth image view, the image view.
/// - The depth format.
/// 
/// ## Arguments
/// 
/// - `instance` (&[`Instance`]) - The Vulkan instance.
/// - `device` (&[`Device`]) - The Vulkan device.
/// - `physical_device` ([`vk::PhysicalDevice`]) - The physical device (gpu).
/// - `width` (`u32`) - wanted image width.
/// - `height` (`u32`) - wanted image height.
/// - `samples_count` ([`vk::SampleCountFlags`]) - number of multi-sampling to generate.
/// 
/// ## Returns
/// 
/// - `Result<(vk::Image, vk::DeviceMemory, vk::ImageView, vk::Format)>`.
pub fn create_depth_objects(
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    width: u32,
    height: u32,
    samples_count: vk::SampleCountFlags,
) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView, vk::Format)> {
    let format = get_depth_format(instance, physical_device)?;

    let (depth_image, depth_image_memory) = create_image(
        instance,
        device,
        physical_device,
        vk::Extent3D { width, height, depth: 1 },
        1,
		samples_count,
        format,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL
    )?;

    let depth_image = depth_image;
    let depth_image_memory = depth_image_memory;
    let depth_image_view = create_image_view(device, depth_image, format, vk::ImageAspectFlags::DEPTH, 1)?;

    Ok((depth_image, depth_image_memory, depth_image_view, format))
}

pub fn get_depth_format(instance: &Instance, physical_device: vk::PhysicalDevice) -> Result<vk::Format> {
    let candidates = &[
        vk::Format::D32_SFLOAT,
        vk::Format::D32_SFLOAT_S8_UINT,
        vk::Format::D24_UNORM_S8_UINT,
    ];

    get_supported_format(
        instance,
        physical_device,
        candidates,
        vk::ImageTiling::OPTIMAL,
        vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT
    )
}