use anyhow::Result;

use vulkanalia::prelude::v1_0::*;

/// Copy a buffer to an image.
/// 
/// ## Arguments
/// 
/// - `device` (&[`Device`]) - Vulkan Device.
/// - `setup_command_buffer` ([`vk::CommandBuffer`]).
/// - `buffer` ([`vk::Buffer`]) - source buffer.
/// - `image` ([`vk::Image`]) - destination image.
/// - `width` (`u32`) - image width.
/// - `height` (`u32`) - image height.
/// ```
pub fn copy_buffer_to_image(
	device: &Device,
	setup_command_buffer: vk::CommandBuffer,
	buffer: vk::Buffer,
	image: vk::Image,
	width: u32,
	height: u32
) -> Result<()> {
	// Buffer parameters setup
	let subresource = vk::ImageSubresourceLayers::builder()
		.aspect_mask(vk::ImageAspectFlags::COLOR)
		.mip_level(0)
		.base_array_layer(0)
		.layer_count(1);

	let region = vk::BufferImageCopy::builder()
		.buffer_offset(0)
		.buffer_row_length(0)
		.buffer_image_height(0)
		.image_subresource(subresource)
		.image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
		.image_extent(vk::Extent3D { width, height, depth: 1 });

	// Buffer record
    unsafe {
        device.cmd_copy_buffer_to_image(
            setup_command_buffer,
            buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region]
        );
    }

	Ok(())
}