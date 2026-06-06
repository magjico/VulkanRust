use std::ptr::copy_nonoverlapping as memcpy;

use anyhow::{Result, anyhow};

use vulkanalia::prelude::v1_0::*;

use crate::resources::{
    create_image, create_image_view, transition_image_layout,
    create_buffer,
    begin_setup_command_buffer, flush_setup_command_buffer,
};
use crate::ops::copy_buffer_to_image;

/// only use to store texture data (for now).
#[derive(Clone, Debug)]
pub struct TextureData {
    pub image: vk::Image,
    pub image_memory: vk::DeviceMemory,
    pub image_view: vk::ImageView,
    pub sampler: vk::Sampler,
    pub mip_levels: u32
}

impl TextureData {
    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn destroy(&mut self, device: &Device) {
        device.destroy_sampler(self.sampler, None);
        device.destroy_image_view(self.image_view, None);
        device.destroy_image(self.image, None);
        device.free_memory(self.image_memory, None);
    }
}

/// Create a texture image and its memory.
/// 
/// ## Arguments
/// 
/// - `instance` (&[`Instance`]) - Vulkan instance.
/// - `device` (&[`Device`]) - Vulkan device.
/// - `physical_device` ([`vk::PhysicalDevice`]) - Used physical device.
/// - `setup_command_buffer` ([`vk::CommandBuffer`]) - A command buffer to execute command from.
/// - `graphics_queue` ([`vk::Queue`]) - A graphics queue to the send the command to the gpu.
/// 
/// ## Returns
/// 
/// - `Result<(vk::Image, vk::DeviceMemory, u32)>` - The texture image, its device memory and its mip maps levels.
pub fn create_texture_image(
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    setup_command_buffer: vk::CommandBuffer,
    graphics_queue: vk::Queue,
    texture_extent: vk::Extent3D,
    mip_levels: u32,
    pixels: &[u8],
    format: vk::Format,
) -> Result<(vk::Image, vk::DeviceMemory)> {
    let staging_size = pixels.len() as u64;

    let (staging_buffer, staging_buffer_memory) = create_buffer(
		instance,
		device,
		physical_device,
		staging_size,
		vk::BufferUsageFlags::TRANSFER_SRC,
		vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE
	)?;

	let memory = unsafe { device.map_memory(
		staging_buffer_memory,
		0,
		staging_size,
		vk::MemoryMapFlags::empty()
	)?};

    unsafe {
        memcpy(pixels.as_ptr(), memory.cast(), pixels.len());
        device.unmap_memory(staging_buffer_memory);
    }

	let (texture_image, texture_image_memory) = create_image(
        instance,
        device,
        physical_device,
        texture_extent,
        mip_levels,
		vk::SampleCountFlags::_1,
        format,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    begin_setup_command_buffer(&device, setup_command_buffer)?;

	transition_image_layout(
		device,
		setup_command_buffer,
		texture_image,
		vk::ImageLayout::UNDEFINED,
		vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        mip_levels,
	)?;

	copy_buffer_to_image(
		device,
		setup_command_buffer,
		staging_buffer,
		texture_image,
		texture_extent.width,
		texture_extent.height
	)?;

	generate_mipmaps(
		instance,
		device,
		physical_device,
        setup_command_buffer,
		texture_image,
		vk::Format::R8G8B8A8_SRGB,
		texture_extent.width,
		texture_extent.height, 
		mip_levels
	)?;

    flush_setup_command_buffer(&device, setup_command_buffer, graphics_queue)?;

    unsafe {
        device.destroy_buffer(staging_buffer, None);
        device.free_memory(staging_buffer_memory, None);
    }

	Ok((texture_image, texture_image_memory))
}

/// Create an image view of a texture image
/// 
/// # Arguments
/// 
/// - `device` (&[`Device`]) - The vulkan device.
/// - `texture_image` ([`vk::Image`]) - The source texture image.
/// - `mip_levels` (`u32`) - The number of mip levels for the source.
/// 
/// # Returns
/// 
/// - `Result<vk::ImageView>` - The texture view.
pub fn create_texture_image_view(
    device: &Device,
    texture_image: vk::Image,
    mip_levels: u32,
) -> Result<vk::ImageView> {
	let texture_image_view = create_image_view(
        device,
        texture_image,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageAspectFlags::COLOR,
        mip_levels,
    )?;

	Ok(texture_image_view)
}

/// Create a texture image sampler which represent the state of a texture image,
/// which is used to read texture data and apply filtering and others transformation for shaders.
/// 
/// ## Arguments
/// 
/// - `device` (&[`Device`]) - Vulkan device.
/// - `mip_levels` (`f32`) - Texture mipmap levels.
/// 
/// ## Returns
/// 
/// - Result<[`vk::Sampler`]>.
/// ```
pub fn create_texture_sampler(device: &Device, mip_levels: f32) -> Result<vk::Sampler> {
    let info = vk::SamplerCreateInfo::builder()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::REPEAT)
        .address_mode_v(vk::SamplerAddressMode::REPEAT)
        .address_mode_w(vk::SamplerAddressMode::REPEAT)
        .anisotropy_enable(true)
        .max_anisotropy(16.0)
        .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
        .unnormalized_coordinates(false)
        .compare_enable(false)
        .compare_op(vk::CompareOp::ALWAYS)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .mip_lod_bias(0.0)
        .min_lod(0.0)
        .max_lod(mip_levels);

    let texture_sampler = unsafe { device.create_sampler(&info, None)? };

    Ok(texture_sampler)
}

/// Generate the number of mipmaps ask for a given texture image.
/// 
/// ## Arguments
/// 
/// - `instance` (&[`Instance`]) - The Vulkan instance.
/// - `device` (&[`Device`]) - The Vulkan device.
/// - `physical_device` ([`vk::PhysicalDevice`]) - The physical device (usualy gpu).
/// - `setup_command_buffer` ([`vk::CommandBuffer`]) - A command buffer to execute command from.
/// - `image` ([`vk::Image`]) - The src texture image.
/// - `format` ([`vk::Format`]) - The wanted image format.
/// - `width` (`u32`) - The src image width.
/// - `height` (`u32`) - The src image height.
/// - `mip_levels` (`u32`) - number of wanted mip levels.
///
/// ## Errors
/// 
/// The physical device does not support linear sampling for this format.
pub fn generate_mipmaps(
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    setup_command_buffer: vk::CommandBuffer,
    image: vk::Image,
	format: vk::Format,
    width: u32,
    height: u32,
    mip_levels: u32,
) -> Result<()> {
    if !supports_linear_blitting(instance, physical_device, format) {
        return Err(anyhow!("Texture image format does not support linear blitting!"));
    }

    let subresource = vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_array_layer(0)
        .layer_count(1)
        .level_count(1);

    let mut barrier = vk::ImageMemoryBarrier::builder()
        .image(image)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .subresource_range(subresource);

    let mut mip_width = width;
    let mut mip_height = height;

    for i in 1..mip_levels {
        barrier.subresource_range.base_mip_level = i - 1;
        barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        barrier.new_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
        barrier.dst_access_mask = vk::AccessFlags::TRANSFER_READ;

        unsafe {
            device.cmd_pipeline_barrier(
                setup_command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[] as &[vk::MemoryBarrier],
                &[] as &[vk::BufferMemoryBarrier],
                &[barrier]
            )
        };

        let src_subresource = vk::ImageSubresourceLayers::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(i - 1)
            .base_array_layer(0)
            .layer_count(1);

        let dst_subresource = vk::ImageSubresourceLayers::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(i)
            .base_array_layer(0)
            .layer_count(1);

        let blit = vk::ImageBlit::builder()
            .src_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: mip_width as i32,
                    y: mip_height as i32,
                    z: 1
                },
            ])
            .src_subresource(src_subresource)
            .dst_offsets([
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: (if mip_width > 1 { mip_width / 2 } else { 1 }) as i32,
                    y: (if mip_height > 1 { mip_height / 2 } else { 1 }) as i32,
                    z: 1,
                },
            ])
            .dst_subresource(dst_subresource);

        unsafe {
            device.cmd_blit_image(
                setup_command_buffer,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[blit],
                vk::Filter::LINEAR
            )
        };

		barrier.old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
		barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
		barrier.src_access_mask = vk::AccessFlags::TRANSFER_READ;
		barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

        unsafe {
            device.cmd_pipeline_barrier(
                setup_command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[] as &[vk::MemoryBarrier],
                &[] as &[vk::BufferMemoryBarrier],
                &[barrier]
            )
        };

		if mip_width > 1 {
			mip_width /= 2;
		}

		if mip_height > 1 {
			mip_height /= 2;
		}
    }

	barrier.subresource_range.base_mip_level = mip_levels - 1;
	barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
	barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
	barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
	barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

    unsafe {
        device.cmd_pipeline_barrier(
            setup_command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[barrier],
        )
    };

    Ok(())
}

/// Check if the physical device support blitting for a specific format.
fn supports_linear_blitting(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    format: vk::Format
) -> bool {
    unsafe { instance
            .get_physical_device_format_properties(physical_device, format)
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR) }
}