use anyhow::{Result, anyhow};

use vulkanalia::prelude::v1_0::*;

use crate::gpu::get_memory_type_index;

/// Create an [`vk::Image`] and its attached [`vk::DeviceMemory`].
/// 
/// ## Arguments
/// 
/// - `instance` (&[`Instance`]) - Vulkan instance.
/// - `device` (&[`Device`]) - Vulkan device.
/// - `physical_device` ([`vk::PhysicalDevice`]) - GPU.
/// - `extent` ( [vk::Extent3D] ) - image extent (width, height and depth). If the texture is 2D then depth value should be 1.
/// - `mip_levels` (`u32`) - image mip levels (usualy 1 except for texture).
/// - `samples` ([`vk::SampleCountFlags`]) - sampling count (flags) for aliasing.
/// - `format` ([`vk::Format`]) - image format.
/// - `tiling` ([`vk::ImageTiling`]) - image tiling.
/// - `usage` ([`vk::ImageUsageFlags`]) - Describe the futur usages of this image (for example COLOR_ATTACHMENT).
/// - `properties` ([`vk::MemoryPropertyFlags`]) - Memory property of the device memory of this image.
/// 
/// ## Returns
/// 
/// - `Result<(vk::Image, vk::DeviceMemory)>`.
/// 
/// ## Examples
/// 
/// ```
/// use crate::resources::create_image;
/// 
/// let (color_image, color_image_memory) = create_image(
/// 	instance,
/// 	device,
/// 	physical_device,
///     swapchain_extent.width,
/// 	swapchain_extent.height,
/// 	1,
/// 	vk::SampleCountFlags::_64, 
/// 	vk::Format::B8G8R8A8_SRGB,
/// 	vk::ImageTiling::OPTIMAL,
/// 	vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
/// 	vk::MemoryPropertyFlags::DEVICE_LOCAL
/// )?;
/// ```
pub fn create_image(
	instance: &Instance,
	device: &Device,
	physical_device: vk::PhysicalDevice,
	extent: vk::Extent3D,
    mip_levels: u32,
	samples: vk::SampleCountFlags,
	format: vk::Format,
	tiling: vk::ImageTiling,
	usage: vk::ImageUsageFlags,
	properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Image, vk::DeviceMemory)> {
	// TODO: manage the 1D texture case.
	let image_type = if extent.depth > 1 {
		vk::ImageType::_3D
	} else  {
		vk::ImageType::_2D
	};

	let info = vk::ImageCreateInfo::builder()
		.image_type(image_type)
		.extent(extent)
		.mip_levels(mip_levels)
		.array_layers(1)
		.format(format)
		.tiling(tiling)
		.initial_layout(vk::ImageLayout::UNDEFINED)
		.usage(usage)
		.sharing_mode(vk::SharingMode::EXCLUSIVE)
		.samples(samples)
		.flags(vk::ImageCreateFlags::empty());

    unsafe {
        let image = device.create_image(&info, None)?;
        let requirements = device.get_image_memory_requirements(image);

        let info = vk::MemoryAllocateInfo::builder()
            .allocation_size(requirements.size)
            .memory_type_index(get_memory_type_index(
                instance,
                physical_device,
                properties,
                requirements
            )?);
        
        let image_memory = device.allocate_memory(&info, None)?;
        device.bind_image_memory(image, image_memory, 0)?;
        Ok((image, image_memory))
    }
}

/// Change an image layout state for GPU synchronization.
/// 
/// ## Arguments
/// 
/// - `device` (`&Device`) - Vulkan Device.
/// - `setup_command_buffer` ([`vk::CommandBuffer`]) - to record operation into.
/// - `image` ([`vk::Image`]) - source image.
/// - `old_layout` ([`vk::ImageLayout`]) - current source image layout.
/// - `new_layout` ([`vk::ImageLayout`]) - wanted source image layout.
/// - `mip_levels` (`u32`) - mip level of the image.
pub fn transition_image_layout(
	device: &Device,
	setup_command_buffer: vk::CommandBuffer,
	image: vk::Image,
	old_layout: vk::ImageLayout,
	new_layout: vk::ImageLayout,
    mip_levels: u32,
) -> Result<()> {
	// Transition barrier masks
	let (
		src_access_mask,
		dst_access_mask,
		src_stage_mask,
		dst_stage_mask,
	) = match (old_layout, new_layout) {
		(vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
			vk::AccessFlags::empty(),
			vk::AccessFlags::TRANSFER_WRITE,
			vk::PipelineStageFlags::TOP_OF_PIPE,
			vk::PipelineStageFlags::TRANSFER,
		),
		(vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
			vk::AccessFlags::TRANSFER_WRITE,
			vk::AccessFlags::SHADER_READ,
			vk::PipelineStageFlags::TRANSFER,
			vk::PipelineStageFlags::FRAGMENT_SHADER,
		),
		_ => return Err(anyhow!("Unsupported image layout transition!")),
	};

	// Access Parameters
	let subresource = vk::ImageSubresourceRange::builder()
		.aspect_mask(vk::ImageAspectFlags::COLOR)
		.base_mip_level(0)
		.level_count(mip_levels)
		.base_array_layer(0)
		.layer_count(1);

	// Sync
	let barrier = vk::ImageMemoryBarrier::builder()
		.old_layout(old_layout)
		.new_layout(new_layout)
		.src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
		.dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
		.image(image)
		.subresource_range(subresource)
		.src_access_mask(src_access_mask)
		.dst_access_mask(dst_access_mask);

	// Commands
    unsafe {
        device.cmd_pipeline_barrier(
            setup_command_buffer,
            src_stage_mask,
            dst_stage_mask,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[barrier],
	    );
    }

	Ok(())
}

/// Create a 2D view of a [`vk::Image`].
/// 
/// ## Arguments
/// 
/// - `device` (&[`Device`]) - The Vulkan device.
/// - `image` ([`vk::Image`]) - Image from which the view is created.
/// - `format` ([`vk::Format`]) - The image format, generally the same as the **swapchain**.
/// - `aspect` ([`vk::ImageAspectFlags`]).
/// - `mip_levels` (`u32`) - number of mip levels of the image (useful for texture).
/// 
/// ## Returns
/// 
/// - Result<[`vk::ImageView`]>.
///  
/// ## Examples
/// 
/// ```
/// use graphic_env::resources::create_image_view;
/// 
/// let view = create_image_view(
///     my_device,
///     my_image,
///     vk::Format::B8G8R8A8_SRGB,
///     1
/// );
/// ```
pub fn create_image_view(
    device: &Device,
    image: vk::Image,
    format: vk::Format,
    aspect: vk::ImageAspectFlags,
    mip_levels: u32,
) -> Result<vk::ImageView> {
    let subresource_range = vk::ImageSubresourceRange::builder()
        .aspect_mask(aspect)
        .base_mip_level(0)
        .level_count(mip_levels)
        .base_array_layer(0)
        .layer_count(1);

    let info = vk::ImageViewCreateInfo::builder()
        .image(image)
        .view_type(vk::ImageViewType::_2D)
        .format(format)
        .subresource_range(subresource_range);

    Ok(unsafe {device.create_image_view(&info, None)?})
}

/// From a list of `candidates` search for the first [`vk::Format`] in it for a [`vk::Image`].
/// This format **must** respect the choosen `tiling` and must be compatible with our `physical_device` features.
/// 
/// ## Arguments
/// 
/// - `instance` (&[`Instance`]) - A Vulkan Instance.
/// - `physical_device` ([`vk::PhysicalDevice`]) - Our physical device.
/// - `candidates` (&[[`vk::Format`]]) - All the possible [`vk::Format`] to search from, classified if possible from the most wanted (index 0) to the less wanted (max index).
/// - `tiling` ([`vk::ImageTiling`]) - The wanted image tiling with this image format.
/// - `features` ([`vk::ImageTiling`]) - The needed feature to implement with this image format.
/// 
/// ## Returns
/// 
/// - `Result<vk::Format>` - The first matching format from the `candidates` if possible.
/// 
/// ## Errors
/// 
/// No image format matched.
/// 
/// ## Examples
/// 
/// ```
/// use graphic_env::resources::get_supported_format;
/// 
/// let candidates = &[
///     vk::Format::D32_SFLOAT,
///     vk::Format::D32_SFLOAT_S8_UINT,
///     vk::Format::D24_UNORM_S8_UINT,
/// ];
/// 
/// let format = get_supported_format(
///     my_vulkan_instance,
///     my_physical_device,
///     candidates,
///     vk::ImageTiling::OPTIMAL,
///     vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT
/// );
/// ```
pub fn get_supported_format(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    candidates: &[vk::Format],
    tiling: vk::ImageTiling,
    features: vk::FormatFeatureFlags,
) -> Result<vk::Format> {
    candidates
        .iter()
        .cloned()
        .find(|f | {
            let properties = unsafe { instance.get_physical_device_format_properties(physical_device, *f) };
            match tiling {
                vk::ImageTiling::LINEAR => properties.linear_tiling_features.contains(features),
                vk::ImageTiling::OPTIMAL => properties.optimal_tiling_features.contains(features),
                _ => false,
            }
        })
        .ok_or_else(|| anyhow!("Failed to find a supported format!"))
}