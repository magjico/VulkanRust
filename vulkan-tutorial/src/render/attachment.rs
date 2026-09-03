use anyhow::Result;

use vulkanalia::prelude::v1_0::*;

use crate::resources::{create_image, create_image_view, get_supported_format};

#[derive(Debug)]
pub struct AttachmentImage {
	pub vk_image: vk::Image,
	pub vk_image_memory: vk::DeviceMemory,
	pub vk_image_view: vk::ImageView,
}

impl AttachmentImage {
	#[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn destroy(&self, device: &Device) {
		device.destroy_image_view(self.vk_image_view, None);
        device.free_memory(self.vk_image_memory, None);
        device.destroy_image(self.vk_image, None);
	}
}

#[derive(Debug)]
pub struct ColorAttachment(AttachmentImage);

impl ColorAttachment {
	/// Create a 3-color attachment:
	/// 
	/// ## Arguments
	/// 
	/// - `instance` ( &[Instance] ) - The Vulkan application's instance.
	/// - `device` ( &[Device] ) - The Vulkan's device.
	/// - `physical_device` ( [vk::PhysicalDevice] ) - Vulkan physical device.
	/// - `width` (`u32`) - wanted image width.
	/// - `height` (`u32`) - wanted image height.
	/// - `sample_count` ( [vk::SampleCountFlags] ) - wanted image sampling (AA).
	/// - `format` ( [vk::Format] ) - wanted image format (for example B8G8R8A8_SRGB).
	pub fn new(
		instance: &Instance,
		device: &Device,
		physical_device: vk::PhysicalDevice,
		extent_width: u32,
        extent_height: u32,
        samples_count: vk::SampleCountFlags,
        vk_format: vk::Format,  
	) -> Result<Self> {
		let (vk_image, vk_image_memory) = create_image(
			instance,
			device,
			physical_device,
			vk::Extent3D { width: extent_width, height: extent_height, depth: 1},
			1,
			samples_count, 
			vk_format,
			vk::ImageTiling::OPTIMAL,
			vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
			vk::MemoryPropertyFlags::DEVICE_LOCAL
		)?;

		let vk_image_view = create_image_view(
			device,
			vk_image,
			vk_format,
			vk::ImageAspectFlags::COLOR,
			1
		)?;

		Ok(Self {
			0: AttachmentImage {
				vk_image,
				vk_image_memory,
				vk_image_view
			}
		})
	}

	#[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn destroy(&self, device: &Device) {
		self.0.destroy(device);
	}

	#[inline]
	pub fn attachment(&self) -> &AttachmentImage { &self.0 }
}

#[derive(Debug)]
pub struct DepthAttachment {
	pub attachment: AttachmentImage,
	pub vk_format: vk::Format,
}

impl DepthAttachment {
	/// Create a depth attachment
	/// 
	/// ## Arguments
	/// 
	/// - `instance` ( &[Instance] ) - The Vulkan instance.
	/// - `device` ( &[Device] ) - The Vulkan device.
	/// - `physical_device` ( [vk::PhysicalDevice] ) - The physical device (gpu).
	/// - `width` ( `u32` ) - wanted image width.
	/// - `height` ( `u32` ) - wanted image height.
	/// - `samples_count` ( [vk::SampleCountFlags] ) - number of multi-sampling to generate.
	pub fn new(
		instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        extent_width: u32,
        extent_height: u32,
        samples_count: vk::SampleCountFlags,
	) -> Result<Self> {
		let vk_format = DepthAttachment::get_depth_format(instance, physical_device)?;

		let (vk_image, vk_image_memory) = create_image(
			instance,
			device,
			physical_device,
			vk::Extent3D { width: extent_width, height: extent_height, depth: 1 },
			1,
			samples_count,
			vk_format,
			vk::ImageTiling::OPTIMAL,
			vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
			vk::MemoryPropertyFlags::DEVICE_LOCAL
		)?;

    	let vk_image_view = create_image_view(device, vk_image, vk_format, vk::ImageAspectFlags::DEPTH, 1)?;

		Ok(Self {
			attachment: AttachmentImage { vk_image, vk_image_memory, vk_image_view },
			vk_format
		})
	}

	#[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn destroy(&self, device: &Device) {
		self.attachment.destroy(device);
	}

	fn get_depth_format(instance: &Instance, physical_device: vk::PhysicalDevice) -> Result<vk::Format> {
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
}