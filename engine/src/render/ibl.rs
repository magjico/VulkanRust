use anyhow::Result;

use vulkanalia::prelude::v1_0::*;

use crate::resources::{ImageResource, create_image, create_image_view};
use crate::constants::BRDF_LUT_SIZE;

/// Pre-integrated split-sum BRDF.
///
/// Red holds the scale applied to F0, green the bias. Generated once at startup
/// by a compute shader and independent of the scene and of the environment.
#[derive(Debug)]
pub struct BRDFLut(ImageResource);

impl BRDFLut {
	pub fn new(
		instance: &Instance,
		device: &Device,
		physical_device: vk::PhysicalDevice,
	) -> Result<Self> {
		let format = vk::Format::R16G16_SFLOAT;

		let (vk_image, vk_image_memory) = create_image(
			instance,
			device,
			physical_device,
			vk::Extent3D { width: BRDF_LUT_SIZE, height: BRDF_LUT_SIZE, depth: 1},
			1,
			vk::SampleCountFlags::_1,
			format,
			vk::ImageTiling::OPTIMAL,
			vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
			vk::MemoryPropertyFlags::DEVICE_LOCAL
		)?;

		let vk_image_view = create_image_view(
			device,
			vk_image,
			format,
			vk::ImageAspectFlags::COLOR,
			1
		)?;

		Ok(Self(ImageResource {
				vk_image,
				vk_image_memory,
				vk_image_view
			})
		)
	}

	/// ## Safety
    ///
    /// The caller must ensure the device is idle and that no descriptor set still
    /// referencing this image is bound by a pending command buffer.
	#[allow(unsafe_op_in_unsafe_fn)]
	pub unsafe fn destroy(&self, device: &Device) {
		self.0.destroy(device);
	}

	/// Runs the compute shader that fills this LUT (see shaders/ibl/brdf_lut.comp), then leaves the image ready
    /// for sampling.
    ///
    /// Records into the shared setup command buffer.
    /// The LUT is suppose to be generated once and never regenerated.
    ///
    /// Must be called while the setup command buffer is recording, and before
    /// a Descriptor writes this image into any descriptor set.
    ///
    /// ## Arguments
    ///
    /// - `device` ( &[Device] ) - The Vulkan device.
    /// - `setup_command_buffer` ( [vk::CommandBuffer] ) - Shared buffer for one-off load-time work.
	pub fn generate(
		&self,
		device: &Device,
		setup_command_buffer: vk::CommandBuffer,
	) -> Result<()> {


		Ok(())
	}

	#[inline]
	pub fn image(&self) -> &ImageResource { &self.0 }
}