use anyhow::Result;

use vulkanalia::prelude::v1_0::*;

use crate::resources::{SetupCommand, ImageResource, create_image, create_image_view};
use crate::gpu::{ComputePipeline, create_storage_image_descriptor_set_layout};
use crate::render::create_storage_image_descriptor_set;
use crate::constants::{BRDF_LUT_SIZE, BRDF_LUT_COMP};

/// Must match `local_size_x` / `local_size_y` in brdf_lut.comp.
const BRDF_LUT_LOCAL_SIZE: u32 = 16;

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

	/// Records the compute work that fills this LUT.
    ///
    /// Writes into the setup command buffer: a transition to `GENERAL` so the
    /// compute shader can write to it, the dispatch itself, then a transition to
    /// `SHADER_READ_ONLY_OPTIMAL` that also orders the write before any later
    /// fragment shader read.
    ///
    /// The pipeline, descriptor pool and set layout are deferred to the setup
    /// batch and they must outlive the submission, but serve no purpose after it.
    ///
    /// ## Arguments
    ///
    /// - `device` ( &[Device] ) - The Vulkan device.
    /// - `setup` ( &mut [SetupCommand] ) - Batch the commands are recorded into.
	pub fn generate(
		&self,
		device: &Device,
		setup: &mut SetupCommand,
	) -> Result<()> {
		// 1- descriptor
		let set_layout = create_storage_image_descriptor_set_layout(device)?;
		let pool = BRDFLut::create_descriptor_pool(device)?;
		let set = create_storage_image_descriptor_set(
			device,
			set_layout,
			pool,
			self.0.vk_image_view
		)?;

		// 2- pipeline
		let pipeline = ComputePipeline::new(device, BRDF_LUT_COMP, &[set_layout])?;

		// 3- record
		// The portion of the image that the barrier covers.
		let subresource_range = vk::ImageSubresourceRange::builder()
			.aspect_mask(vk::ImageAspectFlags::COLOR)
			.base_mip_level(0)
			.level_count(1)
			.base_array_layer(0)
			.layer_count(1);

		// Undefined -> General
		let to_general = vk::ImageMemoryBarrier::builder()
			.old_layout(vk::ImageLayout::UNDEFINED)
			.new_layout(vk::ImageLayout::GENERAL)
			.src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
			.dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
			.src_access_mask(vk::AccessFlags::empty())
			.dst_access_mask(vk::AccessFlags::SHADER_WRITE)
			.image(self.0.vk_image)
			.subresource_range(subresource_range);
		
		// General -> Shader read only + block the read until the write is finish
		let to_read_only = vk::ImageMemoryBarrier::builder()
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .image(self.0.vk_image)
            .subresource_range(subresource_range);

		let groups = BRDF_LUT_SIZE.div_ceil(BRDF_LUT_LOCAL_SIZE);

		unsafe {
			device.cmd_pipeline_barrier(
                setup.command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[] as &[vk::MemoryBarrier],
                &[] as &[vk::BufferMemoryBarrier],
                &[to_general],
            );

			device.cmd_bind_pipeline(
                setup.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.vk_pipeline,
            );

			device.cmd_bind_descriptor_sets(
                setup.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline.vk_layout,
                0,
                &[set],
                &[],
            );

			device.cmd_dispatch(setup.command_buffer, groups, groups, 1);

			device.cmd_pipeline_barrier(
                setup.command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[] as &[vk::MemoryBarrier],
                &[] as &[vk::BufferMemoryBarrier],
                &[to_read_only],
            );
		}

		// 4 - defered cleanup
		setup.defer(move |d| unsafe { pipeline.destroy(d) });
        setup.defer(move |d| unsafe { d.destroy_descriptor_pool(pool, None) });
        setup.defer(move |d| unsafe { d.destroy_descriptor_set_layout(set_layout, None) });
		
		Ok(())
	}

	#[inline]
	pub fn image(&self) -> &ImageResource { &self.0 }

	/// Refactor function that create an "ephemeral" descriptor pool
	/// 
	/// ## Arguments
	/// 
	/// - `device` ( &[Device] ) - The vulkan's device.
	/// 
	/// ## Returns
	/// 
	/// - `Result<vk::DescriptorPool>` - The descriptor pool.
	fn create_descriptor_pool(device: &Device) -> Result<vk::DescriptorPool> {
		let pool_size = vk::DescriptorPoolSize::builder()
            .type_(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1);

        let pool_sizes = &[pool_size];
        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(pool_sizes)
            .max_sets(1);

        Ok(unsafe { device.create_descriptor_pool(&pool_info, None)? })
	}
}