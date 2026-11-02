use anyhow::Result;
use std::ptr::copy_nonoverlapping as memcpy;

use vulkanalia::prelude::v1_0::*;

use crate::resources::{
	SetupCommand,
	ImageResource,
	ImageBuilder,
	create_image_view,
	create_buffer,
	transition_image_layout,
};
use crate::gpu::{
	Pipeline,
	create_storage_image_descriptor_set_layout,
	create_image_to_image_descriptor_set_layout,
};
use crate::assets::EquirectImage;
use crate::ops::copy_buffer_to_image;
use crate::render::{
	create_storage_image_descriptor_set,
	create_image_to_image_descriptor_set,
};
use crate::constants::{
	BRDF_LUT_COMP, BRDF_LUT_SIZE, ENV_CUBEMAP_SIZE, EQUIRECT_TO_CUBEMAP_COMP,
};

/// Workgroup size for the IBL compute shaders, which all walk a 2D grid of texels.
/// 
/// Must match `local_size_x` / `local_size_y` in the shader .comp
const IBL_LOCAL_SIZE: u32 = 16;

/// Pre-integrated split-sum BRDF.
///
/// Red holds the scale applied to F0, green the bias. Generated once at startup
/// by a compute shader and independent of the scene and of the environment.
#[derive(Debug)]
pub struct BRDFLut(ImageResource);

/// The environment panorama, projected onto the six faces of a cube.
///
/// Built once at startup from an equirectangular HDR image. Sampled directly by
/// the skybox, and convolved into the irradiance and prefiltered maps.
#[derive(Debug)]
pub struct EnvironmentCubemap {
	pub resource: ImageResource,
	pub face_size: u32,
}

impl BRDFLut {
	pub fn new(
		instance: &Instance,
		device: &Device,
		physical_device: vk::PhysicalDevice,
	) -> Result<Self> {
		let format = vk::Format::R16G16_SFLOAT;

		let (vk_image, vk_image_memory) = ImageBuilder::new(
				vk::Extent3D { width: BRDF_LUT_SIZE, height: BRDF_LUT_SIZE, depth: 1},
				format,
				vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED
			)
			.build(instance, device, physical_device)?;

		let vk_image_view = create_image_view(
			device,
			vk_image,
			format,
			vk::ImageAspectFlags::COLOR,
			vk::ImageViewType::_2D,
			1,
			1,
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
		let pipeline = Pipeline::new_compute(device, BRDF_LUT_COMP, &[set_layout])?;

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

		let groups = BRDF_LUT_SIZE.div_ceil(IBL_LOCAL_SIZE);

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

impl EnvironmentCubemap {
	pub fn new(
		instance: &Instance,
		device: &Device,
		physical_device: vk::PhysicalDevice,
		face_size: u32,
	) -> Result<Self> {
		let (vk_image, vk_image_memory) = ImageBuilder::new(
				vk::Extent3D { width: face_size, height: face_size, depth: 1 },
				vk::Format::R16G16B16A16_SFLOAT,
				vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED
			)
			.cube()
			.build(instance, device, physical_device)?;

		let vk_image_view = create_image_view(
			device,
			vk_image,
			vk::Format::R16G16B16A16_SFLOAT,
			vk::ImageAspectFlags::COLOR,
			vk::ImageViewType::CUBE,
			1, 
			6
		)?;

		Ok(Self {
			resource: ImageResource { vk_image, vk_image_memory, vk_image_view },
			face_size,
		})
	}

	/// Records the compute work that projects the equirectangular panorama onto
    /// the six cube faces.
    ///
    /// Uploads the panorama as a temporary `R32G32B32A32_SFLOAT` texture, then
    /// dispatches one invocation per texel of each face. Everything created here
    /// except the cubemap itself is deferred to the setup batch.
    ///
    /// ## Arguments
    ///
    /// - `instance` ( &[Instance] ) - The Vulkan instance.
    /// - `device` ( &[Device] ) - The Vulkan device.
    /// - `physical_device` ( [vk::PhysicalDevice] ) - Used to allocate the staging buffer and source image.
    /// - `setup` ( &mut [SetupCommands] ) - Batch the commands are recorded into.
    /// - `equirect` ( &[Equirect] ) - Panorama pixels, in `f32` RGBA.
    /// - `sampler` ( [vk::Sampler] ) - Sampler the compute shader reads the panorama with.
	pub fn generate(
		&self,
		instance: &Instance,
		device: &Device,
		physical_device: vk::PhysicalDevice,
		setup: &mut SetupCommand,
		equirect: &EquirectImage,
		sampler: vk::Sampler,
	) -> Result<()> {
		const SRC_FORMAT: vk::Format = vk::Format::R32G32B32A32_SFLOAT;

		// 1 - Staging Buffer
		let staging_size = std::mem::size_of_val(equirect.pixels.as_slice()) as vk::DeviceSize;

        let (staging_buffer, staging_memory) = create_buffer(
            instance,
            device,
            physical_device,
            staging_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        unsafe {
            let ptr = device.map_memory(staging_memory, 0, staging_size, vk::MemoryMapFlags::empty())?;
            memcpy(equirect.pixels.as_ptr(), ptr.cast(), equirect.pixels.len());
            device.unmap_memory(staging_memory);
        }

		// 2 - Source Image
		let (src_image, src_memory) = ImageBuilder::new(
            vk::Extent3D { width: equirect.width as u32, height: equirect.height as u32, depth: 1 },
            SRC_FORMAT,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        )
        .build(instance, device, physical_device)?;

		let src_view = create_image_view(
            device,
            src_image,
            SRC_FORMAT,
            vk::ImageAspectFlags::COLOR,
			vk::ImageViewType::_2D,
            1,
            1,
        )?;

		transition_image_layout(
			device,
			setup.command_buffer,
			src_image,
			vk::ImageLayout::UNDEFINED,
			vk::ImageLayout::TRANSFER_DST_OPTIMAL,
			1
		)?;

		copy_buffer_to_image(
			device,
			setup.command_buffer,
			staging_buffer,
			src_image,
			equirect.width as u32,
			equirect.height as u32
		)?;

		transition_image_layout(
			device,
			setup.command_buffer,
			src_image,
			vk::ImageLayout::TRANSFER_DST_OPTIMAL,
			vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
			1
		)?;

		// 3 - Descriptors
		let set_layout = create_image_to_image_descriptor_set_layout(device)?;
		let pool = EnvironmentCubemap::create_descriptor_pool(device)?;
		let set = create_image_to_image_descriptor_set(
			device,
			set_layout,
			pool,
			src_view,
			sampler,
			self.resource.vk_image_view,
		)?;

		// 4 - Pipeline
		let pipeline = Pipeline::new_compute(device, EQUIRECT_TO_CUBEMAP_COMP, &[set_layout])?;

		// 5 - Record
		let subresource_range = vk::ImageSubresourceRange::builder()
			.aspect_mask(vk::ImageAspectFlags::COLOR)
			.base_mip_level(0)
			.level_count(1)
			.base_array_layer(0)
			.layer_count(6);

		let to_general = vk::ImageMemoryBarrier::builder()
			.old_layout(vk::ImageLayout::UNDEFINED)
			.new_layout(vk::ImageLayout::GENERAL)
			.src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
			.dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
			.src_access_mask(vk::AccessFlags::empty())
			.dst_access_mask(vk::AccessFlags::SHADER_WRITE)
			.image(self.resource.vk_image)
			.subresource_range(subresource_range);

		let to_read_only = vk::ImageMemoryBarrier::builder()
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .image(self.resource.vk_image)
            .subresource_range(subresource_range);

		let groups = ENV_CUBEMAP_SIZE.div_ceil(IBL_LOCAL_SIZE);

		unsafe {
			device.cmd_pipeline_barrier(
				setup.command_buffer,
				vk::PipelineStageFlags::TOP_OF_PIPE,
				vk::PipelineStageFlags::COMPUTE_SHADER,
				vk::DependencyFlags::empty(),
				&[] as &[vk::MemoryBarrier],
				&[] as &[vk::BufferMemoryBarrier],
				&[to_general]
			);

			device.cmd_bind_pipeline(
				setup.command_buffer,
				vk::PipelineBindPoint::COMPUTE,
				pipeline.vk_pipeline
			);

			device.cmd_bind_descriptor_sets(
				setup.command_buffer,
				vk::PipelineBindPoint::COMPUTE,
				pipeline.vk_layout,
				0,
				&[set],
				&[]
			);

			device.cmd_dispatch(
				setup.command_buffer,
				groups,
				groups,
				6
			);

			device.cmd_pipeline_barrier(
				setup.command_buffer,
				vk::PipelineStageFlags::COMPUTE_SHADER,
				vk::PipelineStageFlags::FRAGMENT_SHADER,
				vk::DependencyFlags::empty(),
				&[] as &[vk::MemoryBarrier],
				&[] as &[vk::BufferMemoryBarrier],
				&[to_read_only]
			);
		}

		setup.defer(move |d| unsafe { pipeline.destroy(d) });
		setup.defer(move |d| unsafe { d.destroy_descriptor_pool(pool, None) });
        setup.defer(move |d| unsafe { d.destroy_descriptor_set_layout(set_layout, None) });
		setup.defer(move |d| unsafe {
            d.destroy_image_view(src_view, None);
            d.destroy_image(src_image, None);
            d.free_memory(src_memory, None);
        });
		setup.defer(move |d| unsafe {
			d.destroy_buffer(staging_buffer, None);
			d.free_memory(staging_memory, None);
		});

		Ok(())
	}

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
		let pool_sizes = &[
            *vk::DescriptorPoolSize::builder()
                .type_(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(1),
            *vk::DescriptorPoolSize::builder()
                .type_(vk::DescriptorType::SAMPLER)
                .descriptor_count(1),
            *vk::DescriptorPoolSize::builder()
                .type_(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1),
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(pool_sizes)
            .max_sets(1);

        Ok(unsafe { device.create_descriptor_pool(&pool_info, None)? })
	}

	#[inline]
	pub fn view(&self) -> vk::ImageView { self.resource.vk_image_view }
	
	/// ## Safety
    ///
    /// The caller must ensure the device is idle and that no descriptor set still
    /// referencing this image is bound by a pending command buffer.
	#[allow(unsafe_op_in_unsafe_fn)]
	pub unsafe fn destroy(&self, device: &Device) {
		self.resource.destroy(device);
	}
}