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
	create_image_to_image_descriptor_set, create_storage_image_descriptor_set, generate_mipmaps,
};
use crate::constants::{
	BRDF_LUT_COMP,
	EQUIRECT_TO_CUBEMAP_COMP,
	IRRADIANCE_COMP,
	PREFILTERED_COMP,
	BRDF_LUT_SIZE,
};

/// Workgroup size for the IBL compute shaders, which all walk a 2D grid of texels.
/// 
/// Must match `local_size_x` / `local_size_y` in the shader .comp
const IBL_LOCAL_SIZE: u32 = 16;
const CUBEMAP_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

/// Pre-integrated split-sum BRDF.
///
/// Red holds the scale applied to F0, green the bias. Generated once at startup
/// by a compute shader and independent of the scene and of the environment.
#[derive(Debug)]
pub struct BRDFLut(pub ImageResource);

/// A cube-shaped image: six square faces sharing one allocation, viewed as a cube.
///
/// ## Fields
///
/// - `resource` ( [ImageResource] ) - The image, its memory, and the cube view.
/// - `face_size` ( `u32` ) - Side length of one face, in texels. Drives the dispatch size when generating.
/// - `mip_levels` ( `u32` ) - Mip levels of the image cubemap
#[derive(Debug)]
pub struct Cubemap {
	pub resource: ImageResource,
	pub face_size: u32,
	pub mip_levels: u32,
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
			0,
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
}

impl Cubemap {
	pub fn new(
		instance: &Instance,
		device: &Device,
		physical_device: vk::PhysicalDevice,
		face_size: u32,
		mip_levels: u32,
	) -> Result<Self> {
		let (vk_image, vk_image_memory) = ImageBuilder::new(
				vk::Extent3D { width: face_size, height: face_size, depth: 1 },
				CUBEMAP_FORMAT,
				vk::ImageUsageFlags::STORAGE
					| vk::ImageUsageFlags::SAMPLED
					| vk::ImageUsageFlags::TRANSFER_SRC
					| vk::ImageUsageFlags::TRANSFER_DST,
			)
			.cube()
			.mip_levels(mip_levels)
			.build(instance, device, physical_device)?;

		let vk_image_view = create_image_view(
			device,
			vk_image,
			CUBEMAP_FORMAT,
			vk::ImageAspectFlags::COLOR,
			vk::ImageViewType::CUBE,
			0,
			mip_levels, 
			6
		)?;

		Ok(Self {
			resource: ImageResource { vk_image, vk_image_memory, vk_image_view },
			face_size,
			mip_levels,
		})
	}
	
	/// ## Safety
    ///
    /// The caller must ensure the device is idle and that no descriptor set still
    /// referencing this image is bound by a pending command buffer.
	#[allow(unsafe_op_in_unsafe_fn)]
	pub unsafe fn destroy(&self, device: &Device) {
		self.resource.destroy(device);
	}
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
pub fn generate_lut(
	device: &Device,
	setup: &mut SetupCommand,
	brdf_lut: &BRDFLut,
) -> Result<()> {
	// 1- descriptor
	let set_layout = create_storage_image_descriptor_set_layout(device)?;
	let pool = create_brdf_lut_descriptor_pool(device)?;
	let set = create_storage_image_descriptor_set(
		device,
		set_layout,
		pool,
		brdf_lut.0.vk_image_view
	)?;

	// 2- pipeline
	let pipeline = Pipeline::new_compute(
		device,
		BRDF_LUT_COMP,
		&[set_layout],
		&[] as &[vk::PushConstantRange],
	)?;

	// 3- record
	// The portion of the image that the barrier covers.
	let subresource_range = get_subresource_range(
		0,
		1,
		1
	);

	// Undefined -> General
	let to_general = vk::ImageMemoryBarrier::builder()
		.old_layout(vk::ImageLayout::UNDEFINED)
		.new_layout(vk::ImageLayout::GENERAL)
		.src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
		.dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
		.src_access_mask(vk::AccessFlags::empty())
		.dst_access_mask(vk::AccessFlags::SHADER_WRITE)
		.image(brdf_lut.0.vk_image)
		.subresource_range(subresource_range);
	
	// General -> Shader read only + block the read until the write is finish
	let to_read_only = vk::ImageMemoryBarrier::builder()
		.old_layout(vk::ImageLayout::GENERAL)
		.new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
		.src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
		.dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
		.src_access_mask(vk::AccessFlags::SHADER_WRITE)
		.dst_access_mask(vk::AccessFlags::SHADER_READ)
		.image(brdf_lut.0.vk_image)
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
pub fn generate_environment(
	instance: &Instance,
	device: &Device,
	physical_device: vk::PhysicalDevice,
	dst: &Cubemap,
	setup: &mut SetupCommand,
	equirect: &EquirectImage,
	sampler: vk::Sampler,
) -> Result<()> {
	let src_view = upload_equirect(instance, device, physical_device, setup, equirect)?;

	// 3 - Record
	record_cubemap_pass(
		device,
		setup,
		EQUIRECT_TO_CUBEMAP_COMP,
		src_view,
		sampler,
		dst
	)?;

	let subresource_range = |base, count| {
		*get_subresource_range(
			base,
			count,
			6
		)
	};

	let level_zero = vk::ImageMemoryBarrier::builder()
		.old_layout(vk::ImageLayout::GENERAL)
		.new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
		.src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
		.dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
		.src_access_mask(vk::AccessFlags::SHADER_WRITE)
		.dst_access_mask(vk::AccessFlags::TRANSFER_READ)
		.image(dst.resource.vk_image)
		.subresource_range(subresource_range(0, 1));

	let rest_of_chain = vk::ImageMemoryBarrier::builder()
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .image(dst.resource.vk_image)
        .subresource_range(subresource_range(1, dst.mip_levels - 1));

	unsafe {
		device.cmd_pipeline_barrier(
			setup.command_buffer,
			vk::PipelineStageFlags::COMPUTE_SHADER,
			vk::PipelineStageFlags::TRANSFER,
			vk::DependencyFlags::empty(),
			&[] as &[vk::MemoryBarrier],
			&[] as &[vk::BufferMemoryBarrier],
			&[level_zero, rest_of_chain],
		);
	}

	generate_mipmaps(
		instance,
		device,
		physical_device,
		setup.command_buffer,
		dst.resource.vk_image,
		CUBEMAP_FORMAT,
		dst.face_size,
		dst.face_size,
		dst.mip_levels,
		6
	)?;


	Ok(())
}

pub fn generate_irradiance(
	device: &Device,
	src: &Cubemap,
	dst: &Cubemap,
	setup: &mut SetupCommand,
	sampler: vk::Sampler,
) -> Result<()> {
	record_cubemap_pass(
		device,
		setup,
		IRRADIANCE_COMP,
		src.resource.vk_image_view,
		sampler,
		dst
	)?;

	let subresource_range = get_subresource_range(
		0,
		1,
		6,
	);

	let to_read_only = vk::ImageMemoryBarrier::builder()
		.old_layout(vk::ImageLayout::GENERAL)
		.new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
		.src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
		.dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
		.src_access_mask(vk::AccessFlags::SHADER_WRITE)
		.dst_access_mask(vk::AccessFlags::SHADER_READ)
		.image(dst.resource.vk_image)
		.subresource_range(subresource_range);

	unsafe {
		device.cmd_pipeline_barrier(
			setup.command_buffer,
			vk::PipelineStageFlags::COMPUTE_SHADER,
			vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::COMPUTE_SHADER,
			vk::DependencyFlags::empty(),
			&[] as &[vk::MemoryBarrier],
			&[] as &[vk::BufferMemoryBarrier],
			&[to_read_only]
		);
	}


	Ok(())
}

/// Convolves the environment cubemap into a chain of increasingly rough specular
/// reflections, one mip level per roughness step.
///
/// Mip 0 holds the mirror reflection, the last level the roughest. The fragment
/// shader picks a level from the material roughness, so the chain length must
/// reach it through `prefilteredCubMipLevels` in the UBO.
///
/// `src` must carry its own mip chain: each sample reads it at a level derived
/// from its probability density, which is what keeps a bright texel from
/// becoming an isolated firefly at high roughness.
pub fn generate_prefiltered(
	device: &Device,
	setup: &mut SetupCommand,
	src: &Cubemap,
	dst: &Cubemap,
	sampler: vk::Sampler,
) -> Result<()> {
	let push_constant_range = *vk::PushConstantRange::builder()
		.stage_flags(vk::ShaderStageFlags::COMPUTE)
		.offset(0)
		.size(size_of::<f32>() as u32);

	let set_layout = create_image_to_image_descriptor_set_layout(device)?;
	let pool = create_image_to_image_descriptor_pool(device, dst.mip_levels)?;
	let pipeline = Pipeline::new_compute(
		device,
		PREFILTERED_COMP,
		&[set_layout],
		&[push_constant_range],
	)?;

	// Undefined -> General
	let to_general = vk::ImageMemoryBarrier::builder()
		.old_layout(vk::ImageLayout::UNDEFINED)
		.new_layout(vk::ImageLayout::GENERAL)
		.src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
		.dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
		.src_access_mask(vk::AccessFlags::empty())
		.dst_access_mask(vk::AccessFlags::SHADER_WRITE)
		.image(dst.resource.vk_image)
		.subresource_range(get_subresource_range(0, dst.mip_levels, 6));

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
			pipeline.vk_pipeline,
		);
	}

	for mip in 0..dst.mip_levels {
		let mip_view = create_image_view(
			device,
			dst.resource.vk_image,
			CUBEMAP_FORMAT,
			vk::ImageAspectFlags::COLOR,
			vk::ImageViewType::CUBE,
			mip,
			1,
			6
		)?;

		let set = create_image_to_image_descriptor_set(
			device,
			set_layout,
			pool,
			src.resource.vk_image_view,
			sampler,
			mip_view
		)?;

		let roughness = mip as f32 / (dst.mip_levels - 1) as f32;
		let mip_size = dst.face_size >> mip;

		let groups = mip_size.div_ceil(IBL_LOCAL_SIZE);

		unsafe {
			device.cmd_bind_descriptor_sets(
				setup.command_buffer,
				vk::PipelineBindPoint::COMPUTE,
				pipeline.vk_layout,
				0,
				&[set],
				&[],
			);

			device.cmd_push_constants(
				setup.command_buffer,
				pipeline.vk_layout,
				vk::ShaderStageFlags::COMPUTE,
				0,
				&roughness.to_ne_bytes(),
			);

			device.cmd_dispatch(setup.command_buffer, groups, groups, 6);
		}

		setup.defer(move |d| unsafe { d.destroy_image_view(mip_view, None) });
	}

	// General -> Shader read only + block the read until the write is finish
	let to_read_only = vk::ImageMemoryBarrier::builder()
		.old_layout(vk::ImageLayout::GENERAL)
		.new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
		.src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
		.dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
		.src_access_mask(vk::AccessFlags::SHADER_WRITE)
		.dst_access_mask(vk::AccessFlags::SHADER_READ)
		.image(dst.resource.vk_image)
		.subresource_range(get_subresource_range(0, dst.mip_levels, 6));

	unsafe {
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

	setup.defer(move |d| unsafe { pipeline.destroy(d) });
	setup.defer(move |d| unsafe { d.destroy_descriptor_pool(pool, None) });
	setup.defer(move |d| unsafe { d.destroy_descriptor_set_layout(set_layout, None) });

	Ok(())
}

/// Uploads a floating-point panorama as a sampled texture.
///
/// The image is a scratch resource for the equirect-to-cubemap pass: nothing
/// reads it after the conversion, so everything it owns is deferred to the setup
/// batch. Only the view is returned, for the descriptor that reads it.
///
/// ## Arguments
///
/// - `instance` ( &[Instance] ) - The Vulkan instance.
/// - `device` ( &[Device] ) - The Vulkan device.
/// - `physical_device` ( [vk::PhysicalDevice] ) - Used to allocate the staging buffer and the image.
/// - `setup` ( &mut [SetupCommand] ) - Batch the commands are recorded into.
/// - `equirect` ( &[EquirectImage] ) - Panorama pixels, in `f32` RGBA.
///
/// ## Returns
///
/// - `Result<`[vk::ImageView]`>` - View on the uploaded panorama.
fn upload_equirect(
	instance: &Instance,
	device: &Device,
	physical_device: vk::PhysicalDevice,
	setup: &mut SetupCommand,
	equirect: &EquirectImage,
) -> Result<vk::ImageView> {
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
		0,
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

	setup.defer(move |d| unsafe {
		d.destroy_image_view(src_view, None);
		d.destroy_image(src_image, None);
		d.free_memory(src_memory, None);
	});
	setup.defer(move |d| unsafe {
		d.destroy_buffer(staging_buffer, None);
		d.free_memory(staging_memory, None);
	});

	Ok(src_view)
}

/// Records a compute pass that reads one image and writes a cubemap.
///
/// Shared by every IBL generation step: descriptors, pipeline, the layout
/// transition around the dispatch. The caller supplies the shader and defers the
/// returned resources.
fn record_cubemap_pass(
	device: &Device,
	setup: &mut SetupCommand,
	shader: &[u8],
	src_view: vk::ImageView,
	sampler: vk::Sampler,
	dst: &Cubemap,
) -> Result<()> {
	let storage_view = create_image_view(
		device,
		dst.resource.vk_image,
		CUBEMAP_FORMAT,
		vk::ImageAspectFlags::COLOR,
		vk::ImageViewType::CUBE,
		0,
		1,
		6,
	)?;

	// 1 - Descriptor
	let set_layout = create_image_to_image_descriptor_set_layout(device)?;
    let pool = create_image_to_image_descriptor_pool(device, 1)?;

	let set = create_image_to_image_descriptor_set(
		device,
		set_layout,
		pool,
		src_view,
		sampler,
		storage_view,
	)?;

	// 2 - Pipeline
	let pipeline = Pipeline::new_compute(
		device,
		shader,
		&[set_layout],
		&[] as &[vk::PushConstantRange]
	)?;

	// 3 - Record
	let subresource_range = get_subresource_range(
		0,
		1,
		6,
	);

	let to_general = vk::ImageMemoryBarrier::builder()
		.old_layout(vk::ImageLayout::UNDEFINED)
		.new_layout(vk::ImageLayout::GENERAL)
		.src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
		.dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
		.src_access_mask(vk::AccessFlags::empty())
		.dst_access_mask(vk::AccessFlags::SHADER_WRITE)
		.image(dst.resource.vk_image)
		.subresource_range(subresource_range);

	let groups = dst.face_size.div_ceil(IBL_LOCAL_SIZE);

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
	};

	setup.defer(move |d| unsafe { d.destroy_image_view(storage_view, None) });
	setup.defer(move |d| unsafe { pipeline.destroy(d) });
    setup.defer(move |d| unsafe { d.destroy_descriptor_pool(pool, None) });
    setup.defer(move |d| unsafe { d.destroy_descriptor_set_layout(set_layout, None) });

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
fn create_image_to_image_descriptor_pool(device: &Device, max_sets: u32) -> Result<vk::DescriptorPool> {
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
		.max_sets(max_sets);

	Ok(unsafe { device.create_descriptor_pool(&pool_info, None)? })
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
fn create_brdf_lut_descriptor_pool(device: &Device) -> Result<vk::DescriptorPool> {
	let pool_size = vk::DescriptorPoolSize::builder()
		.type_(vk::DescriptorType::STORAGE_IMAGE)
		.descriptor_count(1);

	let pool_sizes = &[pool_size];
	let pool_info = vk::DescriptorPoolCreateInfo::builder()
		.pool_sizes(pool_sizes)
		.max_sets(1);

	Ok(unsafe { device.create_descriptor_pool(&pool_info, None)? })
}

#[inline]
fn get_subresource_range(
	base_mip_level: u32,
	level_count: u32,
	layer_count: u32
) -> vk::ImageSubresourceRangeBuilder {
	vk::ImageSubresourceRange::builder()
		.aspect_mask(vk::ImageAspectFlags::COLOR)
		.base_mip_level(base_mip_level)
		.level_count(level_count)
		.base_array_layer(0)
		.layer_count(layer_count)
}