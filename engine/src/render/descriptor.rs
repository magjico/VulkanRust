use anyhow::{Result, anyhow};

use vulkanalia::prelude::v1_0::*;

use super::{
	TextureData,
	TexturesStorage,
	UniformBufferObject
};
use crate::gpu::{
	create_global_descriptor_set_layout,
	create_material_descriptor_set_layout,
	create_storage_descriptor_set_layout,
};
use crate::scene::{
	ECSContext,
	InstanceBuffer,
	Light,
	LightBuffer,
	Material,
	ModelsStorage,
	SkinningBuffer,
	StorageBuffer,
};
use crate::type_safety::TextureId;

// docstring typing import
#[allow(unused_imports)]
use crate::type_safety::{
	ModelId,
	MaterialSetId
};

// region Descriptor Layouts
/// Layouts describing the four descriptor sets bound by the graphics pipeline.
///
/// Sets are ordered by change frequency, as recommended by Vulkan: lower-numbered
/// sets are rebound less often. They are created once and outlive swapchain
/// recreation, unlike the sets allocated from them.
/// 
/// ## Field
/// 
/// [[vk::DescriptorSetLayout]; 4]:
/// - **Set 0**. Camera and PBR uniforms (binding 0) plus the light SSBO (binding 1). Bound once per frame.
/// - **Set 1**. The five PBR textures of a material: base color, metallic-roughness, normal, occlusion and emissive + One PBR sampler. Rebound whenever the material changes.
/// - **Set 2**. Joint matrix SSBO shared by every skinned instance. Bound once per frame.
/// - **Set 3**. Per-instance world matrices and skinning offsets, indexed by `gl_InstanceIndex`. Bound once per frame.
#[derive(Debug)]
pub struct DescriptorLayouts([vk::DescriptorSetLayout; 4]);

impl DescriptorLayouts {
	pub fn new(device: &Device) -> Result<Self> {
		let global_set_layout = create_global_descriptor_set_layout(device)?;
        let material_set_layout = create_material_descriptor_set_layout(device)?;
        let skin_set_layout = create_storage_descriptor_set_layout(device)?;
        let instance_set_layout = create_storage_descriptor_set_layout(device)?;

		Ok(Self([global_set_layout, material_set_layout, skin_set_layout, instance_set_layout]))
	}

	/// `global_set_layout` (**Set 0**). Camera and PBR uniforms (binding 0) plus the light SSBO (binding 1). Bound once per frame.
    #[inline] pub fn global(&self)		-> vk::DescriptorSetLayout { self.0[0] }
	/// `material_set_layout` (**Set 1**). The five PBR textures of a material: base color, metallic-roughness, normal, occlusion and emissive. Rebound whenever the material changes.
    #[inline] pub fn material(&self)	-> vk::DescriptorSetLayout { self.0[1] }
	/// `skin_set_layout` (**Set 2**). Joint matrix SSBO shared by every skinned instance. Bound once per frame.
    #[inline] pub fn skin(&self)		-> vk::DescriptorSetLayout { self.0[2] }
	/// `instance_set_layout` (**Set 3**). Per-instance world matrices and skinning offsets, indexed by `gl_InstanceIndex`. Bound once per frame.
    #[inline] pub fn instance(&self)	-> vk::DescriptorSetLayout { self.0[3] }

	#[inline] pub fn as_slice(&self) -> &[vk::DescriptorSetLayout] { &self.0 }

    /// ## Safety
    ///
    /// The caller must ensure the device is idle, and that no pipeline layout or
    /// command buffer still in use was built from these layouts.
	#[rustfmt::skip]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn destroy(&self, device: &Device) {
        device.destroy_descriptor_set_layout(self.global(), None);
        device.destroy_descriptor_set_layout(self.material(), None);
        device.destroy_descriptor_set_layout(self.skin(), None);
        device.destroy_descriptor_set_layout(self.instance(), None);
    }
}
// endregion

// region Descriptor
/// The descriptor pool and every set allocated from it.
///
/// Sets are allocated once from [DescriptorLayouts] and reused across frames.
/// The global sets are duplicated per swapchain image because they reference the
/// uniform buffers, which are themselves duplicated; the skinning and instance
/// sets each point at a single buffer whose per-frame regions are selected by
/// offset instead.
/// 
/// ## Fields
/// 
/// - `descriptor_pool` ( [vk::DescriptorPool] ) - Pool every set below is allocated from. Destroying it frees all of them at once.
/// - `global_descriptor_sets` ( Vec<[vk::DescriptorSet]> ) - One set per swapchain image, each pointing at that image's uniform buffer and the shared light buffer.
/// - `material_descriptor_sets` ( Vec<[vk::DescriptorSet]> ) - One set per material, across all loaded models concatenated in [ModelId] order. Indexed by [MaterialSetId].
/// - `skinning_descriptor_set` ( [vk::DescriptorSet] ) - Single set pointing at the [SkinningBuffer]. Instances select their joint range through a per-instance offset.
/// - `instance_descriptor_set` ( [vk::DescriptorSet] ) - Single set pointing at the [InstanceBuffer]. Instances are addressed through `firstInstance`.
#[derive(Debug)]
pub struct Descriptors {
    pub descriptor_pool:			vk::DescriptorPool,
    pub global_descriptor_sets:		Vec<vk::DescriptorSet>,
    pub material_descriptor_sets:	Vec<vk::DescriptorSet>,
    pub skinning_descriptor_set:	vk::DescriptorSet,
	pub instance_descriptor_set:	vk::DescriptorSet,
}

impl Descriptors {
	pub fn  new(
        device:						&Device,
        ecs_context:				&ECSContext,
        descriptor_layouts:			&DescriptorLayouts,
        uniform_buffers:			&[vk::Buffer],
        textures:					&TexturesStorage,
        default_texture:			&TextureData,
        texture_sampler:            vk::Sampler,
		models:						&ModelsStorage,
        light_buffer:				&LightBuffer,
        images_count:				usize,
    ) -> Result<Self> {
		let materials: Vec<&Material> = models.iter()
			.flat_map(|model| model.materials_iter())
			.collect();
        let materials_count = materials.len();

        let skinning_buffer = ecs_context.world.get_resource::<SkinningBuffer>()
            .ok_or_else(|| anyhow!("Skinning Buffer not found in ecs_context.world"))?;

		let instance_buffer = ecs_context.world.get_resource::<InstanceBuffer>()
			.ok_or_else(|| anyhow!("Instance Buffer not found in ecs_context.world"))?;

        let descriptor_pool = create_descriptor_pool(
            device,
            images_count		as u32,
            materials_count		as u32,
        )?;
        
        let global_descriptor_sets = create_global_descriptor_sets(
            device,
            images_count,
            descriptor_layouts.global(),
            descriptor_pool,
            uniform_buffers,
            light_buffer,
        )?;

        let material_descriptor_sets = create_material_descriptor_sets(
            device,
            descriptor_layouts.material(),
            descriptor_pool,
            &materials,
            textures,
            default_texture,
            texture_sampler
        )?;

		let skinning_descriptor_set = create_storage_descriptor_set(
			device,
			descriptor_layouts.skin(),
			descriptor_pool,
			skinning_buffer.storage()
		)?;

		let instance_descriptor_set = create_storage_descriptor_set(
			device,
			descriptor_layouts.instance(),
			descriptor_pool,
			instance_buffer.storage()
		)?;

        Ok(Self {
			descriptor_pool,
			global_descriptor_sets,
			material_descriptor_sets,
			skinning_descriptor_set,
			instance_descriptor_set
		})
	}

	pub fn update_global_descriptor_sets(
		&mut self,
		device: &Device,
		uniform_buffers: &[vk::Buffer],
	) {
		for (set, buffer) in self.global_descriptor_sets.iter().zip(uniform_buffers) {
            let buffer_info = &[*vk::DescriptorBufferInfo::builder()
                .buffer(*buffer)
                .offset(0)
                .range(size_of::<UniformBufferObject>() as u64)];

            let ubo_write = vk::WriteDescriptorSet::builder()
                .dst_set(*set)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(buffer_info);

            unsafe { device.update_descriptor_sets(&[ubo_write], &[] as &[vk::CopyDescriptorSet]); }
        }
	}

    /// ## Safety
    ///
    /// The caller must ensure the device is idle, and that none of these sets is still
    /// bound by a pending command buffer.
	#[rustfmt::skip]
    #[allow(unsafe_op_in_unsafe_fn)]
	#[inline]
	pub unsafe fn destroy(&self, device: &Device) {
		device.destroy_descriptor_pool(self.descriptor_pool, None);
	}
}

/// Creates the pool every descriptor set in [Descriptors] is allocated from.
///
/// The pool is sized for the exact number of sets the renderer needs: one global
/// set per swapchain image, one set per material, plus the single skinning and
/// instance sets.
///
/// ## Arguments
///
/// - `device` ( &[Device] ) - The Vulkan device.
/// - `swapchain_images_count` ( `u32` ) - Number of swapchain images; one global descriptor set is allocated per image.
/// - `materials_count` ( `u32` ) - Total number of materials across all loaded models. Each needs five combined image and one sampler.
fn create_descriptor_pool(
    device: &Device,
    swapchain_images_count: u32,
    materials_count: u32,
) -> Result<vk::DescriptorPool> {
    let ubo_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(swapchain_images_count);

    let sampled_image_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::SAMPLED_IMAGE)
        .descriptor_count(materials_count * 5);

    let sampler_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::SAMPLER)
        .descriptor_count(materials_count);

    let ssbo_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(
            1							// Skin SSBOs size      -> a unique global set
            + 1                         // Instance SSBOs size  -> a unique global set
            + swapchain_images_count 	// Light SSBO size      -> 1 for each swapchain image
        );

    let pool_sizes = &[ubo_size, sampled_image_size, sampler_size, ssbo_size];
    let info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(pool_sizes)
        .max_sets(swapchain_images_count + materials_count + 1 + 1);

    let descriptor_pool = unsafe { device.create_descriptor_pool(&info, None)? };

    Ok(descriptor_pool)
}

/// Allocates and writes one set 0 descriptor per swapchain image.
///
/// Each set points at that image's uniform buffer (binding 0) and at the shared
/// light buffer (binding 1). One set per image is required because the uniform
/// buffers are themselves duplicated per image.
///
/// ## Arguments
///
/// - `device` ( &[Device] ) - The Vulkan device.
/// - `swapchain_images_count` ( `usize` ) - Number of swapchain images, and therefore of sets to allocate.
/// - `global_set_layout` ( [vk::DescriptorSetLayout] ) - See [create_global_descriptor_set_layout].
/// - `descriptor_pool` ( [vk::DescriptorPool] ) - See [create_descriptor_pool].
/// - `uniform_buffers` ( &\[[vk::Buffer]] ) - One uniform buffer per swapchain image, in matching order.
/// - `light_buffer` ( &[LightBuffer] ) - Shared light SSBO, referenced by every set.
///
/// ## Returns
///
/// - `Result<Vec<vk::DescriptorSet>>` - One set per swapchain image, in index order.
fn create_global_descriptor_sets(
    device: &Device,
    swapchain_images_count: usize,
    global_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    uniform_buffers: &[vk::Buffer],
    light_buffer: &LightBuffer,
) -> Result<Vec<vk::DescriptorSet>> {
    let layouts = vec![global_set_layout; swapchain_images_count];

    let info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&layouts);

    // to return
    let descriptor_sets = unsafe { device.allocate_descriptor_sets(&info)? };

    for i in 0..swapchain_images_count {
        let ubo_info = &[*vk::DescriptorBufferInfo::builder()
            .buffer(uniform_buffers[i])
            .offset(0)
            .range(size_of::<UniformBufferObject>() as u64)];
        let ubo_write = vk::WriteDescriptorSet::builder()
            .dst_set(descriptor_sets[i])
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(ubo_info);

        let light_info = &[*vk::DescriptorBufferInfo::builder()
            .buffer(light_buffer.buffer)
            .offset(0)
            .range((light_buffer.max_lights * size_of::<Light>()) as u64)];
        let light_write = vk::WriteDescriptorSet::builder()
            .dst_set(descriptor_sets[i])
            .dst_binding(1)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(light_info);

        unsafe { device.update_descriptor_sets(&[ubo_write, light_write], &[] as &[vk::CopyDescriptorSet]) };
    }

    Ok(descriptor_sets)
}

/// Allocates and writes one set 1 descriptor per material.
///
/// Each set binds the material's five PBR textures. Missing textures fall back to
/// `default_texture`, so every set always has all five bindings written.
///
/// Materials must be passed in [ModelId] order, since the resulting index is a
/// [MaterialSetId] — a global index across all loaded models.
///
/// ## Arguments
///
/// - `device` ( &[Device] ) - The Vulkan device.
/// - `material_set_layout` ( [vk::DescriptorSetLayout] ) - See [create_material_descriptor_set_layout].
/// - `descriptor_pool` ( [vk::DescriptorPool] ) - See [create_descriptor_pool].
/// - `materials` ( &\[&[Material]] ) - Every material of every loaded model, concatenated in model order.
/// - `textures` ( &[TexturesStorage] ) - Shared texture storage the material texture ids index into.
/// - `default_texture` ( &[TextureData] ) - Fallback bound wherever a material declares no texture.
/// - `texture_sampler` ( &[vk::Sampler] ) - A unique shared texture sampler
///
/// ## Returns
///
/// - `Result<Vec<vk::DescriptorSet>>` - One set per material, indexable by [MaterialSetId].
fn create_material_descriptor_sets(
    device: &Device,
    material_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    materials: &[&Material],
    textures: &TexturesStorage,
    default_texture: &TextureData,
    texture_sampler: vk::Sampler,
) -> Result<Vec<vk::DescriptorSet>> {
    let layouts = vec![material_set_layout; materials.len()];

    let info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&layouts);

    // to return
    let descriptor_sets = unsafe { device.allocate_descriptor_sets(&info)? };

    // helper function to get texture or the default one
    let get_texture = |id: Option<TextureId>| -> &TextureData {
        if let Some(id) = id {
            textures.get_texture(id)
        } else {
            default_texture
        }
    };

    for (i, material) in materials.iter().enumerate() {
        let image_infos: Vec<vk::DescriptorImageInfo> = [
            material.base_color_texture_idx,
            material.metallic_roughness_texture_idx,
            material.normal_texture_idx,
            material.occlusion_texture_idx,
            material.emissive_texture_idx,
        ].iter().map(|&idx| {
            let tex = get_texture(idx);
            *vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(tex.image_view)
        }).collect();

        let sampler_info = &[*vk::DescriptorImageInfo::builder().sampler(texture_sampler)];

        let images_write = vk::WriteDescriptorSet::builder()
            .dst_set(descriptor_sets[i])
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .image_info(&image_infos);

        let sampler_write = vk::WriteDescriptorSet::builder()
            .dst_set(descriptor_sets[i])
            .dst_binding(1)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .image_info(sampler_info);

        unsafe { device.update_descriptor_sets(&[images_write, sampler_write], &[] as &[vk::CopyDescriptorSet]) };
    }

    Ok(descriptor_sets)
}

/// Allocates and writes a DescriptorSet used to for storage.
///
/// ## Arguments
///
/// - `device` ( &[Device] ) - The Vulkan device.
/// - `descriptor_layout` ( [vk::DescriptorSetLayout] ) - See [create_skinning_descriptor_set_layout].
/// - `descriptor_pool` ( [vk::DescriptorPool] ) - See [create_descriptor_pool].
/// - `storage_buffer` ( &[StorageBuffer] ) - See [StorageBuffer].
fn create_storage_descriptor_set(
	device:				&Device,
    descriptor_layout:	vk::DescriptorSetLayout,
    descriptor_pool:	vk::DescriptorPool,
    storage_buffer:		&StorageBuffer,
) -> Result<vk::DescriptorSet> {
	let layouts = &[descriptor_layout];
    let info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(descriptor_pool)
        .set_layouts(layouts);

    let descriptor_set = unsafe { device.allocate_descriptor_sets(&info)?[0] };

    let buffer_info = &[
        *vk::DescriptorBufferInfo::builder()
            .buffer(storage_buffer.vk_buffer)
            .offset(0)
            .range(storage_buffer.get_range())
    ];

    let ssbo_write = vk::WriteDescriptorSet::builder()
        .dst_set(descriptor_set)
        .dst_binding(0)
        .dst_array_element(0)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(buffer_info);

    unsafe { device.update_descriptor_sets(&[ssbo_write], &[] as &[vk::CopyDescriptorSet]) };

    Ok(descriptor_set)
}
// endregion