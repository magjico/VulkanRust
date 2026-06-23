//! Setup multiple **specific** app objects
use rand::RngExt;
use crate::math::*;

use log::*;

use anyhow::Result;

use vulkanalia::prelude::v1_0::*;

use crate::render::{TextureData, UniformBufferObject, create_texture_image, create_texture_image_view, create_texture_sampler};
use crate::scene::{Material, Model, ModelInstance, Skin};
use crate::constants::*;

//===============================================
// Descriptors
//===============================================

/// Generate a descriptor pool to allocate multiple descriptor sets.
/// so that a shader can access UBO and texture sampling from our shaders 
/// 
/// ## Arguments
/// 
/// - `device` ( &[Device] ) - The Vulkan device.
/// - `swapchain_images_count` (`u32`) - number of swapchain images (we will generate a descriptor set by image).
/// 
/// ## Returns
/// 
/// - `Result<vk::DescriptorPool>`.
pub fn create_descriptor_pool(
    device: &Device,
    swapchain_images_count: u32,
    materials_count: u32,
    skins_count: u32,
) -> Result<vk::DescriptorPool> {
    let ubo_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(swapchain_images_count);

    let sampler_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(materials_count * 5);

    let ssbo_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(skins_count * swapchain_images_count);

    let pool_sizes = &[ubo_size, sampler_size, ssbo_size];
    let info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(pool_sizes)
        .max_sets(swapchain_images_count + materials_count + skins_count * swapchain_images_count);

    let descriptor_pool = unsafe { device.create_descriptor_pool(&info, None)? };

    Ok(descriptor_pool)
}

/// Generate multiple descriptor set for each swapchain image.
/// With those the shaders will have access to the UBO.
/// 
/// ## Arguments
/// 
/// - `device` ( &[Device] ) - The Vulkan device.
/// - `swapchain_images_count` ( `usize` ) - number of swapchain images.
/// - `global_set_layout` ( [`vk::DescriptorSetLayout`] ) - see [create_global_set_layout].
/// - `descriptor_pool` ( [`vk::DescriptorPool`] ) - see [create_descriptor_pool].
/// - `uniform_buffers` ( &[vk::Buffer] ).
/// 
/// ## Returns
/// 
/// - `Result<Vec<vk::DescriptorSet>>`.
pub fn create_global_descriptor_sets(
    device: &Device,
    swapchain_images_count: usize,
    global_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    uniform_buffers: &[vk::Buffer],
) -> Result<Vec<vk::DescriptorSet>> {
    let layouts = vec![global_set_layout; swapchain_images_count];

    let info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&layouts);

    // to return
    let descriptor_sets = unsafe { device.allocate_descriptor_sets(&info)? };

    for i in 0..swapchain_images_count {
        let info = vk::DescriptorBufferInfo::builder()
            .buffer(uniform_buffers[i])
            .offset(0)
            .range(size_of::<UniformBufferObject>() as u64);

        let buffer_info = &[info];
        
        let ubo_write = vk::WriteDescriptorSet::builder()
            .dst_set(descriptor_sets[i])
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(buffer_info);

        unsafe { device.update_descriptor_sets(&[ubo_write], &[] as &[vk::CopyDescriptorSet]) };
    }

    Ok(descriptor_sets)
}


/// Generate multiple descriptor set for each swapchain image.
/// With those the shaders will have access to the textures.
/// 
/// ## Arguments
/// 
/// - `device` ( &[Device] ) - The Vulkan device.
/// - `material_set_layout` ([`vk::DescriptorSetLayout`]) - see [create_material_set_layout].
/// - `descriptor_pool` ([`vk::DescriptorPool`]) - see [create_descriptor_pool].
/// - materials (&\[[Material]]),
/// - textures (&\[[TextureData]]),
/// - default_texture (&[TextureData]),
/// 
/// ## Returns
/// 
/// - `Result<Vec<vk::DescriptorSet>>`.
pub fn create_material_descriptor_sets(
    device: &Device,
    material_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    materials: &[Material],
    textures: &[TextureData],
    default_texture: &TextureData,
) -> Result<Vec<vk::DescriptorSet>> {
    let layouts = vec![material_set_layout; materials.len()];

    let info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&layouts);

    // to return
    let descriptor_sets = unsafe { device.allocate_descriptor_sets(&info)? };

    // helper function to get texture or the default one
    let get_texture = |idx: i32| -> &TextureData {
        if idx >= 0 && textures.len() > idx as usize { &textures[idx as usize] } else { default_texture } 
    };

    for (i, material) in materials.iter().enumerate() {
        debug!("material indices:\n\t- base = {}\n\t- metallic = {}\n\t- normal = {}\n\t- occlusion = {}\n\t- emissive = {}",
            material.base_color_texture_idx,
            material.metallic_roughness_texture_idx,
            material.normal_texture_idx,
            material.occlusion_texture_idx,
            material.emissive_texture_idx,
        );

        let image_infos: Vec<vk::DescriptorImageInfo> = vec![
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
                .sampler(tex.sampler)
        }).collect();

        let sampler_write = vk::WriteDescriptorSet::builder()
            .dst_set(descriptor_sets[i])
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&image_infos);

        unsafe { device.update_descriptor_sets(&[sampler_write], &[] as &[vk::CopyDescriptorSet]) };
    }

    Ok(descriptor_sets)
}

/// Generate a descriptor sets for each skins (**store inside the skin not return**).
/// 
/// ## Arguments
/// 
/// - `device` ( &[Device] ) - Vulkan device.
/// - `skin_set_layout` ( [vk::DescriptorSetLayout] ) - the skin descriptor set layout.
/// - `descriptor_pool` ( [vk::DescriptorPool] ) - the skin descriptor pool.
/// - `skins` ( &mut [[Skin]] ) - All skins that need a descriptor sets.
/// - `images_count` ( `usize` ) - Number of images in flight
pub fn create_skin_descriptor_sets(
    device: &Device,
    skin_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    skins: &mut [Skin],
    images_count: usize,
) -> Result<()> {
    skins.iter_mut().try_for_each(|skin| -> Result<()> {
        let layouts = vec![skin_set_layout; images_count];
        let info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&layouts);
        let descriptor_sets = unsafe { device.allocate_descriptor_sets(&info)? };

        for i in 0..images_count {
            let buffer_info = &[
                *vk::DescriptorBufferInfo::builder()
                    .buffer(skin.ssbo_buffers[i])
                    .offset(0)
                    .range((skin.joints.len() * size_of::<Mat4>()) as vk::DeviceSize)
            ];

            let ssbo_write = vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_sets[i])
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(buffer_info);

            unsafe { device.update_descriptor_sets(&[ssbo_write], &[] as &[vk::CopyDescriptorSet]) };
        }

        skin.descriptor_sets = descriptor_sets;
        Ok(())
    })?;

    Ok(())
}   

//===============================================
// Sync objects
//===============================================

/// Create all the sync objects necessary for the render pipeline
/// 
/// ## Arguments
/// 
/// - `device` ( &[Device] ) - Vulkan device.
/// - `max_frame_in_flight` (`usize`) - max supported frame in flight for assigning semaphores and fences.
/// - `swapchain_images_count` (`usize`) - max number of swapchain image (at the same time) for fences.
/// 
/// ## Returns
/// 
/// `Result<(Vec<vk::Semaphore>, Vec<vk::Semaphore>, Vec<vk::Fence>, Vec<vk::Fence>)>`:
/// - First a `Vec<vk::Semaphore>` for **synchronising the KHR image retrieval**.
/// - Second a `Vec<vk::Semaphore>` for **synchronising when an image finish to render (so we can use it for something else)**.
/// - Third a `Vec<vk::Fence>` to **await the GPU operation on in flight frame before rendering them**.
/// - Fourth a `Vec<vk::Fence>` to **protect a swapchain image to be acquire by 2 differents frames in flight**.
pub fn create_sync_objects(
    device: &Device,
    max_frame_in_flight: usize,
    swapchain_images_count: usize,
) -> Result<(Vec<vk::Semaphore>, Vec<vk::Semaphore>, Vec<vk::Fence>, Vec<vk::Fence>)> {
    let semaphore_info = vk::SemaphoreCreateInfo::builder();
    let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

    let mut image_available_semaphores: Vec<vk::Semaphore> = Vec::new();
    let mut render_finished_semaphores: Vec<vk::Semaphore> = Vec::new();
    let mut in_flight_fences: Vec<vk::Fence> = Vec::new();
    let images_in_flight: Vec<vk::Fence>;

    unsafe {
        for _ in 0..max_frame_in_flight {
            image_available_semaphores
                .push(device.create_semaphore(&semaphore_info, None)?);
            render_finished_semaphores
                .push(device.create_semaphore(&semaphore_info, None)?);

            in_flight_fences.push(device.create_fence(&fence_info, None)?);
        }

        images_in_flight = (0..swapchain_images_count).map(|_| vk::Fence::null()).collect();
    }

    Ok((
        image_available_semaphores,
        render_finished_semaphores,
        in_flight_fences,
        images_in_flight
    ))
}

//===============================================
// models
//===============================================

/// Setup 10 instances objects for instance rendering tests
pub fn setup_object_instances() -> Result<Vec<ModelInstance>> {
    let mut rng = rand::rng();

    let n = 10;
    let x_min = -2_f32;
    let x_max =  2_f32;
    let y_min = -2_f32;
    let y_max =  2_f32;

    let nx = (n as f32).sqrt().ceil() as i32;
    let ny = (n as f32 / nx as f32).ceil() as i32;
    let step_x = if nx > 1 { (x_max - x_min) / (nx - 1) as f32 } else { 0.0 };
    let step_y = if ny > 1 { (y_max - y_min) / (ny - 1) as f32 } else { 0.0 };

    let object_instances: Vec<ModelInstance> = (0..n)
        .map(|i| {
            let ix = i % nx;
            let iy = i / nx;

            let x = x_max + ix as f32 * step_x;
            let y = y_min + iy as f32 * step_y;

            ModelInstance::from_degrees(
                Vec3::new(x, y, 0.0),
                Vec3::new(0.0,0.0, rng.random_range(-180.0..180.0)),
                Vec3::new(rng.random_range(0.5..1.2), rng.random_range(0.5..1.2), rng.random_range(0.5..1.2))
            )
        })
        .collect();

    Ok(object_instances)
}

/// load models and setup their instances for testing purposes.
pub fn load_models() -> Result<Vec<Model>> {
    let mut models = Vec::new();

    let mut model = Model::create_with_obj(MESH_PATH)?;
    model.add_instances(&setup_object_instances()?);

    models.push(model);

    Ok(models)
}

//===============================================
// defaults
//===============================================

pub fn create_default_texture(
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    setup_command_buffer: vk::CommandBuffer,
    graphics_queue: vk::Queue,
) -> Result<TextureData> {
    // White Pixel 1x1 RGBA
    let pixels = [255u8, 255, 255, 255];

    let extent = vk::Extent3D {
        width: 1,
        height: 1,
        depth: 1,
    };

    let mip_levels = 1;

    let (image, image_memory) = create_texture_image(
        instance,
        device,
        physical_device,
        setup_command_buffer,
        graphics_queue,
        extent,
        mip_levels,
        &pixels,
        vk::Format::R8G8B8A8_SRGB,
    )?;

    let image_view = create_texture_image_view(device, image, mip_levels)?;
    let sampler = create_texture_sampler(device, mip_levels as f32)?;

    Ok(TextureData {
        image,
        image_memory,
        image_view,
        sampler,
        mip_levels,
    })
}