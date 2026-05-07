/// Setup multiple **specific** app objects
use rand::RngExt;
use crate::math::*;

use anyhow::Result;

use vulkanalia::prelude::v1_0::*;

use crate::render::UniformBufferObject;
use crate::scene::{Model, ModelInstance};
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
    swapchain_images_count: u32
) -> Result<vk::DescriptorPool> {
    let ubo_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(swapchain_images_count);

    let sampler_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(swapchain_images_count);

    let pool_sizes = &[ubo_size, sampler_size];
    let info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(pool_sizes)
        .max_sets(swapchain_images_count);

    let descriptor_pool = unsafe { device.create_descriptor_pool(&info, None)? };

    Ok(descriptor_pool)
}

/// Generate multiple descriptor set for each swapchain image.
/// With those the shaders will have access to the UBO and texture sampling.
/// 
/// note: you can only use this function after generating the descriptor pool for it.
/// 
/// ## Arguments
/// 
/// - `device` ( &[Device] ) - The Vulkan device.
/// - `swapchain_images_count` (`usize`) - number of swapchain images.
/// - `descriptor_set_layout` ([`vk::DescriptorSetLayout`]) - see [create_descriptor_set_layout].
/// - `descriptor_pool` ([`vk::DescriptorPool`]) - see [create_descriptor_pool].
/// - `uniform_buffers` (`&[vk::Buffer]`).
/// - `texture_image_view` ( [vk::ImageView] ).
/// - `texture_sampler` ( [vk::Sampler] ) - texture sampler.
/// 
/// ## Returns
/// 
/// - `Result<Vec<vk::DescriptorSet>>`.
pub fn create_descriptor_sets(
    device: &Device,
    swapchain_images_count: usize,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    uniform_buffers: &[vk::Buffer],
    texture_image_view: vk::ImageView,
    texture_sampler: vk::Sampler,
) -> Result<Vec<vk::DescriptorSet>> {
    let layouts = vec![descriptor_set_layout; swapchain_images_count];

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

        let info = vk::DescriptorImageInfo::builder()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(texture_image_view)
            .sampler(texture_sampler);

        let image_info = &[info];
        let sampler_write = vk::WriteDescriptorSet::builder()
            .dst_set(descriptor_sets[i])
            .dst_binding(1)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(image_info);

        unsafe { device.update_descriptor_sets(&[ubo_write, sampler_write], &[] as &[vk::CopyDescriptorSet]) };
    }

    Ok(descriptor_sets)
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

/// load a lot of models and setup their instances for testing purposes.
pub fn load_models() -> Result<Vec<Model>> {
    let mut models = Vec::new();

    let mut model = Model::create_with_obj(MESH_PATH)?;
    model.add_instances(&setup_object_instances()?);

    models.push(model);

    Ok(models)
}