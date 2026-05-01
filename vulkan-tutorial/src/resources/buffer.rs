use anyhow::Result;

use std::ptr::copy_nonoverlapping as memcpy;

use vulkanalia::prelude::v1_0::*;

use crate::gpu::get_memory_type_index;
use crate::render::UniformBufferObject;
use crate::geometry::Vertex;

use super::begin_setup_command_buffer;
use super::flush_setup_command_buffer;

//================================================
// Buffers (general)
//================================================

/// Create and return a buffer and its memory object.
/// 
/// ## Arguments
/// 
/// - `instance` (&[`Instance`]) - Vulkan instance.
/// - `device` (&[`Device`]) - Vulkan device.
/// - `physical_device` ([`vk::PhysicalDevice`]) - The physical device.
/// - `size` ([`vk::DeviceSize`]) - Buffer size to allocate.
/// - `usage` ([`vk::BufferUsageFlags`]) - Buffer usage for optimization.
/// - `properties` ([`vk::MemoryPropertyFlags`]) - Buffer properties.
pub fn create_buffer(
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    // Buffer initialization
    let buffer_info = vk::BufferCreateInfo::builder()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = unsafe { device.create_buffer(&buffer_info, None)? };

    // Memory allocation
    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

    let memory_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(requirements.size)
        .memory_type_index(get_memory_type_index(instance, physical_device, properties, requirements)?);

    let buffer_memory = unsafe { device.allocate_memory(&memory_info, None)? };

    unsafe { device.bind_buffer_memory(buffer, buffer_memory, 0)? };

    Ok((buffer, buffer_memory))
}

/// Copy element from one source buffer to a destination buffer
/// 
/// ## Arguments
/// 
/// - `device` (&[`Device`]) - Vulkan device.
/// - `source` ([`vk::Buffer`]) - src buffer to copy.
/// - `destination` ([`vk::Buffer`]) - destination buffer to receive the copy.
/// - `setup_command_buffer` ([`vk::CommandBuffer`]) - command buffer to manage command.
/// - `sizes` (`&[vk::DeviceSize]`) - byte size of each region to copy.
/// - `dst_offsets` (`&[vk::DeviceSize]`) - where in the destination buffer should each region be copied.
/// - `graphics_queue` ([`vk::Queue`]) - the graphic queue to execute (gpu) command needed for the use of the setup command buffer.
pub fn copy_buffers(
    device: &Device,
    source: vk::Buffer,
    destination: vk::Buffer,
    setup_command_buffer: vk::CommandBuffer,
    sizes: &[vk::DeviceSize],
    dst_offsets: &[vk::DeviceSize],
    graphics_queue: vk::Queue,
) -> Result<()> {
    begin_setup_command_buffer(&device, setup_command_buffer)?;

    // Commands
    let mut src_offset = 0;
    let mut regions = Vec::<vk::BufferCopy>::with_capacity(sizes.len());

    for (i, &size) in sizes.iter().enumerate() {
        regions.push(
            vk::BufferCopy::builder()
                .src_offset(src_offset)
                .dst_offset(dst_offsets[i])
                .size(size)
                .build()
        );

        src_offset += size;
    }
    unsafe { device.cmd_copy_buffer(setup_command_buffer, source, destination, &regions) };

    flush_setup_command_buffer(&device, setup_command_buffer, graphics_queue)?;

    Ok(())
}

/// Destroy a list of buffers and free their memories
/// 
/// ## Arguments
/// 
/// - `device` (`&Device`) - Vulkan Device.
/// - `buffers` (`&[vk`) - the buffers to destroys.
/// - `buffers_memory` (`&[vk`) - the memories to free.
pub fn destroy_buffers(
    device: &Device,
    buffers: &[vk::Buffer],
    buffers_memory: &[vk::DeviceMemory]
) {
    unsafe {
        buffers_memory.iter().for_each(|mem| device.free_memory(*mem, None));
        buffers.iter().for_each(|buf| device.destroy_buffer(*buf, None));
    };
}

//===============================================
// FrameBuffer
//===============================================

/// Generate a framebuffer for the msaa sampling inside the swapchain.
/// useless in dynamic rendering.
pub fn create_framebuffers(
    device: &Device,
    render_pass: vk::RenderPass,
    swapchain_image_views: &[vk::ImageView],
    color_view: vk::ImageView,
    depth_view: vk::ImageView,
    width: u32,
    height: u32,
) -> Result<Vec<vk::Framebuffer>> {
    let framebuffers = swapchain_image_views
        .iter()
        .map(|i| {
            let attachments = &[color_view, depth_view, *i];
            let create_info = vk::FramebufferCreateInfo::builder()
                .render_pass(render_pass)
                .attachments(attachments)
                .width(width)
                .height(height)
                .layers(1);

            unsafe { device.create_framebuffer(&create_info, None) }
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(framebuffers)
}

//===============================================
// Interleaved Buffer (Vertex and Index Buffers)
//===============================================

/// Create an interleaved-buffer: a buffer that contain both the information of the index buffer and the vertex buffer
/// First the vertex buffer then the index one.
/// 
/// ## Arguments
/// 
/// - `instance` ( &[Instance] ) - Vulkan instance.
/// - `device` ( &[Device] ) - Vulkan device.
/// - `physical_device` ( [vk::PhysicalDevice] ) - a physical device.
/// - `vertices` ( &[[Vertex]] ) - array of vertices.
/// - `indices` ( &\[u32] ) - array of vertex indices.
/// - `setup_command_buffer` ( [vk::CommandBuffer] ) - setup command buffer use for intern operation.
/// - `graphics_queue` ( [vk::Queue] ) - necessary to work with the setup command buffer.
/// 
/// ## Returns
/// 
/// - `Result<(vk::Buffer, vk::DeviceMemory, u64)>` - The interleaved_buffer, the device memory associated,
/// and finaly an index that say where the index part start in the buffer.
/// ```
pub fn create_interleaved_buffer(
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    vertices: &[Vertex],
    indices: &[u32],
    setup_command_buffer: vk::CommandBuffer,
    graphics_queue: vk::Queue,
) -> Result<(vk::Buffer, vk::DeviceMemory, u64)> {
    let vertex_size = (size_of::<Vertex>() * vertices.len()) as u64;
    let index_size = (size_of::<u32>() * indices.len()) as u64;
    let size = vertex_size + index_size;

    let (staging_buffer, staging_buffer_memory) = create_buffer(
        instance,
        device,
        physical_device,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;

    // Copy (staging)
    let memory = unsafe { device.map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty())? as *mut u8 };
    unsafe {
        memcpy(vertices.as_ptr() as *const u8, memory, vertex_size as usize);
        memcpy(indices.as_ptr() as *const u8, memory.add(vertex_size as usize), index_size as usize);
        device.unmap_memory(staging_buffer_memory);
    }

    // bytes size alignment
    let alignment = size_of::<u32>() as u64;
    let aligned_vertex_size = (vertex_size + alignment - 1) & !(alignment - 1);
    let size = aligned_vertex_size + index_size;

    // Create (interleaved buffer)
    let (interleaved_buffer, interleaved_buffer_memory) = create_buffer(
        instance,
        device,
        physical_device,
        size,
        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL
    )?;

    // Copy
    copy_buffers(
        device,
        staging_buffer,
        interleaved_buffer,
        setup_command_buffer,
        &[vertex_size, index_size],
        &[0, aligned_vertex_size],
        graphics_queue,
    )?;

    // Cleanup
    destroy_buffers(&device, &[staging_buffer], &[staging_buffer_memory]);

    Ok((interleaved_buffer, interleaved_buffer_memory, aligned_vertex_size))
}

//===============================================
// Uniform Buffers
//===============================================

/// Create the selected number of uniform buffers, and their associated memory.
/// Then return them. 
/// 
/// ## Arguments
/// 
/// - `instance` ( &[Instance] ) - Vulkan instance.
/// - `device` ( &[Device] ) - Vulkan device.
/// - `physical_device` ( [vk::PhysicalDevice] ) - a physical device.
/// - `count` ( usize ) - number of buffers to create.
/// 
/// ## Returns
/// 
/// - `Result<(Vec<vk::Buffer>, Vec<vk::DeviceMemory>)>` - the buffers and their memories.
pub fn create_uniform_buffers(
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    count: usize,
) -> Result<(Vec<vk::Buffer>, Vec<vk::DeviceMemory>)> {
    let mut uniform_buffers = Vec::new();
    let mut uniform_buffers_memory = Vec::new();

    for _ in 0..count {
        let (uniform_buffer, uniform_buffer_memory) = create_buffer(
            instance,
            device,
            physical_device,
            size_of::<UniformBufferObject>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        )?;

        uniform_buffers.push(uniform_buffer);
        uniform_buffers_memory.push(uniform_buffer_memory);
    }
    

    Ok((uniform_buffers, uniform_buffers_memory))

}

/// Re-create a list of uniform buffers and their memories
/// 
/// # Arguments
/// 
/// - `instance` (`&Instance`) - Vulkan instance.
/// - `device` (`&Device`) - Vulkan device.
/// - `physical_device` (`vk`) - a physical device.
/// - `uniform_buffers` (`&mut Vec<vk`) - the uniform buffers to recreates.
/// - `uniform_buffers_memory` (`&mut Vec<vk`) - the uniform buffers memories associated.
/// - `count` (`usize`) - the number of buffers to recreate.
pub fn recreate_uniform_buffers(
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    uniform_buffers: &mut Vec<vk::Buffer>,
    uniform_buffers_memory: &mut Vec<vk::DeviceMemory>,
    count: usize,
) -> Result<()> {
    destroy_buffers(&device, &uniform_buffers, &uniform_buffers_memory);

    // TODO: Maybe this should recreate the exact same number of buffer
    // so we can get rid of the count args for .len() 
    (*uniform_buffers, *uniform_buffers_memory) = create_uniform_buffers(
        &instance,
        &device,
        physical_device,
        count
    )?;

    Ok(())
}