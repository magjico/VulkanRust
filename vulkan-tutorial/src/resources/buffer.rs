use anyhow::Result;

use vulkanalia::prelude::v1_0::*;

use crate::gpu::get_memory_type_index;
use crate::render::UniformBufferObject;

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
/// # Returns
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

//===============================================
// Command Buffers
//===============================================

/// Generate a vector of command buffer for each swapchain images with the command pool associated pass as an argument,
/// also generate an **empty** vector of secondary command buffers for each command buffer.
/// 
/// ## Arguments
/// 
/// - `device` ( &[Device] ) - The Vulkan device.
/// - `swapchain_image_count` (`usize`) - The number of swapchain image.
/// - `command_pools` (`Vec<vk`) - Command pools to create the command buffer from.
/// 
/// ## Returns
/// 
/// - `Result<(Vec<vk::CommandBuffer>, Vec<Vec<vk::CommandBuffer>>)>` - A vector of command buffer and a vector of vector of secondary command buffer.
/// ```
pub fn create_command_buffers(
    device: &Device,
    command_pools: &[vk::CommandPool],
) -> Result<(Vec<vk::CommandBuffer>, Vec<Vec<vk::CommandBuffer>>)> {
    let mut command_buffers = Vec::new();

    // command pool association
    for pool in command_pools {
        let allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(*pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffer = unsafe { device.allocate_command_buffers(&allocate_info)?[0] };
        command_buffers.push(command_buffer);
    }

    let secondary_command_buffers: Vec<Vec<vk::CommandBuffer>> = vec![vec![]; command_buffers.len()];

    Ok((command_buffers, secondary_command_buffers))
}

//================================================
// "setup" command buffer
// (not to setup command buffer, but to create a special command buffer)
//================================================

/// Create a **setup command buffer** and return it.
/// A **setup command buffer** is a command buffer use to submit commands from helper function
/// asynchronously and without barriers (e.g. *create_texture_image()*)
/// 
/// ## Arguments
/// 
/// - `device` (&[`Device`]) - The vulkan application device.
/// - `command_pool` ([`vk::CommandPool`]) - The commands pool to record commands inside the setup_command_buffer.
/// 
/// ## Returns
/// 
/// - Result<[`vk::CommandBuffer`]> - The command buffer to record commands of helpers function into.
/// 
/// ## Errors
/// 
/// Allocution error.
pub fn create_setup_command_buffer(device: &Device, command_pool: vk::CommandPool) -> Result<vk::CommandBuffer> {
    let info = vk::CommandBufferAllocateInfo::builder()
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(command_pool)
        .command_buffer_count(1);
    
    let command_buffer = unsafe { device.allocate_command_buffers(&info)?[0] };

    Ok(command_buffer)
}

/// Put a setup_command_buffer (see [`create_setup_command_buffer`]) into the *begin state*.
/// So we can record commands into it, and them flush them using [`flush_setup_command_buffer`].
/// 
/// ## Arguments
/// 
/// - `device` (&[`Device`]) - The vulkan application device.
/// - `setup_command_buffer` ([`vk::CommandBuffer`]).
pub fn begin_setup_command_buffer(device: &Device, setup_command_buffer: vk::CommandBuffer) -> Result<()> {
    let info = vk::CommandBufferBeginInfo::builder()
		.flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

	unsafe {device.begin_command_buffer(setup_command_buffer, &info)?};

    Ok(())
}

/// Flush all commands from a setup_command_buffer (see [`create_setup_command_buffer`]).
/// To records commands a setup_command_buffer must be in a begin state (see [`begin_setup_command_buffer`]).
/// 
/// ## Arguments
/// 
/// - `device` (&[`Device`]) - The vulkan application device.
/// - `setup_command_buffer` ([`vk::CommandBuffer`]).
/// - `graphics_queue` ([`vk::Queue`]) - The graphic queue to which we submit the commands.
pub fn flush_setup_command_buffer(
    device: &Device,
    setup_command_buffer: vk::CommandBuffer,
    graphics_queue: vk::Queue,
) -> Result<()> {
    unsafe {device.end_command_buffer(setup_command_buffer)?};

    let command_buffers = &[setup_command_buffer];
	let info = vk::SubmitInfo::builder()
		.command_buffers(command_buffers);

    unsafe {
        device.queue_submit(graphics_queue, &[info], vk::Fence::null())?;
        device.queue_wait_idle(graphics_queue)?;

        device.reset_command_buffer(setup_command_buffer, vk::CommandBufferResetFlags::empty())?;
    }

    Ok(())
}