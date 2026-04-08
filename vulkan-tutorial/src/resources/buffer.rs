use anyhow::Result;

use vulkanalia::prelude::v1_0::*;

use crate::gpu::get_memory_type_index;

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

//================================================
// setup command buffer
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