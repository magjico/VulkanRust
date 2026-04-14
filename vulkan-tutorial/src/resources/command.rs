use anyhow::{Result, anyhow};

use vulkanalia::prelude::v1_0::*;

use crate::gpu::SuitabilityError;

//================================================
// Command Pool
//================================================


//===============================================
// Command Buffers
//===============================================

/// Generate a vector of command buffer for each swapchain images with the command pool associated pass as an argument,
/// also generate an **empty** vector of secondary command buffers for each command buffer.
/// 
/// ## Arguments
/// 
/// - `device` ( &[Device] ) - The Vulkan device.
/// - `swapchain_image_count` ( usize ) - The number of swapchain image.
/// - `command_pools` ( Vec<[vk::CommandPool]> ) - Command pools to create the command buffer from.
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