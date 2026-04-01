use anyhow::Result;

use vulkanalia::prelude::v1_0::*;

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