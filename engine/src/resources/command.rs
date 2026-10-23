use anyhow::Result;

use vulkanalia::prelude::v1_0::*;

use crate::gpu::QueueFamilyIndices;

//================================================
// Command Pool
//================================================

/// Create a command pool with the reset flags using a graphics queue.
/// TODO: make the user decide which command pool flags he want to use.
/// 
/// ## Arguments
/// 
/// - `device` ( &[Device] ) - Describe this parameter.
/// - `queue_family_indices` ( &mut [QueueFamilyIndices] ).
pub fn create_command_pool(
    device: &Device,
    queue_family_indices: &QueueFamilyIndices,
) -> Result<vk::CommandPool> {
    let info = vk::CommandPoolCreateInfo::builder()
        .queue_family_index(queue_family_indices.get2(vk::QueueFlags::GRAPHICS)?)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

    Ok(unsafe { device.create_command_pool(&info, None)? }) 
}

/// Use [create_command_pool] to generate `count` command pool in a Vec.
pub fn create_command_pools(
    device: &Device,
    queue_family_indices: &QueueFamilyIndices,
    count: usize,
) -> Result<Vec<vk::CommandPool>> {
    let command_pools = (0..count)
        .map(|_| create_command_pool(device, queue_family_indices))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(command_pools)
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
) -> Result<(Vec<vk::CommandBuffer>, Vec<vk::CommandBuffer>)> {
    let mut command_buffers = Vec::new();
    let mut secondary_command_buffers = Vec::new();

    // command pool association
    for pool in command_pools {
        let primary_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(*pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = unsafe { device.allocate_command_buffers(&primary_info)?[0] };
        command_buffers.push(command_buffer);

        let secondary_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(*pool)
            .level(vk::CommandBufferLevel::SECONDARY)
            .command_buffer_count(1);
        let secondary_command_buffer = unsafe { device.allocate_command_buffers(&secondary_info)?[0] };
        secondary_command_buffers.push(secondary_command_buffer);
    }

    Ok((command_buffers, secondary_command_buffers))
}

//================================================
// "setup" command buffer
// (not to setup command buffer, but to create a special command buffer used for copies and transitions at loading time.)
//================================================

/// Owns the command buffer used for one-off load-time work and the resources
/// that must outlive its submission.
///
/// The whole context is consumed by [flush]: once loading is done, neither the pool
/// nor the buffer serve any purpose.
pub struct SetupCommand {
    pub command_buffer: vk::CommandBuffer,
    command_pool: vk::CommandPool,
    delete_queue: Vec<Box<dyn FnOnce(&Device)>>,
}

impl SetupCommand {
    pub fn new(
        device: &Device,
        queue_family_index: u32,
    ) -> Result<Self> {
        let command_pool_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { device.create_command_pool(&command_pool_info, None)? };

        let info = vk::CommandBufferAllocateInfo::builder()
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(command_pool)
            .command_buffer_count(1);
        let command_buffer = unsafe { device.allocate_command_buffers(&info)?[0] };

        let begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { device.begin_command_buffer(command_buffer, &begin_info)? };

        Ok( Self {
            command_buffer,
            command_pool,
            delete_queue: Vec::new()
        })
    }

    /// Queues a destruction to run once the setup work has completed.
    #[inline]
    pub fn defer(&mut self, f: impl FnOnce(&Device) + 'static) {
        self.delete_queue.push(Box::new(f));
    }

    /// Submits the recorded work, waits for it, then destroys every deferred
    /// resource along with the pool and buffer.
    ///
    /// Deletions run in reverse order of registration, so a resource can safely
    /// depend on one queued before it.
    ///
    /// ## Safety
    ///
    /// No other work may still reference the resources recorded into this buffer.
    pub fn flush(
        mut self,
        device: &Device,
        queue: vk::Queue,
    ) -> Result<()> {
        unsafe { device.end_command_buffer(self.command_buffer)? };

        let buffers = &[self.command_buffer];
        let info = vk::SubmitInfo::builder()
		    .command_buffers(buffers);

        unsafe {
            device.queue_submit(queue, &[info], vk::Fence::null())?;
            device.queue_wait_idle(queue)?;

            for f in self.delete_queue.drain(..).rev() {
                f(device);
            }

            device.destroy_command_pool(self.command_pool, None);
        }

        Ok(())
    }
}