use anyhow::{Result, anyhow};

use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::KhrSurfaceExtensionInstanceCommands;

//=======================================================
// Queue Family Indices (related to queue family memory)
//=======================================================

/// Allow you to manipulate graphical queues.
/// 
/// ## Fields
/// 
/// - `graphics` (`u32`) - Address of a graphical queue for rendering commands (support [vk::QueueFlags::GRAPHICS]).
/// - `present` (`u32`) - Address of a graphical queue for displaying in the surface swapchain (support `KHR`).
#[derive(Copy, Clone, Debug)]
pub struct QueueFamilyIndices {
    pub graphics: u32,
    pub present: u32,
}

impl QueueFamilyIndices {
    /// Generate a QueueFamilyIndices by finding the first available graphics and present queues.
    /// 
    /// ## Arguments
    /// 
    /// - `instance` ( &[Instance] ) - The Vulkan instance.
    /// - `physical_device` ( [vk::PhysicalDevice] ) - The physical device to search queue from.
    /// - `surface` ( [vk::SurfaceKHR] ) - The surface KHR for the present queue to be compatible with.
    /// 
    /// ## Returns
    /// 
    /// - `Result<Self>` - Describe the return value.
    /// 
    /// ## Errors
    /// 
    /// Missing a compatible graphic and/or surface queue.
    pub fn get(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR
    ) -> Result<Self> {
        let properties = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

        let graphics = properties
            .iter()
            .position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .map(|i| i as u32);

        let mut present = None;
        unsafe {
            for (index, _) in properties.iter().enumerate() {
                if instance.get_physical_device_surface_support_khr(physical_device, index as u32, surface)? {
                    present = Some(index as u32);
                    break;
                }
            }
        }

        if let (Some(graphics), Some(present)) = (graphics, present) {
            Ok(Self { graphics, present })
        } else {
            Err(anyhow!(SuitabilityError("Missing required queue families.")))
        }
    }
}

//=======================================================
// Compatibility
//=======================================================

use thiserror::Error;

#[derive(Debug, Error)]
#[error("{0}")]
pub struct SuitabilityError(pub &'static str);