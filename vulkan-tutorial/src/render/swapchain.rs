use anyhow::Result;

use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::KhrSurfaceExtensionInstanceCommands;

#[derive(Clone, Debug)]
pub struct SwapchainSupport {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupport {
    pub fn get(instance: &Instance, surface: vk::SurfaceKHR, physical_device: vk::PhysicalDevice) -> Result<Self> {
        Ok(Self {
            capabilities: unsafe { instance.get_physical_device_surface_capabilities_khr(physical_device, surface)? },
            formats: unsafe { instance.get_physical_device_surface_formats_khr(physical_device, surface)? },
            present_modes: unsafe { instance.get_physical_device_surface_present_modes_khr(physical_device, surface)? },
        })
    }
}