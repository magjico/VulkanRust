use anyhow::{Result, anyhow};

use vulkanalia::prelude::v1_0::*;

/// Find, if able, a suitable memory type (from our physical device, e.g. GPU) for all *properties* pass as an arguments.
/// 
/// ## Arguments
/// 
/// - `instance` (&[`Instance`]) - Our Vulkan instance.
/// - `physical_device` ([`vk::PhysicalDevice`]) - Our physical device.
/// - `properties` ([`vk::MemoryPropertyFlags`]) - Property flag which englobe all wanted properties.
/// - `requirements` ([`vk::MemoryRequirements`]) - Wanted memory requirement.
/// 
/// ## Returns
/// 
/// - Result<u32> - the memory type index
/// 
/// ## Errors
/// 
/// No suitable memory type.
pub fn get_memory_type_index(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    properties: vk::MemoryPropertyFlags,
    requirements: vk::MemoryRequirements,
) -> Result<u32> {
    let memory = unsafe {instance.get_physical_device_memory_properties(physical_device)};
    (0..memory.memory_type_count)
        .find(|i| {
            let suitable = (requirements.memory_type_bits & (1 << i)) != 0;
            let memory_type = memory.memory_types[*i as usize];
            suitable && memory_type.property_flags.contains(properties)
        })
        .ok_or_else(|| anyhow!("Failed to find suitable memory type."))
}