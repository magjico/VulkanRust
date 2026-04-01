use anyhow::{Result, anyhow};

use vulkanalia::{prelude::v1_0::{Instance, vk}, vk::InstanceV1_0};

/// From a list of `candidates` search for the first [`vk::Format`] in it for a [`vk::Image`].
/// This format **must** respect the choosen `tiling` and must be compatible with our `physical_device` features.
/// 
/// ## Arguments
/// 
/// - `instance` (&[`Instance`]) - A Vulkan Instance.
/// - `physical_device` ([`vk::PhysicalDevice`]) - Our physical device.
/// - `candidates` (&[[`vk::Format`]]) - All the possible [`vk::Format`] to search from, classified if possible from the most wanted (index 0) to the less wanted (max index).
/// - `tiling` ([`vk::ImageTiling`]) - The wanted image tiling with this image format.
/// - `features` ([`vk::ImageTiling`]) - The needed feature to implement with this image format.
/// 
/// ## Returns
/// 
/// - `Result<vk::Format>` - The first matching format from the `candidates` if possible.
/// 
/// ## Errors
/// 
/// No image format matched.
/// 
/// ## Examples
/// 
/// ```
/// use graphic_env::resources::get_supported_format;
/// 
/// let candidates = &[
///     vk::Format::D32_SFLOAT,
///     vk::Format::D32_SFLOAT_S8_UINT,
///     vk::Format::D24_UNORM_S8_UINT,
/// ];
/// 
/// let format = get_supported_format(
///     my_vulkan_instance,
///     my_physical_device,
///     candidates,
///     vk::ImageTiling::OPTIMAL,
///     vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT
/// );
/// ```
pub fn get_supported_format(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    candidates: &[vk::Format],
    tiling: vk::ImageTiling,
    features: vk::FormatFeatureFlags,
) -> Result<vk::Format> {
    candidates
        .iter()
        .cloned()
        .find(|f | {
            let properties = unsafe { instance.get_physical_device_format_properties(physical_device, *f) };
            match tiling {
                vk::ImageTiling::LINEAR => properties.linear_tiling_features.contains(features),
                vk::ImageTiling::OPTIMAL => properties.optimal_tiling_features.contains(features),
                _ => false,
            }
        })
        .ok_or_else(|| anyhow!("Failed to find a supported format!"))
}
