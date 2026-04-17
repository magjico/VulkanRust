use std::collections::{HashMap, HashSet};

use log::*;

use anyhow::{Result, anyhow};

use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::KhrSurfaceExtensionInstanceCommands;

use crate::constants::PORTABILITY_MACOS_VERSION;

//=======================================================
// Physical Devices
//=======================================================

/// Check if the physical device support all the mandatory requirement.
/// 
/// ## Arguments
/// 
/// - `features` ( &[vk::PhysicalDeviceFeatures] ) - The physical device features to check.
/// - `extensions` ( &HashSet<[vk::StringArray<256>]> ) - The physical device extensions to check.
/// - `mandatory_features` ( &[vk::PhysicalDeviceFeatures] ) - The physical device features to have.
/// - `mandatory_device_extensions` ( &[vk::ExtensionName] ) - The mandatory extensions to support.
/// 
/// ## Returns
/// 
/// - `bool` - **true** if supported else **false**.
fn is_physical_device_supported(
    features: &vk::PhysicalDeviceFeatures,
    extensions: &HashSet<vk::StringArray<256>>,
    mandatory_features: &vk::PhysicalDeviceFeatures,
    mandatory_device_extensions: &[vk::ExtensionName],
) -> bool {
    
    let feat_mem_len  = std::mem::size_of::<vk::PhysicalDeviceFeatures>()
        / std::mem::size_of::<vk::Bool32>();

    let available_feat_slice: &[vk::Bool32] = unsafe {
        std::slice::from_raw_parts(
            features as *const _ as *const vk::Bool32,
            feat_mem_len
        )
    };

    let required_feat_slice: &[vk::Bool32] = unsafe {
        std::slice::from_raw_parts(
            mandatory_features as *const _ as *const vk::Bool32,
            feat_mem_len
        )
    };

    return required_feat_slice.iter().zip(available_feat_slice.iter())
        .all(|(&req, &avail)| req == vk::FALSE || avail == vk::TRUE)
        && mandatory_device_extensions.iter().all(|ext| extensions.contains(ext));
}

/// Score a physical device (useful to compare physical devices between them)
/// 
/// ## Arguments
/// 
/// - `features` ( &[vk::PhysicalDeviceFeatures] ) - The physical device feature.
/// - `properties` ( &[vk::PhysicalDeviceProperties] ) - The physical device properties.
/// - `extensions` ( &HashSet<[vk::StringArray<256>]> ) - The physical device extension support.
/// - `mem_properties` ( &[vk::PhysicalDeviceMemoryProperties] ) - The physical device memory properties.
/// - `optional_features` ( &[vk::PhysicalDeviceFeatures] ) - *Optional features* that add a **+100** to the score for each present.
/// - `optional_device_extensions` ( &[vk::ExtensionName] ) - *Optional extensions* that add a **+100** to the score for each present.
/// 
/// ## Returns
/// 
/// - `Result<u32>` - The physical device score.
fn score_physical_device(
    features: &vk::PhysicalDeviceFeatures,
    properties: &vk::PhysicalDeviceProperties,
    extensions: &HashSet<vk::StringArray<256>>,
    mem_properties: &vk::PhysicalDeviceMemoryProperties,
    optional_features: &vk::PhysicalDeviceFeatures,
    optional_device_extensions: &[vk::ExtensionName],
) -> Result<u32> {
    let feat_mem_len  = std::mem::size_of::<vk::PhysicalDeviceFeatures>()
        / std::mem::size_of::<vk::Bool32>();

    let available_feat_slice: &[vk::Bool32] = unsafe {
        std::slice::from_raw_parts(
            &features as *const _ as *const vk::Bool32,
            feat_mem_len
        )
    };

    let mut score = 0;

    match properties.device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => score += 1000,
        vk::PhysicalDeviceType::INTEGRATED_GPU => score += 100,
        _ => {}
    }

    // optional element check
    let optional_feat_slice: &[vk::Bool32] = unsafe {
        std::slice::from_raw_parts(
            optional_features as * const _ as *const vk::Bool32,
            feat_mem_len)
    };

    score += optional_feat_slice.iter().zip(available_feat_slice.iter())
        .filter(|&(&opt, &avail)| opt == vk::TRUE && avail == vk::TRUE)
        .count() as u32 * 100;

    score += optional_device_extensions.iter()
        .map(|ext| if extensions.contains(ext) {100u32} else {0}).sum::<u32>();

    // memory scoring
    let vram_score: u64 = (0..mem_properties.memory_heap_count as usize)
        .filter(|&i| mem_properties.memory_heaps[i]
            .flags
            .contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
        .map(|i| mem_properties.memory_heaps[i].size)
        .sum();

    score += (vram_score / 1_000_000) as u32;

    // ReBAR / Smart Access Memory
    let has_rebar = (0..mem_properties.memory_type_count as usize)
        .any(|i| mem_properties.memory_types[i].property_flags.contains(
            vk::MemoryPropertyFlags::DEVICE_LOCAL  |
            vk::MemoryPropertyFlags::HOST_VISIBLE  |
            vk::MemoryPropertyFlags::HOST_COHERENT
        ));
    if has_rebar { score += 500; }

    // Readback CPU
    let has_cached = (0..mem_properties.memory_type_count as usize)
        .any(|i| mem_properties.memory_types[i].property_flags.contains(
            vk::MemoryPropertyFlags::HOST_VISIBLE |
            vk::MemoryPropertyFlags::HOST_CACHED
        ));
    if has_cached { score += 100; }

    // GPU discret (2+ heaps)
    if mem_properties.memory_heap_count >= 2 { score += 200; }

    Ok(score)
}

/// Return a list of all available physical devices, scored.
/// 
/// ## Arguments
/// 
/// - `instance` ( &[Instance] ) - Vulkan instance.
/// - `surface` ( &[vk::SurfaceKHR] ) - Choosen surface. Only usefull for onscreen rendering.
/// - `mandatory_features` ( &[vk::PhysicalDeviceFeatures] ) - mandatory physical device features to implement.
/// - `optional_features` ( &[vk::PhysicalDeviceFeatures] ) - optional physical device feature to have (for scoring).
/// - `offscreen_rendering ( bool ) - true => offscreen rendering / false => onscreen rendering.
/// 
/// ## Returns
/// 
/// - `Result<Vec<(vk::PhysicalDevice, u32)>>`.
/// 
/// ## Errors
/// 
/// [SuitabilityError] - A mandatory feature is missing.
pub fn get_physical_devices(
    instance: &Instance,
    surface: vk::SurfaceKHR,
    mandatory_features: &vk::PhysicalDeviceFeatures,
    optional_features: &vk::PhysicalDeviceFeatures,
    mandatory_device_extensions: &[vk::ExtensionName],
    optional_device_extensions: &[vk::ExtensionName],
    mandatory_queue_flags: vk::QueueFlags,
    offscreen_rendering: bool,
) -> Result<Vec<(vk::PhysicalDevice, u32)>> {
    let mut scored_physical_devices: Vec<(vk::PhysicalDevice, u32)> = Vec::new();

    let physical_devices = unsafe { instance.enumerate_physical_devices()? };
    for physical_device in physical_devices {
        // 1 - physical device definition
        let props = unsafe { instance.get_physical_device_properties(physical_device) };
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let feats = unsafe { instance.get_physical_device_features(physical_device) };
        let extensions = unsafe { instance
            .enumerate_device_extension_properties(physical_device, None)?
            .iter()
            .map(|ext| ext.extension_name)
            .collect::<HashSet<_>>()
        };

        // 2 - check mandatory elements
        if !is_physical_device_supported(
            &feats,
            &extensions,
            mandatory_features,
            mandatory_device_extensions
        ) {
            continue;
        }

        if !offscreen_rendering {
            let support = SwapchainSupport::get(instance, surface, physical_device)?;
            if support.formats.is_empty() || support.present_modes.is_empty() {
                continue;
            }

            if !QueueFamilyIndices::test_for(
                &instance,
                physical_device,
                surface,
                mandatory_queue_flags,
                true,
            ) {
                continue;
            }
        }

        // 3 - scoring
        let score = score_physical_device(
            &feats,
            &props,
            &extensions,
            &mem_props,
            optional_features,
            optional_device_extensions
        );

        if let Ok(score) = score {
            scored_physical_devices.push((
                physical_device,
                score
            ));
        }
    }

    scored_physical_devices.sort_by_key(|&(_, score)| std::cmp::Reverse(score));

    Ok(scored_physical_devices)
}


/// Retrieve the best physical device usable that respect optional and mandatory requirement.
/// 
/// ## Arguments
/// 
/// - `instance` ( &[Instance] ) - Vulkan instance.
/// - `surface` ( [vk::SurfaceKHR] ) - Vulkan surface if we are doing on-screen rendering.
/// - `mandatory_features` ( &[vk::PhysicalDeviceFeatures] ) - Features that the physical device must implement to be selected.
/// - `optional_features` ( &[vk::PhysicalDeviceFeatures] ) - Features that is better to be implemented by the physical device but not mandatory.
/// - `mandatory_device_extensions` ( &[vk::ExtensionName] ) - Extension that must be supported by the physical device.
/// - `optional_device_extensions` ( &[vk::ExtensionName] ) - Extension that is better to be supported by the physical device but not mandatory.
/// - `mandatory_queue_flags` ( [vk::QueueFlags] ) - Queue flags that must be respected by the physical devices.
/// - `offscreen_rendering` ( bool ) - If we are doing offscreen rendering.
/// 
/// ## Returns
/// 
/// - `Result<vk::PhysicalDevice>` - The best physical device.
/// 
/// ## Errors
/// 
/// `SuitabilityError` - No physical device which meets all the requirements.
pub fn pick_best_physical_device(
    instance: &Instance,
    surface: vk::SurfaceKHR,
    mandatory_features: &vk::PhysicalDeviceFeatures,
    optional_features: &vk::PhysicalDeviceFeatures,
    mandatory_device_extensions: &[vk::ExtensionName],
    optional_device_extensions: &[vk::ExtensionName],
    mandatory_queue_flags: vk::QueueFlags,
    offscreen_rendering: bool,
) -> Result<vk::PhysicalDevice> {
    let scored_physical_devices = get_physical_devices(
        instance,
        surface,
        mandatory_features,
        optional_features,
        mandatory_device_extensions,
        optional_device_extensions,
        mandatory_queue_flags,
        offscreen_rendering,
    )?;

    if scored_physical_devices.is_empty() {
        return Err(anyhow!(SuitabilityError("No suitable physical device (empty list).")));
    }

    let first = scored_physical_devices.first().unwrap().0;

    let properties =  unsafe { instance.get_physical_device_properties(first) };
    info!("Selected physical device (`{}`).", properties.device_name);

    Ok(first)
}

/// Get the max msaa samples supported by the physical device.
/// 
/// ## Arguments
/// 
/// - `instance` ( &[Instance] ) - Vulkan instance.
/// - `physical_device` ( [vk::PhysicalDevice] ) - Choosen physical device.
/// 
/// ## Returns
/// 
/// - `vk::SampleCountFlags` - Flags that represent the max msaa samples supported.
pub fn get_max_msaa_samples(
	instance: &Instance,
	physical_device: vk::PhysicalDevice,
) -> vk::SampleCountFlags {
	let properties = unsafe { instance.get_physical_device_properties(physical_device) };
	let counts = properties.limits.framebuffer_color_sample_counts
		& properties.limits.framebuffer_depth_sample_counts;

	[
		vk::SampleCountFlags::_64,
		vk::SampleCountFlags::_32,
		vk::SampleCountFlags::_16,
		vk::SampleCountFlags::_8,
		vk::SampleCountFlags::_4,
		vk::SampleCountFlags::_2,
	]
	.iter()
	.cloned()
	.find(|c| counts.contains(*c))
	.unwrap_or(vk::SampleCountFlags::_1)
}

//=======================================================
// Logical Device
//=======================================================

pub fn create_logical_device(
    entry: &Entry,
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    queue_family_indices: &mut QueueFamilyIndices,
    is_validation_enable: bool,
    validation_layer: vk::ExtensionName,
    device_extensions: &[vk::ExtensionName]
) -> Result<(Device, vk::Queue, vk::Queue)> {
    // Queue Create Infos
    let mut unique_indices = HashSet::new();
    let graphics = queue_family_indices.get(vk::QueueFlags::GRAPHICS)?;
    let present = queue_family_indices.present;
    unique_indices.insert(graphics);
    unique_indices.insert(present);

    let queue_priorities = &[1.0];
    let queue_infos = unique_indices
        .iter()
        .map(|i| {
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(*i)
                .queue_priorities(queue_priorities)
        })
        .collect::<Vec<_>>();

    // Layers
    let layers = if is_validation_enable {
        vec![validation_layer.as_ptr()]
    } else {
        vec![]
    };

    // Extensions
    let mut extensions = device_extensions.iter().map(|n| n.as_ptr()).collect::<Vec<_>>();

    // Required by Vulkan SDK on macOS since 1.3.216.
    if cfg!(target_os = "macos") && entry.version()? >= PORTABILITY_MACOS_VERSION {
        extensions.push(vk::KHR_PORTABILITY_SUBSET_EXTENSION.name.as_ptr());
    }

    // Features
    let features = vk::PhysicalDeviceFeatures::builder()
        .sampler_anisotropy(true)
		// Enable sample shading features
		.sample_rate_shading(true);

    // Create
    let info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_infos)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions)
        .enabled_features(&features);

    let device = unsafe { instance.create_device(physical_device, &info, None)? };

    // Queues
    let graphics_queue = unsafe { device.get_device_queue(graphics, 0) };
    let present_queue = unsafe { device.get_device_queue(present, 0) };

    Ok((device, graphics_queue, present_queue))
}

//=======================================================
// Queue Family Indices (related to queue family memory)
//=======================================================

/// This struct hold the indices of the used queue.
/// 
/// ## Fields
/// 
/// - `properties` ( Vec<[vk::QueueFamilyProperties]> ) - All Queue properties for a specific physical device pass in [QueueFamilyIndices::create].
/// - `indices` ( HashMap<[vk::QueueFlags], u32> ) - Hashmap which associate a QueueFlags with a corresponding Queue Family (that implement the flags properties) index if it exist.
/// - `present` ( u32 ) - Present queue index.
#[derive(Clone, Debug, Default)]
pub struct QueueFamilyIndices {
    properties: Vec<vk::QueueFamilyProperties>,
    indices: HashMap<vk::QueueFlags, u32>,
    pub present: u32,
}

impl QueueFamilyIndices {
    pub fn create(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
    ) -> Result<Self> {
        let properties = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

        let mut present = None;
        unsafe {
            for (idx, _) in properties.iter().enumerate() {
                if instance.get_physical_device_surface_support_khr(physical_device, idx as u32, surface)? {
                    present = Some(idx as u32);
                    break;
                }
            }
        };

        if let Some(present) = present {
            Ok( Self {
                properties,
                indices: HashMap::new(),
                present,
            })
        } else {
            Err(anyhow!(SuitabilityError("Missing a queue that support KHR surface format.")))
        }
    }

    pub fn get(
        &mut self,
        queue_flags: vk::QueueFlags,
    ) -> Result<u32> {
        match self.indices.get(&queue_flags) {
            Some(indice) => Ok(*indice),
            None => {
                // find a queue family index
                let mut exact = None;
                let mut fallback = None;

                for (i, p) in self.properties.iter().enumerate() {
                    if p.queue_count == 0 {
                        continue;
                    }

                    // try to find a queue family with the exact flag
                    if p.queue_flags == queue_flags {
                        exact = Some(i as u32);
                        break;
                    }

                    // if we cannot find a queue family with the exact flag,
                    // find one that contain it.
                    if fallback.is_none() && p.queue_flags.contains(queue_flags) {
                        fallback = Some(i as u32);
                    } 
                }
                
                if let Some(queue_index) = exact.or(fallback) {
                    self.indices.insert(
                            queue_flags,
                            queue_index
                        );
                    Ok(queue_index)
                } else {
                    Err(anyhow!(SuitabilityError("Missing required queue families.")))
                }
            }
        }
    } 

    pub fn test_for(
        instance: &Instance,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
        queue_flags: vk::QueueFlags,
        test_khr_support: bool,
    ) -> bool {
        let properties = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

        let queue_idx = properties
            .iter()
            .position(|p| p.queue_flags.contains(queue_flags))
            .map(|i| i as u32);

        let Some(_) = queue_idx else { return false; };

        if test_khr_support {
            unsafe {
                let support_present = properties.iter().enumerate()
                    .any(|(idx, _)| {
                        instance
                            .get_physical_device_surface_support_khr(physical_device, idx as u32, surface)
                            .unwrap_or(false)
                    });

                if !support_present { return false; }
            }
        }

        true
    }
}

//=======================================================
// Compatibility
//=======================================================

use thiserror::Error;

use crate::render::SwapchainSupport;

#[derive(Debug, Error)]
#[error("{0}")]
pub struct SuitabilityError(pub &'static str);