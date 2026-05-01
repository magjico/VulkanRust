use anyhow::{Result, anyhow};
use log::*;

use std::collections::HashSet;

use winit::window::Window;

use vulkanalia::window as vk_window;
use vulkanalia::vk::ExtDebugUtilsExtensionInstanceCommands;
use vulkanalia::prelude::v1_0::*;

use crate::constants::{VALIDATION_ENABLED, VALIDATION_LAYER, ENGINE_NAME, APP_NAME,
    PORTABILITY_MACOS_VERSION};
use crate::debug::debug_callback;

pub fn create_instance(window: &Window, entry: &Entry) -> Result<(Instance, Option<vk::DebugUtilsMessengerEXT>)> {
    // Application Info
    let application_info = vk::ApplicationInfo::builder()
        .application_name(APP_NAME)
        .application_version(vk::make_version(1, 0, 0))
        .engine_name(ENGINE_NAME)
        .engine_version(vk::make_version(1, 0, 0))
        .api_version(vk::make_version(1, 2, 0));

    // Layers
    let available_layers = unsafe { entry
        .enumerate_instance_layer_properties()?
        .iter()
        .map(|l| l.layer_name)
        .collect::<HashSet<_>>() };

    if VALIDATION_ENABLED && !available_layers.contains(&VALIDATION_LAYER) {
        return Err(anyhow!("Validation layer requested but not supported."));
    }

    let layers = if VALIDATION_ENABLED {
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        Vec::new()
    };

    // Extensions
    let mut extensions = vk_window::get_required_instance_extensions(window)
        .iter()
        .map(|e| e.as_ptr())
        .collect::<Vec<_>>();

    // Required by Vulkan SDK on macOS since 1.3.216.
    let flags = if cfg!(target_os = "macos") && entry.version()? >= PORTABILITY_MACOS_VERSION {
        info!("Enabling extensions for macOS portability.");
        extensions.push(vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_EXTENSION.name.as_ptr());
        extensions.push(vk::KHR_PORTABILITY_ENUMERATION_EXTENSION.name.as_ptr());
        vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
    } else {
        vk::InstanceCreateFlags::empty()
    };

    if VALIDATION_ENABLED {
        extensions.push(vk::EXT_DEBUG_UTILS_EXTENSION.name.as_ptr());
    }

    // Create
    let mut info = vk::InstanceCreateInfo::builder()
        .application_info(&application_info)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions)
        .flags(flags);

    let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .user_callback(Some(debug_callback));

    if VALIDATION_ENABLED {
        info = info.push_next(&mut debug_info);
    }

    let instance = unsafe { entry.create_instance(&info, None)? };

    // Logger
    let mut messenger = None;
    if VALIDATION_ENABLED {
        messenger = unsafe { Some(instance.create_debug_utils_messenger_ext(&debug_info, None)?) };
    }

    Ok((instance, messenger))
}