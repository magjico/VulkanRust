use std::ffi::CStr;
use std::os::raw::c_void;

use log::*;

use vulkanalia::prelude::v1_0::*;

/// ## Safety
///
/// `data` must be a valid, non-null pointer to a [vk::DebugUtilsMessengerCallbackDataEXT]
/// that stays alive for the duration of the call, with a valid null-terminated string
/// in its `message` field.
///
/// This function is not meant to be called directly — the Vulkan loader invokes it,
/// and satisfies these requirements.
pub unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    type_: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> vk::Bool32 {
    let data = unsafe { *data };
    let message = unsafe { CStr::from_ptr(data.message) }.to_string_lossy();

    if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::ERROR {
        error!("({:?}) {}", type_, message);
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::WARNING {
        warn!("({:?}) {}", type_, message);
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::INFO {
        debug!("({:?}) {}", type_, message);
    } else {
        trace!("({:?}) {}", type_, message);
    }

    vk::FALSE
}
