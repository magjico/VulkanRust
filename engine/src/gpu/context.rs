use anyhow::Result;

use vulkanalia::prelude::v1_0::*;

use super::{QueueFamilyIndices, get_max_msaa_samples, pick_best_physical_device, create_logical_device};
use crate::constants::{VALIDATION_ENABLED, VALIDATION_LAYER, DEVICE_EXTENSIONS};

//=======================================================
// GPU-Context (regroup all device struct into one for the app management)
//=======================================================

// TODO: maybe transform GPURequirements and GPUContext into a builder architecture object.

/// Store the wanted [vk::PhysicalDevice] features/extensions/parameters to implements.
#[derive(Debug)]
pub struct GPURequirements {
	pub mandatory_features:		vk::PhysicalDeviceFeatures,
	pub optional_features:		vk::PhysicalDeviceFeatures,
	pub mandatory_extensions:	Vec<vk::ExtensionName>,
	pub optional_extensions:	Vec<vk::ExtensionName>,
	pub queue_flags:			vk::QueueFlags,
}

#[derive(Debug)]
pub struct GPUContext {
	pub physical_device:		vk::PhysicalDevice,
	pub graphics_queue:			vk::Queue,
	pub present_queue:			vk::Queue,
	pub max_msaa_samples:		vk::SampleCountFlags,
	pub queue_family_indices:	QueueFamilyIndices,
}

impl GPUContext {
	/// Create a [GPUContext] and a Vulkan [Device].
	/// 
	/// _ps: [Device] is separated from the rest because it is used by absolutely everythings._
	/// 
	/// ## Arguments
	/// 
	/// - `entry` ( &[Entry] ) - Vulkan entry point.
	/// - `instance` ( &[Instance] ) - Vulkan instance.
	/// - `surface` ( [vk::SurfaceKHR] ) - Vulkan window surface.
	/// - `requirements` ( &[GPURequirements] ) - GPU requirements to pick a [vk::PhysicalDevice].
	/// - `disable_screen_rendering` ( `bool` ) - enable offscreen rendering.
	/// 
	/// ## Returns
	/// 
	/// - `Result<(Self, Device)>` - The GPUContext and the Vulkan Device.
	pub fn create(
		entry:						&Entry,
		instance:					&Instance,
		surface:					vk::SurfaceKHR,
		requirements:				&GPURequirements,
		disable_screen_rendering:	bool
	) -> Result<(Self, Device)> {
		let physical_device = pick_best_physical_device(
			instance,
			surface,
			&requirements.mandatory_features,
			&requirements.optional_features,
			&requirements.mandatory_extensions,
			&requirements.optional_extensions,
			requirements.queue_flags,
			disable_screen_rendering
		)?;

		let max_msaa_samples = get_max_msaa_samples(instance, physical_device);

		let mut queue_family_indices = QueueFamilyIndices::create(
			instance,
			physical_device,
			surface
		)?;
		let (device, graphics_queue, present_queue) = create_logical_device(
            entry,
            instance,
            physical_device,
            &mut queue_family_indices,
            VALIDATION_ENABLED,
            VALIDATION_LAYER,
            DEVICE_EXTENSIONS
        )?;

		Ok((
			Self {
				physical_device,
				graphics_queue,
				present_queue,
				max_msaa_samples,
				queue_family_indices
			},
			device
		))
	}
}