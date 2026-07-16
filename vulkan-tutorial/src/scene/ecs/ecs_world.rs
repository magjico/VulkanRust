use anyhow::Result;

use vulkanalia::prelude::v1_0::*;
use bevy_ecs::prelude::*;

use crate::ops::SlotAllocator;
use crate::constants::{MAX_INSTANCES, MAX_JOINT_PER_INSTANCE};
use super::{SSBOSkiningAllocator, Time, SkinningBuffer};


pub fn init_world(
	device: &Device,
	instance: &Instance,
	physical_device: vk::PhysicalDevice,
) -> Result<World> {
	let mut world = World::new();

	let skinning_buffer = SkinningBuffer::create(instance, device, physical_device)?;
	world.insert_resource(skinning_buffer);

	world.insert_resource(SSBOSkiningAllocator(SlotAllocator::new(MAX_INSTANCES, MAX_JOINT_PER_INSTANCE)));
	world.insert_resource(Time::default());
	Ok(world)
}