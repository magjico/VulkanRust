use bevy_ecs::prelude::*;
use vulkanalia::prelude::v1_0::*;

use crate::scene::ModelGraph;
use crate::ops::SlotAllocator;

#[derive(Resource, Default)]
pub struct Time(pub f32);

#[derive(Resource)]
pub struct SSBOSkiningAllocator(pub SlotAllocator);

#[derive(Resource)]
pub struct ModelsStorage(pub Vec<ModelGraph>);

#[derive(Resource)]
pub struct VulkanDevice(pub Device);