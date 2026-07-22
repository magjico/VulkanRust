use bevy_ecs::prelude::*;
use vulkanalia::prelude::v1_0::*;

use crate::scene::ModelGraph;
use crate::ops::SlotAllocator;
use crate::type_safety::ModelId;

#[derive(Resource, Default)]
pub struct Time(pub f32);

#[derive(Resource)]
pub struct SSBOSkiningAllocator(pub SlotAllocator);

#[derive(Resource)]
pub struct ModelsStorage(pub Vec<ModelGraph>);

impl ModelsStorage {
	pub fn get_model(&self, model_id: ModelId) -> &ModelGraph {
		self.0.get(model_id.0)
			.unwrap_or_else(|| panic!("model id ({}) out of bounds for length {}", model_id.0, self.0.len()))
	}

	pub fn get_mut_model(&mut self, model_id: ModelId) -> &mut ModelGraph {
		let len = self.0.len();
		self.0.get_mut(model_id.0)
			.unwrap_or_else(|| panic!("model id ({}) out of bounds for length {}", model_id.0, len))
	}
}

#[derive(Resource)]
pub struct VulkanDevice(pub Device);