use std::collections::HashMap;
use std::slice::{Iter, IterMut};

use bevy_ecs::prelude::*;
use vulkanalia::prelude::v1_0::*;

use crate::scene::ModelGraph;
use crate::ops::SlotAllocator;
use crate::type_safety::{ModelId, TextureId};

#[derive(Resource, Default)]
pub struct Time(pub f32);

#[derive(Resource)]
pub struct SSBOSkiningAllocator(pub SlotAllocator);

#[derive(Resource, Clone, Debug)]
pub struct ModelsStorage(pub Vec<ModelGraph>);

impl ModelsStorage {
	#[inline]
	pub fn get_model(&self, model_id: ModelId) -> &ModelGraph {
		self.0.get(model_id.0)
			.unwrap_or_else(|| panic!("model id ({}) out of bounds for length {}", model_id.0, self.0.len()))
	}

	#[inline]
	pub fn get_mut_model(&mut self, model_id: ModelId) -> &mut ModelGraph {
		let len = self.0.len();
		self.0.get_mut(model_id.0)
			.unwrap_or_else(|| panic!("model id ({}) out of bounds for length {}", model_id.0, len))
	}

	#[inline]
	pub fn get_next_id(&self) -> ModelId {
		ModelId(self.0.len())
	}

	#[inline]
	pub fn iter(&self) -> Iter<'_, ModelGraph> {
		self.0.iter()
	}

	#[inline]
	pub fn iter_mut(&mut self) -> IterMut<'_, ModelGraph> {
		self.0.iter_mut()
	}

	#[inline]
	pub fn push(&mut self, graph: ModelGraph) -> ModelId {
		let id = self.get_next_id();
		self.0.push(graph);
		id
	}
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ModelAssets {
	pub model_id: ModelId,
	pub texture_id: TextureId
}

#[derive(Resource)]
pub struct ModelRegistry {
	pub entries: HashMap<String, ModelAssets>
}

#[derive(Resource)]
pub struct VulkanDevice(pub Device);