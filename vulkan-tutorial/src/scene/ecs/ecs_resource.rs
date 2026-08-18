use std::collections::HashMap;
use std::slice::{Iter, IterMut};

use bevy_ecs::prelude::*;
use vulkanalia::prelude::v1_0::*;

use crate::scene::ModelGraph;
use crate::ops::SlotAllocator;
use crate::type_safety::{ModelId, TextureId, MaterialId};

#[derive(Resource, Default)]
pub struct Time(pub f32);

#[derive(Resource)]
pub struct SSBOSkiningAllocator(pub SlotAllocator);

#[derive(Resource, Clone, Debug)]
pub struct ModelsStorage {
	models: Vec<ModelGraph>,
	material_offsets: Vec<MaterialId>,
}

impl ModelsStorage {
	#[inline]
	pub fn new() -> Self {
		ModelsStorage { models: Vec::new(), material_offsets: Vec::new() }
	}

	#[inline]
	pub fn get_model(&self, model_id: ModelId) -> &ModelGraph {
		self.models.get(model_id.0)
			.unwrap_or_else(|| panic!("model id ({}) out of bounds for length {}", model_id.0, self.models.len()))
	}

	#[inline]
	pub fn get_mut_model(&mut self, model_id: ModelId) -> &mut ModelGraph {
		let len = self.models.len();
		self.models.get_mut(model_id.0)
			.unwrap_or_else(|| panic!("model id ({}) out of bounds for length {}", model_id.0, len))
	}

	#[inline]
	pub fn get_material_offset(&self, model_id: ModelId) -> MaterialId {
		self.material_offsets[model_id.0]
	}

	#[inline]
	pub fn get_next_id(&self) -> ModelId {
		ModelId(self.models.len())
	}

	#[inline]
	pub fn iter(&self) -> Iter<'_, ModelGraph> {
		self.models.iter()
	}

	#[inline]
	pub fn iter_mut(&mut self) -> IterMut<'_, ModelGraph> {
		self.models.iter_mut()
	}

	#[inline]
	pub fn push(&mut self, graph: ModelGraph) -> ModelId {
		let id = self.get_next_id();
		let offset = MaterialId(self.total_materials());

		self.models.push(graph);
		self.material_offsets.push(offset);
		id
	}

	#[inline]
	pub fn total_materials(&self) -> usize {
		self.iter().map(|model| model.materials_iter().count()).sum()
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

impl ModelRegistry {
	#[inline]
	pub fn new() -> Self {
		ModelRegistry { entries: HashMap::new() }
	}
}

#[derive(Resource)]
pub struct VulkanDevice(pub Device);

#[derive(Resource, Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct CurrentFrame(pub usize);