use std::fmt::Display;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NodeId(pub usize);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MaterialId(pub usize);

/// Global ID inside the sets of material descriptor
/// 
/// Contrary to [MaterialId], that index materials with a unique [crate::scene::ModelGraph],
/// this ID covers the concatenation of materials from all loaded models.
/// 
/// [MaterialSetId] = [MaterialId] + `material_offset`
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct MaterialSetId(pub usize);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SkinId(pub usize);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SamplerId(pub usize);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct AnimationId(pub usize);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ModelId(pub usize);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TextureId(pub usize);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct MeshOffset {
	pub vertex_offset: u32,
	pub first_index: u32
}

impl MaterialId {
	#[inline]
	pub fn to_set_id(&self, material_offset: MaterialId) -> MaterialSetId {
		MaterialSetId(self.0 + material_offset.0)
	}
}

impl std::ops::Add<MaterialId> for MaterialId {
	type Output = MaterialId;

	fn add(self, rhs: MaterialId) -> Self::Output {
		MaterialId(self.0 + rhs.0)
	}
}

impl std::ops::Add<TextureId> for TextureId {
	type Output = TextureId;

	fn add(self, rhs: TextureId) -> Self::Output {
		TextureId(self.0 + rhs.0)
	}
}

impl Display for TextureId {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		write!(f, "{}", self.0)
	}
}