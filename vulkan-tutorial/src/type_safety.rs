use std::fmt::Display;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NodeId(pub usize);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MaterialId(pub usize);

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

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MeshOffset {
	pub vertex_offset: u32,
	pub first_index: u32
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