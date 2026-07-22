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
pub struct MeshOffset {
	pub vertex_offset: u32,
	pub first_index: u32
}