use std::hash::{Hash, Hasher};

use vulkanalia::prelude::v1_0::*;

use crate::math::*;
use crate::type_safety::MaterialId;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    pub pos: Vec3,
    pub normal: Vec3,
    pub color: Vec3,
    pub uv0: Vec2,
    pub uv1: Vec2,
    pub tangent: Vec4, // xyz = tangent, w = handedness

    pub joint_indices: UVec4,
    pub joint_weights: Vec4,
}

impl Vertex {
    pub const fn new(
        pos: Vec3,
        normal: Vec3,
        color: Vec3,
        uv0: Vec2,
        uv1: Vec2,
        tangent: Vec4,
        joint_indices: UVec4,
        joint_weights: Vec4,
    ) -> Self {
        Self {
            pos,
            normal,
            color,
            uv0,
            uv1,
            tangent,
            joint_indices,
            joint_weights
        }
    }

    pub fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()
    }

    pub fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 8] {
        let pos = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0)
            .build();
        let normal = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(size_of::<Vec3>() as u32)
            .build();
        let color = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(2)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset((size_of::<Vec3>() * 2) as u32)
            .build();
        let uv0 = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(3)
            .format(vk::Format::R32G32_SFLOAT)
            .offset((size_of::<Vec3>() * 3) as u32)
            .build();
        let uv1 = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(4)
            .format(vk::Format::R32G32_SFLOAT)
            .offset((size_of::<Vec3>() * 3 + size_of::<Vec2>()) as u32)
            .build();
        let tangent = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(5)
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .offset((size_of::<Vec3>() * 3 + size_of::<Vec2>() * 2) as u32)
            .build();
        let joint_indices = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(6)
            .format(vk::Format::R16G16B16A16_UINT)
            .offset((size_of::<Vec4>() + size_of::<Vec3>() * 3 + size_of::<Vec2>() * 2) as u32)
            .build();
        let joint_weights = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(7)
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .offset((size_of::<UVec4>() + size_of::<Vec4>() + size_of::<Vec3>() * 3 + size_of::<Vec2>() * 2) as u32)
            .build();

        [pos, normal, color, uv0, uv1, tangent, joint_indices, joint_weights]
    }
}

impl PartialEq for Vertex {
    fn eq(&self, other: &Self) -> bool {
        self.pos == other.pos
            && self.normal == other.normal
            && self.color == other.color
            && self.uv0 == other.uv0
            && self.uv1 == other.uv1
            && self.tangent == other.tangent
            && self.joint_indices == other.joint_indices
            && self.joint_weights == other.joint_weights
    }
}

impl Eq for Vertex {}

impl Hash for Vertex {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.pos[0].to_bits().hash(state);
        self.pos[1].to_bits().hash(state);
        self.pos[2].to_bits().hash(state);
        self.normal[0].to_bits().hash(state);
        self.normal[1].to_bits().hash(state);
        self.normal[2].to_bits().hash(state);
        self.color[0].to_bits().hash(state);
        self.color[1].to_bits().hash(state);
        self.color[2].to_bits().hash(state);
        self.uv0[0].to_bits().hash(state);
        self.uv0[1].to_bits().hash(state);
        self.uv1[0].to_bits().hash(state);
        self.uv1[1].to_bits().hash(state);
        self.tangent[0].to_bits().hash(state);
        self.tangent[1].to_bits().hash(state);
        self.tangent[2].to_bits().hash(state);
        self.tangent[3].to_bits().hash(state);
        self.joint_indices[0].hash(state);
        self.joint_indices[1].hash(state);
        self.joint_indices[2].hash(state);
        self.joint_indices[3].hash(state);
        self.joint_weights[0].to_bits().hash(state);
        self.joint_weights[1].to_bits().hash(state);
        self.joint_weights[2].to_bits().hash(state);
        self.joint_weights[3].to_bits().hash(state);
    }
}

#[repr(C)]
#[derive(Clone, Debug)]
pub struct Material {
    pub base_color_factor: Vec4,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub emissive_factor: Vec3,

    // Texture indices
    pub base_color_texture_idx: i32,
    pub metallic_roughness_texture_idx: i32,
    pub normal_texture_idx: i32,
    pub occlusion_texture_idx: i32,
    pub emissive_texture_idx: i32,

    // Texture sets (which UV to use)
    pub base_color_texture_set: i32,
    pub metallic_roughness_texture_set: i32,
    pub normal_texture_set: i32,
    pub occlusion_texture_set: i32,
    pub emissive_texture_set: i32,

    // Alpha-mask
    pub alpha_mask: f32,
    pub alpha_mask_cutoff: f32,
}

impl Material {
    pub const fn new() -> Self {
        Material {
            base_color_factor: Vec4::new(1.0, 1.0, 1.0, 1.0),
            metallic_factor: 1.0,
            roughness_factor: 1.0,
            emissive_factor: Vec3::new(1.0, 1.0, 1.0),

            base_color_texture_idx: -1,
            metallic_roughness_texture_idx: -1,
            normal_texture_idx: -1,
            occlusion_texture_idx: -1,
            emissive_texture_idx: -1,

            base_color_texture_set: -1,
            metallic_roughness_texture_set: -1,
            normal_texture_set: -1,
            occlusion_texture_set: -1,
            emissive_texture_set: -1,

            alpha_mask: 0.0,
            alpha_mask_cutoff: 0.5,
        }
    }
}

#[repr(C)]
#[derive(Clone, Debug)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub material_index: Option<MaterialId>,
}

// TODO: mesh builder struct ?
impl Default for Mesh {
    fn default() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
            material_index: None
        }
    }
}