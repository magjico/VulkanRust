use std::hash::{Hash, Hasher};

use vulkanalia::prelude::v1_0::*;

use crate::math::*;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    pub pos: Vec3,
    pub normal: Vec3,
    pub color: Vec3,
    pub tex_coord: Vec2,
}

impl Vertex {
    pub const fn new(pos: Vec3, normal: Vec3, color: Vec3, tex_coord: Vec2) -> Self {
        Self { pos, normal, color, tex_coord }
    }

    pub fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()
    }

    pub fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 4] {
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
        let tex_coord = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(3)
            .format(vk::Format::R32G32_SFLOAT)
            .offset((size_of::<Vec3>() * 3) as u32)
            .build();

        [pos, normal, color, tex_coord]
    }
}

impl PartialEq for Vertex {
    fn eq(&self, other: &Self) -> bool {
        self.pos == other.pos
            && self.normal == other.normal
            && self.color == other.color
            && self.tex_coord == other.tex_coord
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
        self.tex_coord[0].to_bits().hash(state);
        self.tex_coord[1].to_bits().hash(state);
    }
}

#[repr(C)]
#[derive(Clone, Debug)]
pub struct Material {
    pub base_color_factor: Vec4,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub emissive_factor: Vec3,

    pub base_color_texture_idx: i32,
    pub metallic_roughness_texture_idx: i32,
    pub normal_texture_idx: i32,
    pub occlusion_texture_idx: i32,
    pub emissive_texture_idx: i32
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
            emissive_texture_idx: -1
        }
    }
}


#[repr(C)]
#[derive(Clone, Debug)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub material_index: i32,
}

impl Default for Mesh {
    fn default() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
            material_index: -1
        }
    }
}