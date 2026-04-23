use anyhow::Result;

use cgmath::{Vector3, Rad, Matrix4};

use crate::math::*;
use crate::assets::load_obj_model;
use crate::geometry::{Vertex, Mesh};

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ModelInstance {
    pub position: Vector3<f32>,
    pub rotation: Vector3<Rad<f32>>,
    pub scale: Vector3<f32>,
}

impl ModelInstance {
    pub fn new(position: Vec3, rotation_deg: Vec3, scale: Vec3) -> Self {
        Self {
            position,
            rotation: Vector3::new(
                Rad(rotation_deg.x.to_radians()),
                Rad(rotation_deg.y.to_radians()),
                Rad(rotation_deg.z.to_radians()),
            ),
            scale,
        }
    }

    pub fn from_radians(position: Vector3<f32>, rotation: Vector3<Rad<f32>>, scale: Vector3<f32>) -> Self {
        Self { position, rotation, scale }
    }


    pub fn to_model_matrix(&self) -> Mat4 {
        let translation = Matrix4::from_translation(self.position);
        let rotation = Matrix4::from_angle_x(self.rotation.x)
                    * Matrix4::from_angle_y(self.rotation.y)
                    * Matrix4::from_angle_z(self.rotation.z);
        let scale = Matrix4::from_nonuniform_scale(
            self.scale.x,
            self.scale.y,
            self.scale.z,
        );

        translation * rotation * scale
    }
}


#[derive(Clone, Debug)]
pub struct Model {
    pub mesh: Mesh,
    pub instances: Vec<ModelInstance>
}

impl Model {
    pub fn create(vertices: Vec<Vertex>, indices: Vec<u32>) -> Self {
        Self {
            mesh: Mesh { vertices, indices },
            instances: Vec::new()
        }
    }

    pub fn create_with_obj(obj_path: &str) -> Result<Self> {
        Ok( Self {
            mesh: load_obj_model(obj_path)?,
            instances: Vec::new(),
        })
    }

    pub fn add_instance(&mut self, instance: ModelInstance) {
        self.instances.push(instance);
    }

    pub fn add_instances(&mut self, instances: &[ModelInstance]) {
        self.instances.extend_from_slice(instances);
    }
}