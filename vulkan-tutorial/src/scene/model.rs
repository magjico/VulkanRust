use std::rc::{Rc, Weak};
use std::cell::RefCell;

use log::*;
use anyhow::{Result, anyhow};
use cgmath::{Deg, Euler, Rad, VectorSpace};

use vulkanalia::prelude::v1_0::*;

use crate::math::*;
use crate::assets::load_obj_model;
use crate::scene::Skin;
use super::{Vertex, Mesh, Node, Animation, Material, PathType};

//===============================================
// Model-Instance 
//===============================================

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ModelInstance {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl ModelInstance {
    pub const fn new(position: Vec3, rotation: Quat, scale: Vec3) -> Self {
        Self {
            position,
            rotation,
            scale,
        }
    }

    pub fn from_radians(position: Vec3, rotation: Vec3, scale: Vec3) -> Self {
        Self {
            position,
            rotation: Quat::from(Euler {
                x: Rad(rotation.x),
                y: Rad(rotation.y),
                z: Rad(rotation.z),
            }),
            scale
        }
    }

    pub fn from_degrees(position: Vec3, rotation: Vec3, scale: Vec3) -> Self {
        Self {
            position,
            rotation: Quat::from(Euler {
                x: Deg(rotation.x),
                y: Deg(rotation.y),
                z: Deg(rotation.z)
            }),
            scale
        }
    }

    pub fn to_model_matrix(&self) -> Mat4 {
        let translation = Mat4::from_translation(self.position);
        let rotation = Mat4::from(self.rotation); 
        let scale = Mat4::from_nonuniform_scale(
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
            mesh: Mesh { vertices, indices, material_index: -1 },
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

//===============================================
// Model-Graph
//===============================================

#[derive(Clone, Debug)]
pub struct ModelGraph {
    pub nodes: Vec<Rc<RefCell<Node>>>,
    pub linear_nodes: Vec<Weak<RefCell<Node>>>,
    pub materials: Vec<Material>,
    pub animations: Vec<Animation>,

    pub skins: Vec<Skin>,
}

impl ModelGraph {
    pub fn find_node(self, name: &str) -> Option<Rc<RefCell<Node>>> {
        self.linear_nodes
            .iter()
            .find(|weak_ref| {
                weak_ref.upgrade()
                    .map(|node| node.borrow().name == name)
                    .unwrap_or(false)
            })
            .and_then(|weak_ref| weak_ref.upgrade())
    }

    pub fn update_animation(&mut self, index: usize, delta_time: f32) -> Result<()> {
        if self.animations.is_empty() || index >= self.animations.len() {
            return Err(anyhow!("Invalid animation requested (empty animation list or ask for an animation index > animations lenght)"));
        }

        let animation = &mut self.animations[index];

        animation.current_time += delta_time;
        while animation.current_time >= animation.end {
            animation.current_time -= animation.end - animation.start;
        }

        animation.channels.iter()
            .try_for_each(|channel| {
                if channel.sampler_index >= animation.samplers.len() {
                    return Err(anyhow!("channel sampler index {} >= animation samplers size {}", channel.sampler_index, animation.samplers.len()));
                }
                let sampler = &animation.samplers[channel.sampler_index];

                let keyframe_it = sampler.inputs.partition_point(|&probe_time|
                    probe_time < animation.current_time);

                if keyframe_it < sampler.inputs.len() && keyframe_it > 0 {
                    let i = keyframe_it - 1;

                    let interp_factor = (animation.current_time - sampler.inputs[i])
                        / (sampler.inputs[i + 1] - sampler.inputs[i]);
                    
                    if let Some(node) = channel.node.upgrade() {
                        let mut node = node.borrow_mut();

                        match channel.path {
                            PathType::TRANSLATION => {
                                let start = sampler.outputs_vec3[i];
                                let end = sampler.outputs_vec3[i + 1];
                                node.translation = start.lerp(end, interp_factor);
                            }
                            PathType::ROTATION => {
                                let start = vec4_to_quat(sampler.outputs_vec4[i]);
                                let end = vec4_to_quat(sampler.outputs_vec4[i + 1]);
                                node.rotation = start.slerp(end, interp_factor);
                            }
                            PathType::SCALE => {
                                let start = sampler.outputs_vec3[i];
                                let end = sampler.outputs_vec3[i + 1];
                                node.scale = start.lerp(end, interp_factor);
                            }
                            // TODO: support morph type animation
                            PathType::MORPH => {
                                warn!("Morph type animation not yet supported.")
                            },
                        }
                    }
                }
                else {
                    warn!("channel.node.upgrade() failed — node was dropped!");
                }
                Ok(())
            })?;

        Ok(())
    }

    pub fn update_animations(&mut self, delta_time: f32) {
        for i in 0..self.animations.len() {
            debug!("animation {} current time: {}", i, self.animations[i].current_time);
            if let Err(e) = self.update_animation(i, delta_time) {
                warn!("Animation update failed: {}", e);
            }
        }
    }

    #[rustfmt::skip]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn destroy(&self, device: &Device) {
        self.skins.iter().for_each(|skin| skin.destroy(device));
    }
}