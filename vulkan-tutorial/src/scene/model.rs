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
            if let Err(e) = self.update_animation(i, delta_time) {
                warn!("Animation update failed: {}", e);
            }
        }
    }

    pub fn get_debug_info(&self) -> Result<()> {
        let mut min_pos = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
        let mut max_pos = Vec3::new(f32::MIN, f32::MIN, f32::MIN);

        for w_node in &self.linear_nodes {
            let Some(ref_node) = w_node.upgrade() else {
                debug!("node upgrade failed");
                continue
            };
            let node = ref_node.borrow();

            if node.name == "Z_UP" {
                debug!("Z_UP transform: t={:?} r={:?} s={:?}", 
                    node.translation, node.rotation, node.scale);
                
                debug!("Z_UP global matrix: {:?}", node.get_global_matrix());
            }

            if node.mesh.is_none() {
                debug!("node '{}' has no mesh", node.name);
                continue;
            }
            let mesh = node.mesh.as_ref().unwrap();
            debug!("node '{}' has {} vertices", node.name, mesh.vertices.len());

            let global_matrix = node.get_global_matrix();

            for vertex in &mesh.vertices {
                let world_pos = global_matrix * Vec4::new(
                    vertex.pos.x,
                    vertex.pos.y,
                    vertex.pos.z,
                    1.0
                );

                min_pos.x = min_pos.x.min(world_pos.x);
                min_pos.y = min_pos.y.min(world_pos.y);
                min_pos.z = min_pos.z.min(world_pos.z);
                max_pos.x = max_pos.x.max(world_pos.x);
                max_pos.y = max_pos.y.max(world_pos.y);
                max_pos.z = max_pos.z.max(world_pos.z);
            }
        }

        let center = (min_pos + max_pos) / 2.0;
        let size   = max_pos - min_pos;

        debug!(
            "Model Debug Info:\n\t- min position: {:?}\n\t- max position: {:?}\n\t- center: {:?}\n\t- size: {:?}",
            min_pos, max_pos, center, size
        );

        Ok(())
    }

    #[rustfmt::skip]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn destroy(&self, device: &Device) {
        self.skins.iter().for_each(|skin| skin.destroy(device));
    }
}