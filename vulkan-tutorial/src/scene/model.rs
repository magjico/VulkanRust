use log::*;
use anyhow::{Result, anyhow};
use cgmath::{Deg, Euler, Rad, VectorSpace};

use crate::math::*;
use crate::ops::{FlatGraph, NodeId};
use crate::assets::load_obj_model;
use crate::ops::Node;
use super::{Vertex, Mesh, Animation, Material, PathType};

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
            mesh: Mesh { vertices, indices, material_index: None },
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

/// Structure that hold the data of a model-node in a scenegraph
#[repr(C)]
#[derive(Debug)]
pub struct ModelNodeData {
    pub name: String,
    pub mesh: Option<Mesh>,

    // For animation
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,

    pub skin: i32,
}

impl Default for ModelNodeData {
    fn default() -> Self {
        Self {
            name: String::new(),
            mesh: None,
            translation: Vec3::new(0.0, 0.0, 0.0),
            rotation: Quat::new(1.0, 0.0, 0.0, 0.0),
            scale: Vec3::new(1.0, 1.0, 1.0),
            skin: -1,
        }
    }
}

impl Default for Node<ModelNodeData> {
    fn default() -> Self {
        Self {
            parent: None,
            childs: Vec::new(),
            value: ModelNodeData::default()
        }
    }
}

impl Node<ModelNodeData> {
    pub fn get_local_matrix(&self) -> Mat4 {
        // debug!("model node info:\n- translation: {:?}\n- rotation: {:?}\n- scale: {:?}", self.translation, self.rotation, self.scale);
        let translation = Mat4::from_translation(self.value.translation);
        let rotation = Mat4::from(self.value.rotation); 
        let scale = Mat4::from_nonuniform_scale(
            self.value.scale.x,
            self.value.scale.y,
            self.value.scale.z,
        );

        translation * rotation * scale
    }
}
/// Skin structure for vertex skinning.
/// Create a relationship of vertices to bones.
/// When bones transform, the vertices transform
/// along with them proportionally to their weight to the bone.
/// 
/// ## Fields
/// 
/// - `name` ( __String__ ).
/// - `skeleton_root` ( __Weak<RwLock<[Node]>>__ ) - Useful for certain type of optimisations or to place a skeleton in one go.
/// - `inverse_bind_mats` ( __Vec<[Mat4]>__ ) - Transforms the geometry into the space of the respective joint.
/// - `joints` ( __Vec<[NodeId]>__ ) - Contains the nodes used as joints in this skin.
#[repr(C)]
#[derive(Clone, Debug)]
pub struct Skin {
    pub name: String,
    pub skeleton_root: Option<NodeId>,
    pub inverse_bind_mats: Vec<Mat4>,
    pub joints: Vec<NodeId>,
}

impl Skin {
	pub fn get_joint_matrices(&self, graph: &ModelGraph) -> Vec<Mat4> {
        self.joints.iter().enumerate()
            .map(|(i, joint_id)| {
                let global = graph.get_global_matrix_of(*joint_id);
                global * self.inverse_bind_mats[i]
            })
            .collect()
    }
}

#[derive(Debug)]
pub struct ModelGraph {
	pub graph:          FlatGraph<ModelNodeData>,
    pub roots:          Vec<NodeId>,
	pub materials:		Vec<Material>,
	pub animations:		Vec<Animation>,
	pub skins:			Vec<Skin>,
}

impl ModelGraph {
    pub fn find_node(self, name: &str) -> Option<NodeId> {
        self.graph.get_iterator()
            .position(|node| node.value.name == name)
            .map(NodeId)
    }

    pub fn get_global_matrix_of(&self, id: NodeId) -> Mat4 {
        let node = self.graph.get(id);

        let mut global = node.get_local_matrix();
        let mut parent = node.parent;

        while let Some(parent_index) = parent {
            let current = self.graph.get(parent_index);

            global = current.get_local_matrix() * global;
            parent = current.parent;
        }

        global
    }

    pub fn apply_pose(
        &mut self,
        index: usize,
        current_time: f32
    ) -> Result<()> {
        if self.animations.is_empty() || index >= self.animations.len() {
            return Err(anyhow!("Invalid animation index ({}) > animations lenght ({})", index, self.animations.len()));
        }

        let animation = &self.animations[index];

        for channel in animation.get_channels_iter() {
            let sampler = animation.get_sampler(channel.sampler_id);

            let keyframe_it = sampler.inputs.partition_point(|&probe_time|
                probe_time < current_time);

            if keyframe_it == 0 || keyframe_it >= sampler.inputs.len() {
                warn!("channel.node.upgrade() failed — node was dropped!");
                return Ok(());
            } 

            let i = keyframe_it - 1;
            let interp_factor = (current_time - sampler.inputs[i])
                / (sampler.inputs[i + 1] - sampler.inputs[i]);
                
            
            let node = self.graph.get_mut(channel.node_id);

            match channel.path {
                PathType::TRANSLATION => {
                    let start = sampler.outputs_vec3[i];
                    let end = sampler.outputs_vec3[i + 1];
                    node.value.translation = start.lerp(end, interp_factor);
                }
                PathType::ROTATION => {
                    let start = vec4_to_quat(sampler.outputs_vec4[i]);
                    let end = vec4_to_quat(sampler.outputs_vec4[i + 1]);
                    node.value.rotation = start.slerp(end, interp_factor);
                }
                PathType::SCALE => {
                    let start = sampler.outputs_vec3[i];
                    let end = sampler.outputs_vec3[i + 1];
                    node.value.scale = start.lerp(end, interp_factor);
                }
                // TODO: support morph type animation
                PathType::MORPH => warn!("Morph type animation not yet supported."),
            }
        }
        Ok(())
    }

    pub fn get_debug_info(&self) -> Result<()> {
        let mut min_pos = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
        let mut max_pos = Vec3::new(f32::MIN, f32::MIN, f32::MIN);

        
        for (i, node) in self.graph.get_iterator().enumerate() {
            let global_mat = self.get_global_matrix_of(NodeId(i));

            if node.value.name == "Z_UP" {
                debug!("Z_UP transform: t={:?} r={:?} s={:?}", 
                    node.value.translation, node.value.rotation, node.value.scale);
            
                debug!("Z_UP global matrix: {:?}", global_mat);
            }

            if node.value.mesh.is_none() {
                debug!("node '{}' has no mesh", node.value.name);
                continue;
            }
            let mesh = node.value.mesh.as_ref().unwrap();
            debug!("node '{}' has {} vertices", node.value.name, mesh.vertices.len());

            for vertex in &mesh.vertices {
                let world_pos = global_mat * Vec4::new(
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

            let center = (min_pos + max_pos) / 2.0;
            let size   = max_pos - min_pos;

            debug!(
                "Model Debug Info:\n\t- min position: {:?}\n\t- max position: {:?}\n\t- center: {:?}\n\t- size: {:?}",
                min_pos, max_pos, center, size
            );
        };
        
        Ok(())
    }
}