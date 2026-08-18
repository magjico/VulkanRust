use std::slice::{Iter, IterMut};

use cgmath::SquareMatrix;
use log::*;
use anyhow::{Result, anyhow};
use cgmath::VectorSpace;

use crate::math::*;
use crate::type_safety::*;
use crate::ops::{FlatGraph, Node};

use super::{Mesh, Animation, Material, PathType};

// region Model-Graph
/// Describe a type of model node transformation for animation
#[derive(Debug, Clone, Copy)]
pub enum NodeTransform {
    Trs { translation: Vec3, rotation: Quat, scale: Vec3 },
    Matrix(Mat4),
}

impl NodeTransform {
    pub fn get_matrix(&self) -> Mat4 {
        match self {
            NodeTransform::Trs { translation, rotation, scale } => {
                Mat4::from_translation(*translation)
                    * Mat4::from(*rotation)
                    * Mat4::from_nonuniform_scale(scale.x, scale.y, scale.z)
            }
            NodeTransform::Matrix(m) => *m
        }
    }

    pub fn get_trs(&self) -> (Vec3, Quat, Vec3) {
        match self {
            NodeTransform::Trs { translation, rotation, scale }
                => (*translation, *rotation, *scale),
            NodeTransform::Matrix(m) => decompose(m)
        }
    }

    pub fn get_scale(&self) -> Vec3 {
        match self {
            NodeTransform::Trs { scale, .. } => *scale,
            NodeTransform::Matrix(m) => extract_scale_from_mat4(&m),
        }
    }

    pub fn set_translation(&mut self, trans: Vec3) {
        match self {
            NodeTransform::Trs { translation, .. } => *translation = trans,
            NodeTransform::Matrix(m) => {
                warn!("<set_translation> - animating a node with a matrix transform — glTF spec violation, writing directly into matrix transform");
                m.w.x = trans.x;
                m.w.y = trans.y;
                m.w.z = trans.z;
            }
        }
    }

    pub fn set_rotation(&mut self, rot: Quat) {
        let scale = self.get_scale();

        match self {
            NodeTransform::Trs { rotation, .. } => *rotation = rot,
            NodeTransform::Matrix(m) => {     
                warn!("<set_rotation> - animating a node with a matrix transform — glTF spec violation, writing directly into matrix transform");   
                let rotation = Mat3::from(rot);

                m.x.x = rotation.x.x * scale.x;
                m.x.y = rotation.x.y * scale.x;
                m.x.z = rotation.x.z * scale.x;

                m.y.x = rotation.y.x * scale.y;
                m.y.y = rotation.y.y * scale.y;
                m.y.z = rotation.y.z * scale.y;

                m.z.x = rotation.z.x * scale.z;
                m.z.y = rotation.z.y * scale.z;
                m.z.z = rotation.z.z * scale.z;
            }
        }
    }

    pub fn set_scale(&mut self, scl: Vec3) {
        match self {
            NodeTransform::Trs { scale, .. } => *scale = scl,
            NodeTransform::Matrix(m) => {
                warn!("<set_scale> - animating a node with a matrix transform — glTF spec violation, writing directly into matrix transform");
                let (x, y, z) = extract_normalize_rot_from_mat4(m);

                m.x.x = x.x * scl.x; 
                m.x.y = x.y * scl.x;
                m.x.z = x.z * scl.x;

                m.y.x = y.x * scl.y;
                m.y.y = y.y * scl.y;
                m.y.z = y.z * scl.y;
                
                m.z.x = z.x * scl.z;
                m.z.y = z.y * scl.z;
                m.z.z = z.z * scl.z;
            }
        }
    }
}

/// Structure that hold the data of a model-node in a scenegraph
#[repr(C)]
#[derive(Debug, Clone)]
pub struct ModelNodeData {
    pub name: String,
    pub mesh: Option<Mesh>,

    // For animation
    pub transform: NodeTransform,
    pub global_transform: Mat4, // The result of multiple TRS transformations is not always representable as a TRS. That's why its a Mat4.
    pub skin: i32,
}

impl Default for ModelNodeData {
    fn default() -> Self {
        let transform = NodeTransform::Trs {
            translation: Vec3::new(0.0, 0.0, 0.0),
            rotation: Quat::new(1.0, 0.0, 0.0, 0.0),
            scale: Vec3::new(1.0, 1.0, 1.0)
        };

        Self {
            name: String::new(),
            mesh: None,
            transform: transform,
            global_transform: Mat4::identity(),
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
    #[inline]
    pub fn get_local_matrix(&self) -> Mat4 {
        self.value.transform.get_matrix()
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
#[derive(Debug, Clone)]
pub struct ModelGraph {
	pub graph:		FlatGraph<ModelNodeData>,
    pub roots:		Vec<NodeId>,
	materials:		Vec<Material>,
	animations:		Vec<Animation>,
	skins:			Vec<Skin>,
}

impl ModelGraph {
	pub fn new(
        graph: FlatGraph<ModelNodeData>,
        roots: Vec<NodeId>,
        materials: Vec<Material>,
        animations: Vec<Animation>,
        skins: Vec<Skin>,
    ) -> Self {
        Self {
            graph,
            roots,
            materials,
            animations,
            skins,
        }
    }

    #[inline]
    pub fn find_node(&self, name: &str) -> Option<NodeId> {
        self.graph.iter()
            .position(|node| node.value.name == name)
            .map(NodeId)
    }
    #[inline]
    pub fn get_global_matrix_of(&self, id: NodeId) -> Mat4 {
        self.graph.get(id).value.global_transform
    }

    pub fn get_skinning_joint_matrices(&self, skin_id: SkinId) -> Vec<Mat4> {
        let skin = self.get_skin(skin_id);

        skin.joints.iter().enumerate()
            .map(|(i, joint_id)| {
                let global = self.get_global_matrix_of(*joint_id);
                global * skin.inverse_bind_mats[i]
            })
            .collect()
    }

    #[inline]
    pub fn get_skin(&self, skin_id: SkinId) -> &Skin {
        &self.skins[skin_id.0]
    }
    #[inline]
    pub fn get_material(&self, material_id: MaterialId) -> &Material {
        &self.materials[material_id.0]
    }
    #[inline]
    pub fn get_animation(&self, animation_id: AnimationId) -> &Animation {
		&self.animations[animation_id.0]
	}
    #[inline]
    pub fn skins_iter(&self) -> Iter<'_, Skin> {
        self.skins.iter()
    }
    #[inline]
    pub fn skins_iter_mut(&mut self) -> IterMut<'_, Skin> {
        self.skins.iter_mut()
    }
    #[inline]
    pub fn materials_iter(&self) -> Iter<'_, Material> {
        self.materials.iter()
    }
    #[inline]
    pub fn materials_iter_mut(&mut self) -> IterMut<'_, Material> {
        self.materials.iter_mut()
    }
    #[inline]
    pub fn animations_iter(&self) -> Iter<'_, Animation> {
        self.animations.iter()
    }
    #[inline]
    pub fn animations_iter_mut(&mut self) -> IterMut<'_, Animation> {
        self.animations.iter_mut()
    }

    pub fn propagate_transforms(&mut self) {
        self.graph.with_levels(|graph, levels| {
                for level in levels {
                    for &id in level {
                        let local = graph.get(id).get_local_matrix();
                        let global = if let Some(parent) = graph.get(id).parent {
                            graph.get(parent).value.global_transform * local
                        } else {
                            local
                        };
                        graph.get_mut(id).value.global_transform = global;
                    }
                }
            }
        );
    }

    pub fn apply_pose(
        &mut self,
        animation_id: AnimationId,
        current_time: f32
    ) -> Result<()> {
        if self.animations.is_empty() || animation_id.0 >= self.animations.len() {
            return Err(anyhow!("Invalid animation index ({}) > animations lenght ({})", animation_id.0, self.animations.len()));
        }

        let animation = &self.animations[animation_id.0];

        for channel in animation.get_channels_iter() {
            let sampler = animation.get_sampler(channel.sampler_id);

            let keyframe_it = sampler.inputs.partition_point(|&probe_time|
                probe_time < current_time);

            if keyframe_it == 0 || keyframe_it >= sampler.inputs.len() {
                continue;
            } 

            let i = keyframe_it - 1;
            let interp_factor = (current_time - sampler.inputs[i])
                / (sampler.inputs[i + 1] - sampler.inputs[i]);
                
            
            let node = self.graph.get_mut(channel.node_id);

            match channel.path {
                PathType::TRANSLATION => {
                    let start = sampler.outputs_vec3[i];
                    let end = sampler.outputs_vec3[i + 1];
                    node.value.transform.set_translation(start.lerp(end, interp_factor));
                }
                PathType::ROTATION => {
                    let start = vec4_to_quat(sampler.outputs_vec4[i]);
                    let end = vec4_to_quat(sampler.outputs_vec4[i + 1]);
                    node.value.transform.set_rotation(start.slerp(end, interp_factor));
                }
                PathType::SCALE => {
                    let start = sampler.outputs_vec3[i];
                    let end = sampler.outputs_vec3[i + 1];
                    node.value.transform.set_scale(start.lerp(end, interp_factor));
                }
                // TODO: support morph type animation
                PathType::MORPH => warn!("Morph type animation not yet supported."),
            }
        }
        Ok(())
    }
    
    pub fn offset_materials_texture_ids(&mut self, offset: TextureId) {
        for material in &mut self.materials {
            material.offset_texture_ids(offset);
        }
    }

    pub fn get_debug_info(&self) -> Result<()> {
        let mut min_pos = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
        let mut max_pos = Vec3::new(f32::MIN, f32::MIN, f32::MIN);

        
        for (i, node) in self.graph.iter().enumerate() {
            let global_mat = self.get_global_matrix_of(NodeId(i));

            let (translation, rotation, scale) = node.value.transform.get_trs();

            if node.value.name == "Z_UP" {
                debug!("Z_UP transform: t={:?} r={:?} s={:?}", translation, rotation, scale);
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

// endregion