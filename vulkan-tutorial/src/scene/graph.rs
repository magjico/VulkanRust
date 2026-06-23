/// Scene-graph implementation
use std::cell::RefCell;
use std::rc::{Weak, Rc};
use std::ptr::copy_nonoverlapping as memcpy;
use anyhow::{Result, anyhow};

use vulkanalia::prelude::v1_0::*;

use cgmath::SquareMatrix;

use super::Mesh;
use crate::{math::*, scene::ModelInstance};

//===============================================
// Scene-Graph
//===============================================

/// Structure for a node in a scene-graph
#[repr(C)]
#[derive(Clone, Debug)]
pub struct Node {
    pub name: String,

    pub parent: Option<Weak<RefCell<Node>>>,
    pub childs: Vec<Rc<RefCell<Node>>>,

    pub mesh: Option<Mesh>,
    
    // For animation
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,

    pub skin: i32,

    pub homogeneous_matrix: Mat4,
}

impl Default for Node {
    fn default() -> Self {
        Self {
            name: String::new(),
            parent: None,
            childs: Vec::new(),
            mesh: None,
            translation: Vec3::new(0.0, 0.0, 0.0),
            rotation: Quat::new(1.0, 0.0, 0.0, 0.0),
            scale: Vec3::new(1.0, 1.0, 1.0),
            skin: -1,
            homogeneous_matrix: Mat4::identity(),
        }
    }
}

impl Node {
    pub fn get_local_matrix(&self) -> Mat4 {
        // debug!("model node info:\n- translation: {:?}\n- rotation: {:?}\n- scale: {:?}", self.translation, self.rotation, self.scale);

        let translation = Mat4::from_translation(self.translation);
        let rotation = Mat4::from(self.rotation); 
        let scale = Mat4::from_nonuniform_scale(
            self.scale.x,
            self.scale.y,
            self.scale.z,
        );

        translation * rotation * scale * self.homogeneous_matrix
    }

    pub fn get_global_matrix(&self) -> Mat4 {
        let mut global = self.get_local_matrix();
        let mut current = self.parent.clone();

        while let Some(weak_ref) = current {
            if let Some(parent) = weak_ref.upgrade() {
                let parent = parent.borrow();
                global = parent.get_local_matrix() * global;
                current = parent.parent.clone();
            } else {
                break
            }
        }

        global
    }

    pub fn to_instance(&self) -> ModelInstance {
        ModelInstance::new(self.translation, self.rotation, self.scale)
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
/// - `skeleton_root` ( __Weak<RefCell<[Node]>>__ ) - Useful for certain type of optimisations or to place a skeleton in one go.
/// - `inverse_bind_mats` ( __Vec<[Mat4]>__ ) - Transforms the geometry into the space of the respective joint.
/// - `joints` ( __Vec<Weak<RefCell<[Node]>>>__ ) - Contains the nodes used as joints in this skin.
/// - `ssbo_buffers` ( __Vec<[vk::Buffer]>__ ) - Contains the buffers (**one for each frame in flight**) that hold the calculated joint matrices,
/// 	ready to be passed to the shader.
/// - `ssbo_memories` ( __Vec<[vk::DeviceMemory]>__) - Contains the buffers memories.
/// - `descriptor_set` ( __Vec<[vk::DescriptorSet]>__ ) - Contains descriptor sets that point to each buffers.
#[repr(C)]
#[derive(Clone, Debug)]
pub struct Skin {
    pub name: String,
    pub skeleton_root: Option<Weak<RefCell<Node>>>,
    pub inverse_bind_mats: Vec<Mat4>,
    pub joints: Vec<Weak<RefCell<Node>>>,

    pub ssbo_buffers: Vec<vk::Buffer>,
	pub ssbo_memories: Vec<vk::DeviceMemory>,
	pub descriptor_sets: Vec<vk::DescriptorSet>,

    pub joint_matrices: Vec<Mat4> // CPU cache for joint matrices
}

// TODO: add a skin::destroy()
impl Skin {
	pub fn get_joint_matrices(&self) -> Result<Vec<Mat4>> {
        self.joints.iter().enumerate()
            .map(|(i, w_joint)| {
                let joint = w_joint.upgrade()
                    .ok_or_else(|| anyhow!("joint {} was dropped", i))?;
                let global = joint.borrow().get_global_matrix();
                Ok(global * self.inverse_bind_mats[i])
            })
            .collect()
    }

    /// Need to be called after an update on the joint_matrices.
    pub fn update_ssbo(&mut self, device: &Device, image_index: usize) -> Result<()> {
        self.joint_matrices = self.get_joint_matrices()?;
        let size = (self.joint_matrices.len() * size_of::<Mat4>()) as vk::DeviceSize;

        unsafe {
            let ptr = device.map_memory(
                self.ssbo_memories[image_index],
                0,
                size,
                vk::MemoryMapFlags::empty()
            )?;

            memcpy(self.joint_matrices.as_ptr(), ptr.cast(), self.joint_matrices.len());
            device.unmap_memory(self.ssbo_memories[image_index]);
        }

        Ok(())
    }

    #[rustfmt::skip]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn destroy(&self, device: &Device) {
        self.ssbo_buffers.iter().for_each(|buf| device.destroy_buffer(*buf, None));
        self.ssbo_memories.iter().for_each(|mem| device.free_memory(*mem, None));
    }
}