/// Scene-graph implementation
use std::{cell::RefCell, rc::{Weak, Rc}};
use log::*;

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
