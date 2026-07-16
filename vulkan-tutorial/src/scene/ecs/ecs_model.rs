use anyhow::{Result, anyhow};
use cgmath::{Deg, Rad, Euler};

use bevy_ecs::prelude::*;

use crate::math::{Mat4, Quat, Vec3};
use super::{SkinningBuffer, VulkanDevice, SSBOSkiningAllocator, ModelsStorage, Time};
use super::super::AnimationSpec;

//===============================================
// Component
//===============================================

/// Local transform, what we will edit.
/// 
/// ## Fields
/// 
/// - `position` ( [Vec3] )
/// - `rotation` ( [Quat] )
/// - `scale` ( [Vec3] )
#[repr(C)]
#[derive(Component)]
pub struct Transform {
	pub position:	Vec3,
	pub rotation:	Quat,
	pub scale:		Vec3
}

impl Transform {
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

/// Global Transform -- is the matrix that will place the entity in the world space
/// **compute only at each frame**.
/// Then **READ** when needed by the renderer, collider, etc.
/// 
/// ## Fields
/// 
/// - `field_0` ( [Mat4] ) - World transform.
#[repr(C)]
#[derive(Component)]
pub struct GlobalTransform(pub Mat4);

/// Manage the hierarchie between entities, i.e - arm parent is body.
/// 
/// ## Fields
/// 
/// - `field_0` ( [Entity] ) - Parent entity.
#[derive(Component)]
pub struct Parent(pub Entity);

/// Bridge between an entity index and its geometry to avoid duplication.
/// 
/// ## Fields
/// 
/// - `model_index` ( `usize` ) - Index to retrieve the data from a Vec<[ModelGraph]>.
#[derive(Component)]
pub struct MeshHandle {
    pub model_index: usize
}

/// Manage animation for an entity.
/// 
/// ## Fields
/// 
/// - `skin_index` ( `usize` ) - skin index of the skins vector of a ModelGraph.
/// - `anim_index` ( `usize` ) - animation index of the animations vector of a ModelGraph.
/// - `anim_time` ( `f32` ) - current animation time for this single entity.
/// - `ssbo_offset` ( `u32` ) - ssbo offset of this single entity.
#[derive(Component)]
pub struct SkeletonInstance {
    pub skin_index:     usize,
    pub anim_index:     usize,
    pub anim_time:      f32,
    pub ssbo_offset:    u32
}

//===============================================
// Queries
//===============================================

pub fn propagate_transforms_from_root(
    mut query: Query<(&Transform, &mut GlobalTransform), Without<Parent>>
) {
    for (transform, mut global) in &mut query {
        global.0 = transform.to_model_matrix();
    }
}

// TODO: for now propagate at max one level upward, make it to propagate from all the level upwards
pub fn propagate_transforms_to_children(
    parent_query: Query<&GlobalTransform, Without<Parent>>,
    mut query: Query<(&Transform, &Parent, &mut GlobalTransform)>
) {
    for (transform, parent, mut global) in &mut query {
        if let Ok(parent_global) = parent_query.get(parent.0) {
            global.0 = parent_global.0 * transform.to_model_matrix();
        }
    }
}

pub fn update_skeletons(
	mut query: Query<(&MeshHandle, &mut SkeletonInstance)>,
	models: Res<ModelsStorage>,
	skinning_buffer: Res<SkinningBuffer>,
	device: Res<VulkanDevice>,
	time: Res<Time>,
) -> Result<() >{
	for (mesh_handle, mut skeleton) in &mut query {
		let model = &models.0[mesh_handle.model_index];
		let anim = &model.animations[skeleton.anim_index];

		skeleton.anim_time += time.0;
		while anim.current_time >= anim.end {
			anim.current_time -= anim.end - anim.start;
		}

		//TODO ...
		
		let joint_mat = &model.skins[skeleton.anim_index].get_joint_matrices(graph)?; 

	}

	Ok(())
}

//===============================================
// Spawner
//===============================================

/// Manage the spawn of one model type.
pub struct ModelSpawnBuilder {
	model_index: usize,

	skin_index: Option<usize>,
	anim_index: Option<usize>,
	anim_time: Option<f32>
}

impl ModelSpawnBuilder {
	pub fn new(model_index: usize) -> Self {
		Self {
			model_index,
			skin_index: None,
			anim_index: None,
			anim_time: None
		}
	}

	pub fn with_animation(mut self, anim_specs: AnimationSpec) -> Self {
		(self.skin_index, self.anim_index) = match anim_specs {
			AnimationSpec::Idle { skin_index } => (Some(skin_index), None),
			AnimationSpec::Animated { skin_index, anim_index } => (Some(skin_index), Some(anim_index)),
			AnimationSpec::None => (None, None)
		};
		self
	}

	pub fn animation_start_at(mut self, anim_time: f32) -> Self {
		self.anim_time = Some(anim_time);
		self
	}

	pub fn spawn_at(
		self,
		world: &mut World,
		transform: Transform,
	) -> Result<Entity> {
		let global_transform = GlobalTransform(transform.to_model_matrix());

		// check if the entity have any skeleton
		let skeleton = self.skin_index
			.map(|skin_index| -> Result<SkeletonInstance> {
				let mut ssbo_allocator = world.get_resource_mut::<SSBOSkiningAllocator>()
					.ok_or_else(|| anyhow!("Cannot find a ssbo allocator to spawn model."))?;

				let ssbo_offset = ssbo_allocator.0.allocate()
					.ok_or_else(|| anyhow!("SSBO capacity exceeded - cannot allocatate skinning slot"))?;

				Ok(SkeletonInstance {
					skin_index,
					anim_index: self.anim_index.unwrap_or(0),
					anim_time: self.anim_time.unwrap_or(0.0),
					ssbo_offset
				})
			})
			.transpose()?;

		let mut entity = world.spawn(
			(
				transform,
				global_transform,
				MeshHandle { model_index: self.model_index },
			)
		);

		if let Some(skeleton) = skeleton {
			entity.insert(skeleton);
		}

		Ok(entity.id())
	}
}

