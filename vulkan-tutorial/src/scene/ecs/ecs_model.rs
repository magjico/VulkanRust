use anyhow::{Result, anyhow};
use cgmath::{Deg, Rad, Euler};
use log::*;

use bevy_ecs::prelude::*;

use crate::math::{Mat4, Quat, Vec3};
use crate::type_safety::{SkinId, AnimationId, ModelId};
use super::{SkinningBuffer, SSBOSkiningAllocator, ModelsStorage, Time};
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
/// - `model_id` ( [ModelId] ) - Index to retrieve the data from a Vec<[ModelGraph]>.
#[derive(Component)]
pub struct MeshHandle {
    pub model_id: ModelId
}

/// Manage animation for an entity.
/// 
/// ## Fields
/// 
/// - `skin_id` ( [SkinId] ) - skin index of the skins vector of a ModelGraph.
/// - `anim_id` ( [AnimationId] ) - animation index of the animations vector of a ModelGraph.
/// - `anim_time` ( `f32` ) - current animation time for this single entity.
/// - `ssbo_offset` ( `u32` ) - ssbo offset of this single entity.
#[derive(Component)]
pub struct SkeletonInstance {
    pub skin_id:		SkinId,
    pub anim_id:		AnimationId,
    pub anim_time:		f32,
    pub ssbo_offset:	u32
}

//===============================================
// Queries
//===============================================

// --- animation --- //

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
	mut models: ResMut<ModelsStorage>,
	skinning_buffer: Res<SkinningBuffer>,
	time: Res<Time>,
) -> Result<() >{
	for (mesh_handle, mut skeleton) in &mut query {
		let model = models.get_mut_model(mesh_handle.model_id);
		let anim = model.get_animation(skeleton.anim_id);

		skeleton.anim_time += time.0;
		while skeleton.anim_time >= anim.get_end() {
			skeleton.anim_time -= anim.get_end() - anim.get_start();
		}

		model.apply_pose(skeleton.anim_id, skeleton.anim_time)?;
		
		let joint_mats = model.get_skinning_joint_matrices(skeleton.skin_id);
		skinning_buffer.write_slice(skeleton.ssbo_offset, &joint_mats)?;
	}

	Ok(())
}

/// Same as [update_skeletons] but that can be pass to a [Schedule] with [Schedule::add_systems].
pub fn update_skeletons_wrapped(
    query: Query<(&MeshHandle, &mut SkeletonInstance)>,
    models: ResMut<ModelsStorage>,
    skinning_buffer: Res<SkinningBuffer>,
    time: Res<Time>,
) {
    if let Err(e) = update_skeletons(query, models, skinning_buffer, time) {
        warn!("update_skeletons failed: {}", e);
    }
}

//===============================================
// Spawner
//===============================================

/// Manage the spawn of one model type.
pub struct ModelSpawnBuilder {
	model_id: ModelId,

	skin_id: Option<SkinId>,
	anim_id: Option<AnimationId>,
	anim_time: Option<f32>
}

impl ModelSpawnBuilder {
	pub fn new(model_id: ModelId) -> Self {
		Self {
			model_id,
			skin_id: None,
			anim_id: None,
			anim_time: None
		}
	}

	pub fn with_animation(mut self, anim_specs: AnimationSpec) -> Self {
		(self.skin_id, self.anim_id) = match anim_specs {
			AnimationSpec::Idle { skin_id } => (Some(skin_id), None),
			AnimationSpec::Animated { skin_id, anim_id } => (Some(skin_id), Some(anim_id)),
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
		let skeleton = self.skin_id
			.map(|skin_id| -> Result<SkeletonInstance> {
				let mut ssbo_allocator = world.get_resource_mut::<SSBOSkiningAllocator>()
					.ok_or_else(|| anyhow!("Cannot find a ssbo allocator to spawn model."))?;

				let ssbo_offset = ssbo_allocator.0.allocate()
					.ok_or_else(|| anyhow!("SSBO capacity exceeded - cannot allocatate skinning slot"))?;

				Ok(SkeletonInstance {
					skin_id:	skin_id,
					anim_id:	self.anim_id.unwrap_or(AnimationId(0)),
					anim_time:	self.anim_time.unwrap_or(0.0),
					ssbo_offset
				})
			})
			.transpose()?;

		let mut entity = world.spawn(
			(
				transform,
				global_transform,
				MeshHandle { model_id: self.model_id },
			)
		);

		if let Some(skeleton) = skeleton {
			entity.insert(skeleton);
		}

		Ok(entity.id())
	}
}

