use bevy_ecs::prelude::*;

use super::{GlobalTransform, MeshHandle, SkeletonInstance};

pub struct ECSContext {
	pub world: World,
	pub schedule: Schedule,

	// cached for opti
	pub cached_renderable_query: QueryState<(&'static GlobalTransform, &'static MeshHandle, Option<&'static SkeletonInstance>)>,
}

impl ECSContext {
	pub fn init() -> Self {
		let mut world = World::new();
		let schedule = Schedule::default();
		let cached_renderable_query = world.query::<(
			&GlobalTransform,
			&MeshHandle,
			Option<&SkeletonInstance>,
		)>();

		Self {
			world,
			schedule,
			cached_renderable_query
		}
	}
}