use bevy_ecs::prelude::*;

use super::{GlobalTransform, MeshHandle, SkeletonInstance};

pub struct ECSContext {
	pub world: World,
	pub schedule: Schedule,
}

impl ECSContext {
	#[inline]
	pub fn init() -> Self {
		Self {
			world: World::new(),
			schedule: Schedule::default()
		}
	}

	#[inline]
	pub fn get_renderable_query<'a>(&mut self) -> QueryState<(&'a GlobalTransform, &'a MeshHandle, Option<&'a SkeletonInstance>)> {
		self.world.query::<(&'a GlobalTransform, &'a MeshHandle, Option<&'a SkeletonInstance>)>()
	}
}