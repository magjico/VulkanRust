use anyhow::{Result, anyhow};
use std::slice::Iter;

use crate::math::*;
use crate::type_safety::{SamplerId, NodeId};

#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub enum PathType {
    TRANSLATION,
    ROTATION,
    SCALE,
    MORPH,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub enum InterpolationType {
    LINEAR,
    STEP,
    CUBICSPLINE,
}

/// Describe the animated state of a Model.
/// 
/// ## Variants
/// 
/// - `None` - No skin and no animation.
/// - `Idle { skin_index }` - Skin but no animation.
/// - `Animated { skin_index, anim_index }` - Skin and animation.
pub enum AnimationSpec {
	None,
	Idle { skin_index: usize },
	Animated { skin_index: usize, anim_index: usize },
}


/// Structure for animation key-frames.
/// 
/// ## Fields
/// 
/// - `path` ( [PathType] ) - type of key-frame animation.
/// - `node_id` ( [NodeId] ) - index-reference to a model node (from a scene-graph) to animate.
/// - `sampler_index` ( [SamplerId] ) - index inside a [AnimationSampler] table.
#[derive(Clone, Debug)]
pub struct AnimationChannel {
    pub path: PathType,
    pub node_id: NodeId,
    pub sampler_id: SamplerId,
}

/// Structure for animation interpolation.
/// 
/// ## Fields
/// 
/// - `interpolation_type` ( [InterpolationType] ).
/// - `inputs` ( Vec\<f32> ) - Key frame timestamps.
/// - `outputsVec4` ( Vec\<Vec4> ) - Key frame values (for rotations).
/// - `outputsVec3` ( Vec\<Vec3> ) - Key frame values (for translations and scales).
#[derive(Clone, Debug)]
pub struct AnimationSampler {
    pub interpolation_type: InterpolationType,
    pub inputs: Vec<f32>,
    pub outputs_vec4: Vec<Vec4>,
    pub outputs_vec3: Vec<Vec3>,
}

/// Structure for animation
/// 
/// ## Fields
/// 
/// - `name` ( String ) - Animation name.
/// - `samplers` ( Vec<[AnimationSampler]> ) - animation samples.
/// - `channels` ( Vec<[AnimationChannel]> ) - animation channels.
/// - `start` ( f32 ) - animation start time value.
/// - `end` ( f32 ) - animation end time value.
/// - `current_time` ( f32 ) - current animation time value.
#[derive(Clone, Debug)]
pub struct Animation {
    name: String,
    samplers: Vec<AnimationSampler>,
    channels: Vec<AnimationChannel>,
    start: f32,
    end: f32,
}

impl Animation {
    pub fn new(
        name: String,
        samplers: Vec<AnimationSampler>,
        channels: Vec<AnimationChannel>,
        start: f32,
        end: f32,
    ) -> Self {
        Self { name, samplers, channels, start, end }
    }

    pub fn get_sampler(&self, id: SamplerId) -> &AnimationSampler {
        self.samplers.get(id.0)
            .unwrap_or_else(|| panic!("animation sampler id ({}) out of bounds for length {}", id.0, self.samplers.len()))
    }

    #[inline]
    pub fn get_channels_iter(&self) -> Iter<'_, AnimationChannel>{
        self.channels.iter()
    }
    #[inline]
    pub fn get_name(&self) -> &String { &self.name }
    #[inline]
    pub fn get_start(&self) -> f32 { self.start }
    #[inline]
    pub fn get_end(&self) -> f32 { self.end }
    #[inline]
    pub fn builder(start_time: f32, end_time: f32) -> AnimationBuilder {
        AnimationBuilder::new(start_time, end_time)
    }
}

#[derive(Clone, Debug)]
pub struct AnimationBuilder {
    pub name: String,
    pub samplers: Vec<AnimationSampler>,
    pub channels: Vec<AnimationChannel>,
    pub start: f32,
    pub end: f32,
}

impl AnimationBuilder {
    pub fn new(start_time: f32, end_time: f32) -> Self {
        Self {
            name: String::new(),
            samplers: Vec::new(),
            channels: Vec::new(),
            start: start_time,
            end: end_time,
        }
    }

    pub fn name(mut self, name: String) -> Self {
        self.name = name;
        self
    }

    pub fn samplers(mut self, samplers: Vec<AnimationSampler>) -> Self {
        self.samplers = samplers;
        self
    }

    pub fn channels(mut self, channels: Vec<AnimationChannel>) -> Self {
        self.channels = channels;
        self
    }

    pub fn anim_start_time(mut self, start: f32) -> Self {
        self.start = start;
        self
    }

    pub fn anim_end_time(mut self, end: f32) -> Self {
        self.end = end;
        self
    }

    pub fn build(self) -> Result<Animation> {
        if self.start >= self.end {
            return Err(anyhow!("Animation '{}' - start ({}) must be < end ({})", self.name, self.start, self.end));
        }

        for ch in &self.channels {
            if ch.sampler_id.0 >= self.samplers.len() {
                return Err(
                    anyhow!("Animation '{}' - channel sampler_id {} out of bounds for samplers length {}",
                    self.name, ch.sampler_id.0, self.samplers.len())
                );
            }
        }

        Ok( Animation::new(
            self.name,
            self.samplers,
            self.channels,
            self.start,
            self.end,
        ))
    } 
}