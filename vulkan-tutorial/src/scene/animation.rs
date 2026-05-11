use std::{cell::RefCell, rc::Weak};

use super::Node;

use crate::math::*;

pub enum PathType {
    TRANSLATION,
    ROTATION,
    SCALE,
}

pub enum InterpolationType {
    LINEAR,
    STEP,
    CUBICSPLINE,
}

/// Structure for animation key-frames.
/// 
/// ## Fields
/// 
/// - `path` ( [PathType] ) - type of key-frame animation.
/// - `node` ( Weak<RefCell<[Node]>> ) - reference to a model node (from a scene-graph) to animate.
/// - `sampler_index` ( usize ) - index inside a [AnimationSampler] table.
pub struct AnimationChannel {
    pub path: PathType,
    pub node: Weak<RefCell<Node>>,
    pub sampler_index: usize,
}


/// Structure for animation interpolation.
/// 
/// ## Fields
/// 
/// - `interpolation_type` ( [InterpolationType] ).
/// - `inputs` ( Vec\<f32> ) - Key frame timestamps.
/// - `outputsVec4` ( Vec\<Vec4> ) - Key frame values (for rotations).
/// - `outputsVec3` ( Vec\<Vec3> ) - Key frame values (for translations and scales).
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
pub struct Animation {
    pub name: String,
    pub samplers: Vec<AnimationSampler>,
    pub channels: Vec<AnimationChannel>,
    pub start: f32,
    pub end: f32,
    pub current_time: f32,
}

impl Default for Animation {
    fn default() -> Self {
        Self {
            name: String::new(),
            samplers: Vec::new(),
            channels: Vec::new(),
            start: f32::MIN,
            end: f32::MAX,
            current_time: 0.0
        }
    }
}
