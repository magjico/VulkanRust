use std::mem::offset_of;

use crate::math::*;

/// UBO to pass to the shaders
/// 
/// # Fields
/// 
/// - `view` (`Mat4`) - View matrix.
/// - `proj` (`Mat4`) - Proj matrix.
/// - `light_positions` (`[Vec4; 4]`) - Position and radius	(xyz = position, w = radius).
/// - `light_colors` (`[Vec4; 4]`) - RGB color and intensity (xyz = rgb, w = intensity).
/// - `cam_pos` (`Vec4`) - Camera position for view-dependent effects.
/// - `exposure` (`f32`) - Exposure for HDR rendering.
/// - `gamme` (`f32`) - Gamma correction value.
/// - `prefiltered_cube_mip_levels` (`f32`) - For image-based lighting.
/// - `scale_ibl_ambient` (`f32`) - Scale factor for ambient lighting.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct UniformBufferObject {
    pub view: Mat4,
    pub proj: Mat4,
    
    // PBR parameters
    pub cam_pos: Vec4,
    pub exposure: f32,
    pub gamma: f32,
    pub prefiltered_cube_mip_levels: f32,
    pub scale_ibl_ambient: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct PushConstants {
    // vertex shader push constants
    pub model: Mat4,							// 64	bytes	->	 64 / 128
    pub skin_count: i32,							//  4	bytes	->	 68 / 128

    // fragment shader push constants
    pub metallic_factor: f32,					//  4	bytes	->	 72 / 128
	pub roughness_factor: f32,					//  4	bytes	->	 76 / 128
	pub _padding: f32,							//  4	bytes	->	 80 / 128
	pub base_color_factor: Vec4,				// 16	bytes	->	 96 / 128
	pub base_color_texture_set: i32,			//  4	bytes	->	100 / 128
	pub physical_descriptor_texture_set: i32,	//  4	bytes	->	104 / 128
	pub normal_texture_set: i32,				//  4	bytes	->	108 / 128
	pub occlusion_texture_set: i32,				//  4	bytes	->	112 / 128
	pub emissive_texture_set: i32,				//  4	bytes	->	116 / 128
	pub alpha_mask: f32,						//  4	bytes	->	120	/ 128
	pub alpha_mask_cutoff: f32,					//  4	bytes	->	124	/ 128

	// TOTAL: 124 bytes out of 128 bytes used.
}

impl PushConstants {
	#[inline]
	pub fn get_frag_offset() -> u32 { offset_of!(PushConstants, metallic_factor) as u32 }
}