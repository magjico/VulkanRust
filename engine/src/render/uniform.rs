use std::mem::offset_of;

use crate::math::*;
use crate::scene::{Material, Camera};
use crate::constants::CORRECTION;

/// Contained instance object data
/// 
/// ## Fields
/// 
/// - `model` ( [Mat4] ).
/// - `ssbo_offset` ( `u32` ).
/// - `_padding` (`[u32; 3]`) - `std430` requires to align the structure to 16 bytes.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct InstanceData {
    pub model: Mat4,
    pub ssbo_offset: u32,
    pub _padding: [u32; 3],
}

/// UBO to pass to the shaders
/// 
/// ## Fields
/// 
/// - `view` (`Mat4`) - View matrix.
/// - `proj` (`Mat4`) - Proj matrix.
/// - `light_positions` (`[Vec4; 4]`) - Position and radius (xyz = position, w = radius).
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

impl UniformBufferObject {
    // TODO: pass all raw argument into a struct RenderParams
    pub fn from_camera(
        camera: &Camera,
        aspect_ratio: f32
    ) -> Self {
        let view = camera.get_view_matrix();

        let proj = CORRECTION * camera.get_projection_matrix(
            aspect_ratio,
            Some(0.1),
            Some(1000.0)
        );

        let cam_pos = {
            let pos = camera.get_position();
            Vec4::new(pos.x, pos.y, pos.z, 1.0)
        };

        UniformBufferObject {
            view,
            proj,
            cam_pos,
            exposure: 4.5,
            gamma: 2.2,
            prefiltered_cube_mip_levels: 1.0,
            scale_ibl_ambient: 1.0
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct PushConstants {
    // vertex shader push constants
    pub node: Mat4,                             // 64	bytes	->	 64 / 128

    // fragment shader push constants
    pub base_color_factor: Vec4,               	// 16   bytes   ->   80 / 128
    pub emissive_factor: Vec3,                  // 12   bytes   ->   92 / 128
    pub metallic_factor: f32,					//  4	bytes	->	 96 / 128
    pub roughness_factor: f32,					//  4	bytes	->	100 / 128
	pub base_color_texture_set: i32,			//  4	bytes	->  104 / 128
	pub physical_descriptor_texture_set: i32,	//  4	bytes	->	108 / 128
	pub normal_texture_set: i32,				//  4	bytes	->	112 / 128
	pub occlusion_texture_set: i32,				//  4	bytes	->	116 / 128
	pub emissive_texture_set: i32,				//  4	bytes	->	120 / 128
	pub alpha_mask: f32,						//  4	bytes	->	124	/ 128
	pub alpha_mask_cutoff: f32,					//  4	bytes	->	128	/ 128

	// TOTAL: 128 bytes out of 128 bytes used.
}

impl PushConstants {
    pub const NO_SKIN: u32 = u32::MAX;

    pub fn new(
        node_matrix: Mat4,
        material: Option<&Material>
    ) -> Self {
        Self {
            node: node_matrix,

            base_color_factor:                  material.map(|m| m.base_color_factor).unwrap_or(Vec4::new(1.0, 1.0, 1.0, 1.0)),
            emissive_factor:                    material.map(|m| m.emissive_factor).unwrap_or(Vec3::new(0.0, 0.0, 0.0)),
            metallic_factor:                    material.map(|m| m.metallic_factor).unwrap_or(1.0),
            roughness_factor:                   material.map(|m| m.roughness_factor).unwrap_or(1.0),
            base_color_texture_set:             material.map(|m| m.base_color_texture_set).unwrap_or(-1),
            physical_descriptor_texture_set:    material.map(|m| m.metallic_roughness_texture_set).unwrap_or(-1),
            normal_texture_set:                 material.map(|m| m.normal_texture_set).unwrap_or(-1),
            occlusion_texture_set:              material.map(|m| m.occlusion_texture_set).unwrap_or(-1),
            emissive_texture_set:               material.map(|m| m.emissive_texture_set).unwrap_or(-1),

            alpha_mask:                         material.map(|m| m.alpha_mask).unwrap_or(0.0),
            alpha_mask_cutoff:                  material.map(|m| m.alpha_mask_cutoff).unwrap_or(0.5)
        }
    }

	#[inline]
	pub fn get_frag_offset() -> u32 { offset_of!(PushConstants, base_color_factor) as u32 }

    #[inline]
    pub fn get_frag_size() -> u32 { size_of::<PushConstants>() as u32 - Self::get_frag_offset() }
}