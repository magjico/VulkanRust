// TODO: implement a generic tray for all io / so we need to implement load and save in a generic way.
use std::fs::File;
use std::io::BufReader;
use std::collections::HashMap;
use anyhow::{Result, anyhow};
use png::Decoder;
use log::*;

use cgmath::{vec2, vec3};

use gltf::{buffer::Data, image::Source, iter};
use basis_universal::{TranscodeParameters, Transcoder, TranscoderTextureFormat};
use vulkanalia::prelude::v1_0::*;

use crate::math::{Vec3, Vec4};
use crate::render::{create_texture_image, create_texture_image_view, create_texture_sampler};
use crate::scene::{Material, Mesh, ModelGraph, Vertex};


//===============================================
// texture file
//===============================================

/// Create a texture image from a .png.
///
/// *__TODO:__ add the possibility to manually select the mipmaps level.*
/// 
/// ## Arguments
/// 
/// - `instance` (&[`Instance`]) - Vulkan instance.
/// - `device` (&[`Device`]) - Vulkan device.
/// - `physical_device` ([`vk::PhysicalDevice`]) - Used physical device.
/// - `texture_path` (`&str`) - path to the .png texture file.
/// - `setup_command_buffer` ([`vk::CommandBuffer`]) - A command buffer to execute command from.
/// - `graphics_queue` ([`vk::Queue`]) - A graphics queue to the send the command to the gpu.
/// 
/// ## Returns
/// 
/// - `Result<(vk::Image, vk::DeviceMemory, u32)>` - The texture image, its device memory and its mip maps levels.
pub fn load_texture(
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    texture_path: &str,
    setup_command_buffer: vk::CommandBuffer,
    graphics_queue: vk::Queue,
) -> Result<(vk::Image, vk::DeviceMemory, u32)> {
    let image = File::open(texture_path)?;

	let io_buf = BufReader::new(image); // IO Buffer
	let decoder = Decoder::new(io_buf); // PNG Decoder
	let mut reader = decoder.read_info()?; // PNG reader

	let size = reader.info().raw_bytes() as u64;
	let (width, height) = reader.info().size();

	let mut pixels = vec![0; size as usize];
	reader.next_frame(&mut pixels)?;

    let mip_levels = (width.max(height) as f32).log2().floor() as u32 + 1;

    let (texture_image, texture_image_memory) = create_texture_image(
        instance,
        device,
        physical_device,
        setup_command_buffer,
        graphics_queue,
        vk::Extent3D {width, height, depth: 1}, 
        mip_levels,
        &pixels,
        vk::Format::R8G8B8A8_SRGB,
    )?;

    Ok((texture_image, texture_image_memory, mip_levels))

}

//===============================================
// .obj
//===============================================

/// Load a .obj 3D model and return its vertices and the associated indexes.
/// 
/// # Arguments
/// 
/// - `path` (`&str`) - path to the .obj
/// 
/// # Returns
/// 
/// - `Result<(Vec<Vertex>, Vec<u32>)>` - vertices and indices.
pub fn load_obj_model(path: &str) -> Result<Mesh> {
    // Model
    let mut reader = BufReader::new(File::open(path)?);

    let (models, _) = tobj::load_obj_buf(
        &mut reader,
        &tobj::LoadOptions {
            triangulate: true,
            single_index: true,
            ..Default::default()
        },
        |_| Ok(Default::default()),
    )?;

    // Vertices / Indices
    let mut unique_vertices = HashMap::new();
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for model in &models {
        for index in &model.mesh.indices {
            let pos_offset = (3 * index) as usize;
            let tex_coord_offset = (2 * index) as usize;

            let vertex = Vertex {
                pos: vec3(
                    model.mesh.positions[pos_offset],
                    model.mesh.positions[pos_offset + 1],
                    model.mesh.positions[pos_offset + 2],
                ),
                normal: if model.mesh.normals.is_empty() {
                    vec3(0.0, 0.0, 0.0)
                } else {
                    vec3(
                        model.mesh.normals[pos_offset],
                        model.mesh.normals[pos_offset + 1],
                        model.mesh.normals[pos_offset + 2],
                    )
                },
                color: vec3(1.0, 1.0, 1.0),
                tex_coord: vec2(
                    model.mesh.texcoords[tex_coord_offset],
                    1.0 - model.mesh.texcoords[tex_coord_offset + 1],
                ),
            };

            if let Some(index) = unique_vertices.get(&vertex) {
                indices.push(*index as u32);
            } else {
                let index = vertices.len();
                unique_vertices.insert(vertex, index);
                vertices.push(vertex);
                indices.push(index as u32);
            }
        }
    }

    Ok(Mesh {vertices, indices, material_index: -1})
}

//===============================================
// glTF
//===============================================

fn load_gltf_textures(
    device: &Device,
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    textures: iter::Textures,
    setup_command_buffer: vk::CommandBuffer,
    graphics_queue: vk::Queue,
    buffers: &[Data],
) -> Result<Vec<(vk::Image, vk::DeviceMemory, vk::ImageView, vk::Sampler)>> {

    let mut vk_textures = Vec::new();
    'texture: for (i, texture) in textures.enumerate() {
        let image = texture.source();

        // let name = image.name()
        //     .filter(|n| !n.is_empty())
        //     .map(|n| n.to_string())
        //     .unwrap_or_else(|| format!("texture_{}", i));

        // 1 - check if the image is embedded as KTX2
        let ktx2_data: Vec<u8> = match image.source() {
            Source::View { view, mime_type } if mime_type == "image/ktx2" => {
                    let buffer_data = &buffers[view.buffer().index()];
                    let start = view.offset();
                    let end = start + view.length();
                    buffer_data[start..end].to_vec()
            }
            Source::Uri { uri, mime_type }
                if mime_type.unwrap_or("") == "image/ktx2" || uri.ends_with(".ktx2") => {
                    std::fs::read(uri)?
            }
            _ => {
                warn!("Texture {} is not KTX2, skipping", i);
                continue 'texture;
            }
        };

        // 2 - create texture image
        let ktx_texture = ktx2::Reader::new(&ktx2_data)?;
        let header = ktx_texture.header();

        let format = header.format
            .map(|f| vk::Format::from_raw(f.value() as i32))
            .ok_or_else(|| anyhow!("KTX2 texture {} as no format", i))?;

        let extent = vk::Extent3D {
            width: header.pixel_width,
            height: header.pixel_height,
            depth: header.pixel_depth.max(1),
        };
        let mip_levels = header.level_count;

        let pixel_data = if header.supercompression_scheme == Some(ktx2::SupercompressionScheme::BasisLZ) {
            // transcode texture
            let mut transcoder = Transcoder::new();

            if transcoder.prepare_transcoding(&ktx2_data).is_err() {
                warn!("Failed to prepare transcoding: {}", i);
                continue 'texture;
            }

            // let transcode_format = TranscoderTextureFormat::BC7_RGBA;

            let mut all_levels_data = Vec::new();
            for level in 0..mip_levels {
                let params = TranscodeParameters {
                    image_index: 0,
                    level_index: level as u32,
                    decode_flags: None,
                    output_row_pitch_in_blocks_or_pixels: None,
                    output_rows_in_pixels: None,
                };

                // TODO: support multiple textures format transcoder in case the gpu doesn't support BC7.
                match transcoder.transcode_image_level(
                    &ktx2_data,
                    TranscoderTextureFormat::BC7_RGBA,
                    params,
                ) {
                    Ok(data) => all_levels_data.extend_from_slice(&data),
                    Err(e) => {
                        warn!("Failed to transcode level {}: {:?}", level, e);
                        continue 'texture;
                    }
                }
            }
            all_levels_data
        } else {
            // no need for transcoding
            let mut all_levels_data = Vec::new();
            for level in ktx_texture.levels() {
                all_levels_data.extend_from_slice(&level.data);
            }
            all_levels_data
        };

        // create Vulkan Image, Memory and View
        let (texture_image, texture_image_memory) = create_texture_image(
            instance,
            device,
            physical_device,
            setup_command_buffer,
            graphics_queue,
            extent,
            mip_levels,
            &pixel_data,
            format,
        )?;
        let texture_image_view = create_texture_image_view(&device, texture_image, mip_levels)?;
        let texture_sampler = create_texture_sampler(&device, mip_levels as f32)?;
                
        vk_textures.push((texture_image, texture_image_memory, texture_image_view, texture_sampler));
    }

    Ok(vk_textures)
}

fn load_gltf_materials(
    materials: iter::Materials,
) -> Result<Vec<Material>> {

    let materials: Vec<Material> = materials.map(|material| {
        let pbr = material.pbr_metallic_roughness();
        let base_color_factor = pbr.base_color_factor();

        Material {
            base_color_factor: Vec4::new(
                base_color_factor[0],
                base_color_factor[1],
                base_color_factor[2],
                base_color_factor[3]
            ),
            metallic_factor: pbr.metallic_factor(),
            roughness_factor: pbr.roughness_factor(),
            emissive_factor: {
                let e = material.emissive_factor();
                Vec3::new(e[0], e[1], e[2])
            },

            base_color_texture_idx: pbr.base_color_texture()
                .map(|info| info.texture().source().index() as i32)
                .unwrap_or(-1),
            metallic_roughness_texture_idx: pbr.metallic_roughness_texture()
                .map(|info| info.texture().source().index() as i32)
                .unwrap_or(-1),
            normal_texture_idx: material.normal_texture()
                .map(|text| text.texture().source().index() as i32)
                .unwrap_or(-1),
            occlusion_texture_idx: material.occlusion_texture()
                .map(|text| text.texture().source().index() as i32)
                .unwrap_or(-1),
            emissive_texture_idx: material.emissive_texture()
                .map(|text| text.texture().source().index() as i32)
                .unwrap_or(-1),
        }
    }).collect();

    Ok(materials)
}
 
pub fn load_gltf_model(
    device: &Device,
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    path: &str,
    setup_command_buffer: vk::CommandBuffer,
    graphics_queue: vk::Queue,
) -> Result<()> {
    // warn: this method does not support reading gltf / glb from web.
    let (document, buffers, images) = gltf::import(path)?;

    // 1 - load textures
    let textures = load_gltf_textures(
        device,
        instance,
        physical_device,
        document.textures(),
        setup_command_buffer,
        graphics_queue,
        &buffers
    )?;

    // 2 - load materials
    let materials = load_gltf_materials(document.materials())?;




    Ok(())
}