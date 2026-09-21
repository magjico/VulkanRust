// TODO: implement a generic tray for all io / so we need to implement load and save in a generic way.
use std::fs::File;
use std::io::BufReader;
use std::collections::HashMap;
use std::path::Path;

use log::*;
use anyhow::{Result, anyhow};
use cgmath::{SquareMatrix, vec2, vec3};

use basis_universal::{TranscodeParameters, Transcoder, TranscoderTextureFormat};
use png::Decoder;
use image;

use gltf::iter;
use gltf::buffer::Data;
use gltf::image::Source;
use gltf::accessor::Dimensions;
use gltf::animation::{Interpolation, Property};

use vulkanalia::prelude::v1_0::*;

use crate::ops::{FlatGraph, Node};
use crate::type_safety::{NodeId, MaterialId, TextureId, SamplerId};
use crate::math::*;
use crate::render::*;
use crate::scene::*;

// region load texture file
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
// endregion

// region load obj
/// Load a .obj 3D model and return its vertices and the associated indexes.
/// 
/// ## Arguments
/// 
/// - `path` (`&str`) - path to the .obj
/// 
/// ## Returns
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
                uv0: vec2(
                    model.mesh.texcoords[tex_coord_offset],
                    1.0 - model.mesh.texcoords[tex_coord_offset + 1],
                ),
                uv1: vec2(0.0, 0.0),
                tangent: Vec4::new(1.0, 0.0, 0.0, 1.0),
                joint_indices: UVec4::new(0, 0, 0, 0),
                joint_weights: Vec4::new(1.0, 0.0, 0.0, 0.0),
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

    Ok(Mesh {vertices, indices, ..Default::default()})
}

pub fn load_3d_content(
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    obj_path: &str,
    texture_path: &str,
    setup_command_buffer: vk::CommandBuffer,
    graphics_queue: vk::Queue,
) -> Result<(Mesh, TextureData)> {
    let mesh = load_obj_model(obj_path)?;
    let (texture_image, texture_image_memory, mip_levels) = load_texture(
        instance,
        device,
        physical_device,
        texture_path,
        setup_command_buffer,
        graphics_queue
    )?;

    let texture_image_view = create_texture_image_view(device, texture_image, mip_levels)?;

    let texture = TextureData {
        image: texture_image,
        image_memory: texture_image_memory,
        image_view: texture_image_view,
    };

    Ok((mesh, texture))
}
// endregion

// region load glTF
/// check if the texture mimetype is ktx2
fn is_ktx2(image_src: &Source) -> bool {
    match image_src {
        Source::View { mime_type, .. } => *mime_type == "image/ktx2",
        Source::Uri { uri, mime_type } => mime_type.unwrap_or("") == "image/ktx2" || uri.ends_with(".ktx2")
    }
}

/// check if the texture mimetype is png or jpeg
fn is_png_jpeg(image_src: &Source) -> bool {
    match image_src {
        Source::View { mime_type, .. } => *mime_type == "image/png" || *mime_type == "image/jpeg",
        Source::Uri { uri, .. } => uri.ends_with(".png") || uri.ends_with(".jpg") || uri.ends_with(".jpeg"),
    }
}

fn load_ktx2_textures(
    ktx2_data: &[u8],
    texture_idx: usize,
) -> Result<(Vec<u8>, vk::Format, vk::Extent3D, u32)> {
    let ktx_texture = ktx2::Reader::new(&ktx2_data)?;
    let header = ktx_texture.header();

    let format = header.format
        .map(|f| vk::Format::from_raw(f.value() as i32))
        .ok_or_else(|| anyhow!("KTX2 texture {} as no format", texture_idx))?;

    let extent = vk::Extent3D {
        width: header.pixel_width,
        height: header.pixel_height,
        depth: header.pixel_depth.max(1),
    };
    let mip_levels = header.level_count;

    let pixel_data = if header.supercompression_scheme == Some(ktx2::SupercompressionScheme::BasisLZ) {
        // transcode texture
        let mut transcoder = Transcoder::new();

        if transcoder.prepare_transcoding(ktx2_data).is_err() {
            return Err(anyhow!("Failed to prepare transcoding: {}", texture_idx));
            // continue 'texture;
        }

        // let transcode_format = TranscoderTextureFormat::BC7_RGBA;

        let mut all_levels_data = Vec::new();
        for level in 0..mip_levels {
            let params = TranscodeParameters {
                image_index: 0,
                level_index: level,
                decode_flags: None,
                output_row_pitch_in_blocks_or_pixels: None,
                output_rows_in_pixels: None,
            };

            // TODO: support multiple textures format transcoder in case the gpu doesn't support BC7.
            match transcoder.transcode_image_level(
                ktx2_data,
                TranscoderTextureFormat::BC7_RGBA,
                params,
            ) {
                Ok(data) => all_levels_data.extend_from_slice(&data),
                Err(e) => {
                    return Err(anyhow!("Failed to transcode level {}: {:?}", level, e));
                    // continue 'texture;
                }
            }
        }
        all_levels_data
    } else {
        // no need for transcoding
        let mut all_levels_data = Vec::new();
        for level in ktx_texture.levels() {
            all_levels_data.extend_from_slice(level.data);
        }
        all_levels_data
    };

    Ok((pixel_data, format, extent, mip_levels))
}

fn load_rgba_texture(
    raw_data: &[u8],
) -> Result<(Vec<u8>, vk::Format, vk::Extent3D, u32)> {
    let img = image::load_from_memory(raw_data)?.to_rgba8();
    let (w, h) = img.dimensions();
    let extent = vk::Extent3D { width: w, height: h, depth: 1 };
    let mip_levels = (w.max(h) as f32).log2().floor() as u32 + 1;

    Ok((
        img.into_raw(),
        vk::Format::R8G8B8A8_SRGB,
        extent,
        mip_levels,
    ))
}

fn load_gltf_textures(
    device: &Device,
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    textures: iter::Textures,
    setup_command_buffer: vk::CommandBuffer,
    graphics_queue: vk::Queue,
    buffers: &[Data],
    base_dir: &Path
) -> Result<Vec<TextureData>> {
    let mut vk_textures = Vec::new();
    'texture: for (i, texture) in textures.enumerate() {
        let image = texture.source();

        // 1 - Extract raw texture information
        let image_src = image.source();
        debug!("Texture source type: {:?}", image_src);

        let raw_data: Vec<u8> = match image_src {
            Source::View { ref view, .. } => {
                let buffer_data = &buffers[view.buffer().index()];
                let start = view.offset();
                let end = start + view.length();
                buffer_data[start..end].to_vec()
            }
            Source::Uri { uri, .. } => {
                let file_path =  base_dir.join(uri);
                std::fs::read(&file_path)
                    .map_err(|e| anyhow!("Failed to read {}: {}", file_path.display(), e))?
            }
        };

        // 2 - Load texture information
        let (pixels, format, extent, mip_levels) = if is_ktx2(&image_src) {
            match load_ktx2_textures(&raw_data, i) {
                Ok(data) => data,
                Err(e) => {
                    warn!("Could not load ktx2 texture {} > {}\nskipping.", i, e);
                    continue 'texture;
                }
            }
        }
        else if is_png_jpeg(&image_src) {
            match load_rgba_texture(&raw_data) {
                Ok(data) => data,
                Err(e) => {
                    warn!("Could not load png/jpeg texture {} > {}\nskipping.", i, e);
                    continue 'texture;
                }
            }
        }
        else {
            warn!("Texture {} is not KTX2 or JPEG or PNG, skipping", i);
            continue 'texture;
        };

        // 3 - Create Vulkan Image, Memory, View and Sampler
        let (texture_image, texture_image_memory) = create_texture_image(
            instance,
            device,
            physical_device,
            setup_command_buffer,
            graphics_queue,
            extent,
            mip_levels,
            &pixels,
            format,
        )?;
        let texture_image_view = create_texture_image_view(device, texture_image, mip_levels)?;
                
        // vk_textures.push((texture_image, texture_image_memory, texture_image_view, texture_sampler));
        vk_textures.push(TextureData {
            image: texture_image,
            image_memory: texture_image_memory,
            image_view: texture_image_view,
        });
    }

    Ok(vk_textures)
}

fn load_gltf_materials(
    materials: iter::Materials,
) -> Result<Vec<Material>> {

    let materials: Vec<Material> = materials.map(|material| {
        let pbr = material.pbr_metallic_roughness();
        let base_color_factor = Vec4::from(pbr.base_color_factor());
        let emissive_factor = Vec3::from(material.emissive_factor());

        let alpha_mask = match material.alpha_mode() {
            gltf::material::AlphaMode::Mask => 1.0,
            _ => 0.0,
        };

        Material {
            base_color_factor,
            metallic_factor: pbr.metallic_factor(),
            roughness_factor: pbr.roughness_factor(),
            emissive_factor,

            base_color_texture_idx: pbr.base_color_texture()
                .map(|info| Some(TextureId(info.texture().source().index())))
                .unwrap_or_default(),
            metallic_roughness_texture_idx: pbr.metallic_roughness_texture()
                .map(|info| Some(TextureId(info.texture().source().index())))
                .unwrap_or_default(),
            normal_texture_idx: material.normal_texture()
                .map(|text| Some(TextureId(text.texture().source().index())))
                .unwrap_or_default(),
            occlusion_texture_idx: material.occlusion_texture()
                .map(|text| Some(TextureId(text.texture().source().index())))
                .unwrap_or_default(),
            emissive_texture_idx: material.emissive_texture()
                .map(|text| Some(TextureId(text.texture().source().index())))
                .unwrap_or_default(),

            base_color_texture_set: pbr.base_color_texture()
                .map(|text| text.tex_coord() as i32)
                .unwrap_or(-1),
            metallic_roughness_texture_set: pbr.metallic_roughness_texture()
                .map(|text| text.tex_coord() as i32)
                .unwrap_or(-1),
            normal_texture_set: material.normal_texture()
                .map(|text| text.tex_coord() as i32)
                .unwrap_or(-1),
            occlusion_texture_set: material.occlusion_texture()
                .map(|text| text.tex_coord() as i32)
                .unwrap_or(-1),
            emissive_texture_set: material.emissive_texture()
                .map(|text| text.tex_coord() as i32)
                .unwrap_or(-1),
            
            alpha_mask,
            alpha_mask_cutoff: material.alpha_cutoff()
                .unwrap_or(0.5)

        }
    }).collect();

    Ok(materials)
}

fn load_gltf_animations(
    animations: iter::Animations,
    buffers: &[Data],
) -> Result<Vec<Animation>> {
    let animations: Vec<Animation> = animations.map(|anim| {
        // Samplers
        let samplers: Vec<AnimationSampler> = anim.samplers().map(|sampler| {
            let interpolation_type = match sampler.interpolation() {
                Interpolation::CubicSpline => InterpolationType::CUBICSPLINE,
                Interpolation::Linear => InterpolationType::LINEAR,
                Interpolation::Step => InterpolationType::STEP,
            };

            let inputs_access = sampler.input();
            let inputs_view = inputs_access.view().unwrap();
            let inputs_buffer = &buffers[inputs_view.buffer().index()];
            let start = inputs_view.offset() + inputs_access.offset();
            let inputs_data = &inputs_buffer[start..start + inputs_access.count() * inputs_access.size()];
            let inputs: Vec<f32> = inputs_data.chunks(4) // f32 size
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect();

            let outputs_access = sampler.output();
            let outputs_view = outputs_access.view().unwrap();
            let outputs_buffer = &buffers[outputs_view.buffer().index()];
            let start = outputs_access.offset() + outputs_view.offset();
            let output_data = &outputs_buffer[start..start + outputs_access.count() * outputs_access.size()];

            let (outputs_vec3, outputs_vec4) = match outputs_access.dimensions() {
                Dimensions::Vec3 => (
                    output_data
                        .chunks(12)
                        .map(|chunk| Vec3::new(
                            f32::from_le_bytes(chunk[0..4].try_into().unwrap()),
                            f32::from_le_bytes(chunk[4..8].try_into().unwrap()),
                            f32::from_le_bytes(chunk[8..12].try_into().unwrap())
                        ))
                        .collect(),
                    Vec::new()
                ),
                Dimensions::Vec4 => (
                    Vec::new(),
                    output_data
                        .chunks(16)
                        .map(|chunk| Vec4::new(
                            f32::from_le_bytes(chunk[0..4].try_into().unwrap()),
                            f32::from_le_bytes(chunk[4..8].try_into().unwrap()),
                            f32::from_le_bytes(chunk[8..12].try_into().unwrap()),
                            f32::from_le_bytes(chunk[12..16].try_into().unwrap())
                        ))
                        .collect()
                ),
                _ => (Vec::new(), Vec::new())
            };

            debug!("sampler interp={:?} inputs={} vec3={} vec4={}",
                interpolation_type,
                inputs.len(),
                outputs_vec3.len(),
                outputs_vec4.len());

            AnimationSampler {
                interpolation_type,
                inputs,
                outputs_vec4,
                outputs_vec3,
            }
        }).collect();

        // Channel
        let channels: Vec<AnimationChannel> = anim.channels().map(|channel| {
            let path = match channel.target().property() {
                Property::Translation => PathType::TRANSLATION,
                Property::Rotation => PathType::ROTATION,
                Property::Scale => PathType::SCALE,
                Property::MorphTargetWeights => PathType::MORPH,
            };

            let node_id = NodeId(channel.target().node().index());
            let sampler_id = SamplerId(channel.sampler().index());

            AnimationChannel {
                path,
                node_id,
                sampler_id,
            }
        })
        .collect();

        let (start, end) = samplers.iter().fold((f32::MAX, f32::MIN), |(start, end), s| {
            let s_start = s.inputs.first().cloned().unwrap_or(f32::MAX);
            let s_end   = s.inputs.last().cloned().unwrap_or(f32::MIN);
            (start.min(s_start), end.max(s_end))
        });

        // debug!("loading animation {}", anim.name().unwrap_or("no_name").to_string());

        Animation::new(
            anim.name().unwrap_or("").to_string(),
            samplers,
            channels,
            start,
            end
        )
    }).collect();

    
    Ok(animations)
}

pub fn load_gltf_skins(
    skins: iter::Skins,
    buffers: &[Data],
) -> Result<Vec<Skin>> {
    let skins = skins.map(|skin| -> Result<Skin> {
        let name = skin.name().unwrap_or("").to_string();

        let skeleton_root = skin.skeleton()
            .map(|node| NodeId(node.index()));

        let joints: Vec<NodeId> = skin.joints()
            .map(|node| NodeId(node.index()))
            .collect();

        let inverse_bind_mats = skin.reader(|buffer| Some(&buffers[buffer.index()]))
            .read_inverse_bind_matrices()
            .map(|iter| iter.map(Mat4::from).collect())
            .unwrap_or_else(|| vec![Mat4::identity(); joints.len()]);

        debug!("skin '{}' : {} joints, {} inverse_bind_mats", name, joints.len(), inverse_bind_mats.len());

        Ok(Skin {
            name,
            skeleton_root,
            inverse_bind_mats,
            joints,
        })

    }).collect::<Result<Vec<_>>>()?;

    Ok(skins)
}

pub fn load_gltf_model(
    device: &Device,
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    path: &str,
    setup_command_buffer: vk::CommandBuffer,
    graphics_queue: vk::Queue,
) -> Result<(ModelGraph, Vec<TextureData>)> {
    info!("loading model: {:?}", path);

    // warn: this method does not support reading gltf / glb from web.
    let (document, buffers, _) = gltf::import(path)?;
    let base_dir = std::path::Path::new(path).parent()
        .ok_or_else(|| anyhow!("Invalid path"))?;

    // 1 - load textures
    let textures = load_gltf_textures(
        device,
        instance,
        physical_device,
        document.textures(),
        setup_command_buffer,
        graphics_queue,
        &buffers,
        base_dir
    )?;

    // 2 - load materials
    let materials = load_gltf_materials(document.materials())?;

    // 3 - create a scene-graph
    // Use a two-pass approach to ensure all nodes exist before we try to link them together.

    // 3.a - create all nodes data 
    let mut linear_nodes: Vec<Node<ModelNodeData>> = document.nodes().map(|node| {
        let transform = match node.transform() {
            gltf::scene::Transform::Matrix { matrix } => {
                NodeTransform::Matrix(Mat4::from(matrix))
            },
            gltf::scene::Transform::Decomposed { translation, rotation, scale } => {
                NodeTransform::Trs {
                    translation: Vec3::new(translation[0], translation[1], translation[2]),
                    rotation: Quat::new(rotation[3], rotation[0], rotation[1], rotation[2]),
                    scale: Vec3::new(scale[0], scale[1], scale[2])
                }
            },
        };

        let skin_index = node.skin().map(|skin| skin.index() as i32).unwrap_or(-1);
        let model_node_data = ModelNodeData {
            name: node.name().unwrap_or("").to_string(),
            transform,
            skin: skin_index,

            ..ModelNodeData::default()
        };

        Node::<ModelNodeData> {
            parent: None,
            childs: Vec::new(),
            value: model_node_data,
        }
    }).collect();
    info!("loaded {} nodes for this model", linear_nodes.len());

    for node in document.nodes() {
        // 3.b - establish parent-child relationships
        let parent_id = NodeId(node.index());
        for child in node.children() {
            let child_id = NodeId(child.index());

            linear_nodes[child_id.0].parent = Some(parent_id);
            linear_nodes[parent_id.0].childs.push(child_id);
        }

        // 4 - Load Mesh
        if let Some(gltf_mesh) = node.mesh() {
            let mut mesh = Mesh::default();

            for primitive in gltf_mesh.primitives() {
                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

                if let Some(iter) = reader.read_indices() {
                    let vertex_base = mesh.vertices.len() as u32;
                    let first_index = mesh.indices.len() as u32;

                    // mesh.material_id = primitive.material().index().map(|i| Some(MaterialId(i))).unwrap_or(None);

                    let positions = reader.read_positions().expect("primitive has no positions");
                    let mut normals = reader.read_normals();
                    let mut uv0 = reader.read_tex_coords(0).map(|t| t.into_f32());
                    let mut uv1 = reader.read_tex_coords(1).map(|t| t.into_f32());
                    let mut joints = reader.read_joints(0).map(|j| j.into_u16());
                    let mut weights = reader.read_weights(0).map(|w| w.into_f32());
                    let mut tangents = reader.read_tangents();

                    for p in positions {
                        let n = normals.as_mut().and_then(|iter| iter.next()).unwrap_or([0.0, 0.0, 1.0]);
                        let t0 = uv0.as_mut().and_then(|iter| iter.next()).unwrap_or([0.0, 0.0]);
                        let t1 = uv1.as_mut().and_then(|iter| iter.next()).unwrap_or([0.0, 0.0]);
                        let j = joints.as_mut().and_then(|iter| iter.next()).unwrap_or([0, 0, 0, 0]);
                        let w = weights.as_mut().and_then(|iter| iter.next()).unwrap_or([1.0, 0.0, 0.0, 0.0]);
                        let tg = tangents.as_mut().and_then(|iter| iter.next()).unwrap_or([1.0, 0.0, 0.0, 1.0]);

                        mesh.vertices.push(Vertex {
                            pos:				Vec3::new(p[0], p[1], p[2]),
                            normal:				Vec3::new(n[0], n[1], n[2]),
                            color:				Vec3::new(1.0, 1.0, 1.0),
                            uv0:			    Vec2::new(t0[0], t0[1]),
                            uv1:                Vec2::new(t1[0], t1[1]),
                            tangent:            Vec4::new(tg[0], tg[1], tg[2], tg[3]),
                            joint_indices:		UVec4::new(j[0], j[1], j[2], j[3]),
							joint_weights:		Vec4::new(w[0], w[1], w[2], w[3]),
                        });
                    }

                    let indices: Vec<u32> = iter.into_u32().map(|i| i + vertex_base).collect();
                    let index_count = indices.len() as u32;
                    mesh.indices.extend(indices);

                    mesh.primitives.push(
                        Primitive { 
                            first_index,
                            index_count,
                            material_id: primitive.material().index().map(MaterialId),
                        }
                    );
                }
                else {
                    warn!("Primitive without index; skipping.")
                }
            }
            
            linear_nodes[node.index()].value.mesh = Some(mesh);
        }
    }

    let roots: Vec<NodeId> = document.default_scene()
        .or_else(|| document.scenes().next())
        .ok_or_else(|| anyhow!("glTF has no scene"))?
        .nodes()
        .map(|node| NodeId(node.index()))
        .collect();

    let graph = FlatGraph::new(linear_nodes);
    
    // 5 - skins
    let skins = load_gltf_skins(
        document.skins(),
        &buffers,
    )?;

    // 6 - animations
    let animations = load_gltf_animations(
        document.animations(),
        &buffers,
    )?;

    let model = ModelGraph::new(
        graph,
        roots,
        materials,
        animations,
        skins,
    );

    model.get_debug_info()?;

    Ok((
        model,
        textures
    ))
}

fn register_model_textures(
    textures_storage: &mut TexturesStorage,
    model: &mut ModelGraph,
    new_textures: Vec<TextureData>
) -> TextureId {
    let offset = textures_storage.extend(new_textures);
    model.offset_materials_texture_ids(offset);
    offset
}

pub fn load_model_with_offset(
    device: &Device,
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    path: &str,
    name: &str,
    setup_command_buffer: vk::CommandBuffer,
    graphics_queue: vk::Queue,
    models: &mut ModelsStorage,
    textures: &mut TexturesStorage,
    registry: &mut ModelRegistry,
) -> Result<()> {
    let (mut model_graph, model_textures) = load_gltf_model(
        device, instance, physical_device, path, setup_command_buffer, graphics_queue,
    )?;

    let texture_id = register_model_textures(textures, &mut model_graph, model_textures);
    
    let model_id = models.push(model_graph);
    
    registry.entries.insert(name.to_string(), ModelAssets { model_id, texture_id });

    Ok(())
}
// endregion