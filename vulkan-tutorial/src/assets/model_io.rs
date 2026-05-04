// TODO: implement a generic tray for all io / so we need to implement load and save in a generic way.
use std::{collections::HashMap, io::BufReader};
use std::fs::File;

use anyhow::Result;
use cgmath::{vec2, vec3};

use crate::scene::{Vertex, Mesh};

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