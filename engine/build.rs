use std::process::Command;

fn main() {
    println!("cargo::rerun-if-changed=shaders/hello_model/shader.vert");
    println!("cargo::rerun-if-changed=shaders/hello_model/shader.frag");
    println!("cargo::rerun-if-changed=shaders/ibl/brdf_lut.comp");
    println!("cargo::rerun-if-changed=shaders/ibl/equirect_to_cubemap.comp");
    println!("cargo::rerun-if-changed=shaders/ibl/irr_map.comp");
    println!("cargo::rerun-if-changed=shaders/ibl/prefiltered.comp");
    println!("cargo::rerun-if-changed=shaders/skybox/skybox.vert");
    println!("cargo::rerun-if-changed=shaders/skybox/skybox.frag");
    println!("cargo:warning=Compiling shaders...");

    Command::new("glslc")
        .args(["shaders/hello_model/shader.vert", "-o", "shaders/hello_model/vert.spv"])
        .status()
        .expect("Failed to compile the vertex shader");

    Command::new("glslc")
        .args(["shaders/hello_model/shader.frag", "-o", "shaders/hello_model/frag.spv"])
        .status()
        .expect("Failed to compile the fragment shader");

    Command::new("glslc")
        .args(["shaders/ibl/brdf_lut.comp", "-o", "shaders/ibl/brdf_lut.spv"])
        .status()
        .expect("Failed to compile the BRDF LUT compute shader");

    Command::new("glslc")
        .args(["shaders/ibl/equirect_to_cubemap.comp", "-o", "shaders/ibl/equirect_to_cubemap.spv"])
        .status()
        .expect("Failed to compile Equirect to Rectangle compute shader");

    Command::new("glslc")
        .args(["shaders/ibl/irr_map.comp", "-o", "shaders/ibl/irr_map.spv"])
        .status()
        .expect("Failed to compile Irradiance Cubemap compute shader");

    Command::new("glslc")
        .args(["shaders/ibl/prefiltered.comp", "-o", "shaders/ibl/prefiltered.spv"])
        .status()
        .expect("Failed to compile Prefiltered Cubemap compute shader");

    Command::new("glslc")
        .args(["shaders/skybox/skybox.vert", "-o", "shaders/skybox/skybox_vert.spv"])
        .status()
        .expect("Failed to compile skybox vertex shader");

    Command::new("glslc")
        .args(["shaders/skybox/skybox.frag", "-o", "shaders/skybox/skybox_frag.spv"])
        .status()
        .expect("Failed to compile skybox fragment shader");

}