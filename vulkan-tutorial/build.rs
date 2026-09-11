use std::process::Command;

fn main() {
    println!("cargo::rerun-if-changed=shaders/hello_model/shader.vert");
    println!("cargo::rerun-if-changed=shaders/hello_model/shader.frag");
    println!("cargo:warning=Compiling shaders...");

    Command::new("glslc")
        .args(["shaders/hello_model/shader.vert", "-o", "shaders/hello_model/vert.spv"])
        .status()
        .expect("Failed to compile the vertex shader");

    Command::new("glslc")
        .args(["shaders/hello_model/shader.frag", "-o", "shaders/hello_model/frag.spv"])
        .status()
        .expect("Failed to compile the fragment shader");
}