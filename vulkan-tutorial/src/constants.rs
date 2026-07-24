use vulkanalia::prelude::v1_0::*;
use vulkanalia::Version;

use crate::math::Mat4;

//==================================
// Rendering Consts
//==================================

/// This matrix is use to correct the proj matrix from the OpenGL range to the Vulkan one.
/// 
/// **WARN**: this is a **column-major** matrix.
pub const CORRECTION: Mat4 = Mat4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, -1.0, 0.0, 0.0,
    0.0, 0.0, 1.0 / 2.0, 0.0,
    0.0, 0.0, 1.0 / 2.0, 1.0
);

pub const MAX_INSTANCES: u32 = 256;
pub const MAX_JOINT_PER_INSTANCE: u32 = 64;
pub const MAX_TOTAL_JOINTS: usize = (MAX_INSTANCES * MAX_JOINT_PER_INSTANCE) as usize;

//==================================
// Info Consts
//==================================

/// The Vulkan SDK version that started requiring the portability subset extension for macOS.
pub const PORTABILITY_MACOS_VERSION: Version = Version::new(1, 3, 216);
/// Engine name
pub const ENGINE_NAME: &[u8] = b"No Engine\0";
/// Application Name
pub const APP_NAME: &[u8] = b"Vulkan (Rust)\0";

//==================================
// Params Consts
//==================================

// TODO: move this params bellow into a init parameters files 
/// Whether the validation layers should be enabled.
pub const VALIDATION_ENABLED: bool = cfg!(debug_assertions);
/// The name of the validation layers.
pub const VALIDATION_LAYER: vk::ExtensionName = vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");
/// The required device extensions.
pub const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[vk::KHR_SWAPCHAIN_EXTENSION.name,
    vk::KHR_SYNCHRONIZATION2_EXTENSION.name, vk::KHR_DYNAMIC_RENDERING_EXTENSION.name];
/// The maximum number of frames that can be processed concurrently.
pub const MAX_FRAMES_IN_FLIGHT: usize = 2;
/// Vert shader
pub const VERT: &[u8] = include_bytes!("../shaders/hello_model/vert.spv");
/// Frag shader
pub const FRAG: &[u8] = include_bytes!("../shaders/hello_model/frag.spv");

//==================================
// Paths
//==================================

// 1 - Viking room
/// The texture test path
pub const TEXTURE_PATH: &str = "resources/textures/viking_room.png";
/// The obj test path
pub const MESH_PATH: &str = "resources/3D_meshes/viking_room.obj";

// 2 - glb 
/// Duck - mesh store with texture
pub const DUCK_PATH: &str = "resources/glb/glTF-Sample-Models/2.0/glTF/Duck.glb";
/// Flying Helmet - texture separated from mesh
pub const FLIGHT_HELM_PATH: &str = "resources/glb/glTF-Sample-Models/2.0/FlightHelmet/glTF/FlightHelmet.gltf";
/// Cesium Man
pub const CESIUM_MAN_PATH: &str = "resources/glb/glTF-Sample-Models/2.0/CesiumMan/glTF/CesiumMan.gltf";
/// To load multiple models, we can use a list of tuples with the path and the name of the model.
pub const MODEL_INFO: [(&str, &str); 2] = [
    (CESIUM_MAN_PATH, "CesiumMan"),
    (FLIGHT_HELM_PATH, "FlightHelmet")
];

// 3 - Inputs
pub const INPUT_PATH: &str = "config/input.toml";