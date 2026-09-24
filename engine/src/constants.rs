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

/// The maximum number of frames that can be processed concurrently.
pub const MAX_FRAMES_IN_FLIGHT: usize = 2;
/// Maximum number of skinned instances that can be rendered simultaneously.
/// Defines the slot count of the skinning `SlotAllocator`.
pub const MAX_INSTANCES: u32 = 256;
/// Maximum number of joints a single skeleton may have.
/// Each instance reserves this many matrices, whether it uses them all or not.
pub const MAX_JOINT_PER_INSTANCE: u32 = 64;
/// Number of joint matrices reserved for one full frame (all instances).
/// Used to offset writes and reads into the per-frame region of the skinning buffer.
pub const FRAME_STRIDE: usize = (MAX_INSTANCES * MAX_JOINT_PER_INSTANCE) as usize;
/// Total joint matrix capacity of the skinning buffer, duplicated per frame in flight
/// so the CPU never overwrites data a pending frame is still reading.
pub const MAX_TOTAL_JOINTS: usize = FRAME_STRIDE * MAX_FRAMES_IN_FLIGHT;
/// Total instance data capacity, duplicated per frame in flight for the same reason.
pub const MAX_INSTANCES_DATA: usize = MAX_INSTANCES as usize * MAX_FRAMES_IN_FLIGHT;

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
/// Vert shader
pub const VERT: &[u8] = include_bytes!("../shaders/hello_model/vert.spv");
/// Frag shader
pub const FRAG: &[u8] = include_bytes!("../shaders/hello_model/frag.spv");
/// Compute shader
pub const COMP: &[u8] = include_bytes!("../shaders/hello_model/comp.spv");


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
pub const DUCK_KEY: &str = "Duck";
/// Flying Helmet - texture separated from mesh
pub const FLIGHT_HELM_PATH: &str = "resources/glb/glTF-Sample-Models/2.0/FlightHelmet/glTF/FlightHelmet.gltf";
pub const FLIGHT_HELM_KEY: &str = "FlightHelmet";
/// Cesium Man
pub const CESIUM_MAN_PATH: &str = "resources/glb/glTF-Sample-Models/2.0/CesiumMan/glTF/CesiumMan.gltf";
pub const CESIUM_MAN_KEY: &str = "CesiumMan";
/// Brain Stem
pub const BRAIN_STEM_PATH: &str = "resources/glb/glTF-Sample-Models/2.0/BrainStem/glTF/BrainStem.gltf";
pub const BRAIN_STEM_KEY: &str = "BrainStem";
/// To load multiple models, we can use a list of tuples with the path and the name of the model.
pub const MODEL_INFO: [(&str, &str); 2] = [
    (CESIUM_MAN_PATH, CESIUM_MAN_KEY),
    (BRAIN_STEM_PATH, BRAIN_STEM_KEY)
];

// 3 - Inputs
pub const INPUT_PATH: &str = "config/input.toml";