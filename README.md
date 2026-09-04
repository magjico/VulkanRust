# VulkanRust

A personal Vulkan engine in Rust — built with [`vulkanalia`](https://github.com/KyleMayes/vulkanalia), [`winit`](https://github.com/rust-windowing/winit) and [`bevy_ecs`](https://github.com/bevyengine/bevy). Originally adapted from [vulkan-tutorial.com](https://vulkan-tutorial.com) / [KyleMayes/vulkanalia tutorial](https://kylemayes.github.io/vulkanalia/), extended into a small PBR engine with ECS, skeletal animation and instanced rendering.

> **Note:** `vulkan-tutorial/` was renamed to `vulkan-engine/` as part of [#31](https://github.com/magjico/VulkanRust/issues/31). All sources, shaders and config now live under `vulkan-engine/`. If you have a local clone on `main`, run `git pull --rebase` — `git mv` preserves history.

Forked from [magjico/VulkanRust](https://github.com/magjico/VulkanRust).

## Features

- **Vulkan 1.x** via `vulkanalia 0.35` (dynamic rendering `VK_KHR_dynamic_rendering`, `VK_KHR_synchronization2`, MSAA, depth/color attachments)
- **Window + input** via `winit 0.30` + `winit_input_helper`
- **ECS** via `bevy_ecs 0.19` — transforms, skeletons, instances, lights as components/resources (`propagate_transforms_from_root`, `update_skeletons`)
- **PBR rendering** — `shaders/hello_model/shader.vert/.frag` (push constants for model/material, descriptor sets for global/material/skin/instance)
- **Skeletal animation** — up to 256 instances × 64 joints, double-buffered per frame-in-flight (`constants.rs: MAX_INSTANCES`, `FRAME_STRIDE`)
- **Instanced rendering** with material sorting / descriptor-set tracking (draw call coalescing ~500→900 fps optimization)
- **Asset loading** — glTF 2.0 (`gltf` + `tobj`), PNG/KTX2/EXR/`basis-universal`/`image`, default 1×1 white texture fallback
- **Config-driven input** — `vulkan-engine/config/input.toml`
- Shader compilation via `vulkan-engine/build.rs` + `glslc` (Vulkan SDK)

Demo models loaded by default (`constants.rs: MODEL_INFO`): `CesiumMan` (with animation + skin) and `BrainStem` — spawned as 5 entities in `setup.rs: spawn_from_cesium_man_instances` / `spawn_from_brain_stem_instance`.

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Rust stable | `edition = "2024"` (`vulkan-engine/Cargo.toml:4`), install via `rustup` with `clippy` component |
| Vulkan SDK / driver | Loader + validation layers + `glslc`/`spirv-tools`. Ubuntu: `sudo apt install libvulkan1 vulkan-tools libvulkan-dev vulkan-validationlayers-dev spirv-tools` (see `.github/workflows/rust.yml:27-30`). Windows/macOS: install [LunarG Vulkan SDK](https://www.lunarg.com/vulkan-sdk/) and ensure `glslc` is on `PATH` |
| GPU with Vulkan 1.1+ | Discrete GPU recommended; validation layers enabled in `debug_assertions` (`constants.rs: VALIDATION_ENABLED`) |

## Quick Start

```bash
# 1. Clone (fork URL shown)
git clone https://github.com/JITHUKANNAN-TJ/VulkanRust.git
cd VulkanRust

# 2. Build — Cargo.toml lives inside vulkan-engine/
cd vulkan-engine
cargo build --verbose

# or from repo root:
cargo build --verbose --manifest-path vulkan-engine/Cargo.toml

# 3. Run (window 1024×768, FPS logged every second)
cargo run --verbose
# or
cargo run --manifest-path vulkan-engine/Cargo.toml
```

Shaders are compiled automatically by `vulkan-engine/build.rs` on first build (`glslc shaders/hello_model/shader.vert -o shaders/hello_model/vert.spv`). If `glslc` is missing the build will `panic!` with `Failed to compile the vertex shader`. Manual fallback:

```bash
cd vulkan-engine
glslc shaders/hello_model/shader.vert -o shaders/hello_model/vert.spv
glslc shaders/hello_model/shader.frag -o shaders/hello_model/frag.spv
# Windows alternative
shaders\hello_model\compile.bat
```

### Lint & Test

```bash
cd vulkan-engine
cargo clippy --all-targets -- -D warnings
cargo test --verbose
```

CI (`.github/workflows/rust.yml`) runs the same steps on `ubuntu-latest` with `Swatinem/rust-cache` and Vulkan SDK install.

## Controls

Default bindings in `vulkan-engine/config/input.toml`:

| Action | Key |
|--------|-----|
| forward / backward | `w` / `s` |
| left / right | `a` / `d` |
| up / down | `space` / `ctrl_left` |
| look | `mouse right-drag` (dx, dy × sensitivity) |
| zoom | `scroll wheel` |

Camera (`scene/camera.rs`, `CameraBuilder`) starts at `(0, 8, 4)` looking at `(0, 0, 0.75)`, `movement_speed 10.0`, `mouse_sensitivity 0.02`. ESC / window close exits (handled in `app.rs: AppManager::about_to_wait`).

## Project Structure

```
VulkanRust/
├── .github/workflows/rust.yml   # CI: Rust + Vulkan SDK, cargo build/clippy/test (working-directory: vulkan-engine/)
├── .gitignore                    # ignores target/, *.spv, *.glb/.gltf, vulkan-engine/resources/...
├── LICENSE                       # Apache-2.0
├── README.md                     # this file
└── vulkan-engine/                # ← renamed from vulkan-tutorial/ [#31]
    ├── Cargo.toml                # package `graphic-env` 0.1.5
    ├── build.rs                  # glslc shader compilation
    ├── config/input.toml         # key bindings
    ├── shaders/
    │   ├── hello_model/shader.{vert,frag} (+ vert.spv/frag.spv built)
    │   └── hello_triangle/       # legacy triangle example
    └── src/
        ├── main.rs               # winit EventLoop → AppManager
        ├── lib.rs                # module re-exports
        ├── app.rs                # AppManager + App (52k loc, Vulkan init, frame loop)
        ├── constants.rs          # CORRECTION matrix, MAX_FRAMES_IN_FLIGHT, paths, MODEL_INFO
        ├── setup.rs              # descriptor pools/sets, sync objects, ECS init, model loading
        ├── gpu/{device,instance,memory,pipeline,shader}.rs
        ├── render/{color,depth,swapchain,texture,uniform}.rs
        ├── resources/{buffer,command,image}.rs
        ├── scene/{animation,camera,geometry,light,model,ecs/*}.rs
        ├── assets/model_io.rs    # glTF loading
        ├── ops/{allocator,copy,graph}.rs
        ├── input/keyboard.rs
        ├── math.rs
        ├── type_safety.rs
        └── debug.rs
```

## Architecture Overview

1. **Entry** (`src/main.rs`) — `EventLoop::new()` → `AppManager { input: WinitInputHelper }`.
2. **Init** (`app.rs: App::create`) — `Entry` → `Instance` → `surface` → `pick_best_physical_device` → `create_logical_device` → `DeviceData` (MSAA samples, queue families) → `ECSContext::init` (`SkinningBuffer`, `InstanceBuffer`, `SlotAllocator`) → `SwapchainData` → `CommandData` → `ColorData`/`DepthData` → `PipelineData` (global/material/skin/instance layouts) → `ModelsStorage`/`TexturesStorage` via `load_gltf_models` → `BuffersData` (interleaved VBO/IBO, UBO) → `LightBuffer` → `DescriptorData` → `SyncData` → `Camera`.
3. **Frame** (`App::render`, `CommandData::update_command_buffer` + `record_secondary_command_buffer`) — acquire swapchain image → update UBO (`view`/`proj` with `CORRECTION`) → sort instances by `model_id`, build `instance_data` + `draw_list` sorted by `MaterialSetId` → write `InstanceBuffer` → record secondary command buffer (bind pipeline/VBO/IBO, global/skin/instance descriptors, per-material bind tracking, `cmd_push_constants` vert/frag, `cmd_draw_indexed` instanced) → submit → present → rotate `frame = (frame+1) % MAX_FRAMES_IN_FLIGHT`.
4. **ECS** (`setup.rs: init_ecs_context`) — `World` resources: `VulkanDevice`, `SkinningBuffer`, `InstanceBuffer`, `SSBOSkiningAllocator`, `Time`, `CurrentFrame`; schedule: `propagate_transforms_from_root → propagate_transforms_to_children → update_skeletons_wrapped`.

## Configuration

- `vulkan-engine/config/input.toml` — remap keys without recompiling. Reloaded at startup via `InputBindings::bind_from_file` (`constants.rs: INPUT_PATH`).
- `vulkan-engine/src/constants.rs` — tweak `VALIDATION_ENABLED`, `DEVICE_EXTENSIONS`, `MAX_INSTANCES`, `MODEL_INFO`, shader `VERT`/`FRAG` `include_bytes!` paths.

Adding a model: add `(path, key)` to `MODEL_INFO`, register spawn in `setup.rs`, place `.glb/.gltf` under `vulkan-engine/resources/glb/` (gitignored by default — commit sample assets explicitly or adjust `.gitignore:18`).

## Troubleshooting

- **`cargo build` fails at repo root: `could not find Cargo.toml`** — `Cargo.toml` is inside `vulkan-engine/`. `cd vulkan-engine` or use `--manifest-path`.
- **`Failed to compile the vertex shader` / `glslc not found`** — install Vulkan SDK and add `glslc` to `PATH`; `spirv-tools` needed on Ubuntu.
- **Validation layer errors** — enabled only in debug (`cfg!(debug_assertions)`). Check `vulkan-validationlayers-dev` installed.
- **Black screen / no models** — resources under `vulkan-engine/resources/` are gitignored (`*.glb/.gltf` etc. in `.gitignore:12-13` plus `vulkan-engine/resources/...` lines). Ensure sample `CesiumMan`/`BrainStem` assets exist or the app will still run but spawn fallback.

## Contributing

PRs welcome — this repo tracks upstream `magjico/VulkanRust` issues. For #31-type docs tasks, branch from `main`, run `cargo clippy` + `cargo test` inside `vulkan-engine/`, and reference the issue (`Fixes #31`).

## License

[Apache-2.0](LICENSE) — same as upstream. Copyright notice in `LICENSE` Appendix.

## Acknowledgements

- [Alexander Overvoorde / vulkan-tutorial.com](https://vulkan-tutorial.com) and [KyleMayes/vulkanalia](https://kylemayes.github.io/vulkanalia/) (tutorial adaptation)
- [Khronos Vulkan Samples](https://github.com/KhronosGroup/Vulkan-Samples)
- [vulkano-rs](https://github.com/vulkano-rs/vulkano) for community reference
