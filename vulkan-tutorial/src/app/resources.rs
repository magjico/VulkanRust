use anyhow::Result;

use winit::window::Window;

use vulkanalia::prelude::v1_0::*;

use crate::math::Vec3;
use crate::gpu::{
	Pipeline,
    GPUContext,
};
use crate::render::{
	Swapchain,
	ColorAttachment,
	DepthAttachment,
	DescriptorLayouts,
	Descriptors,
	CommandRecorder,
	TexturesStorage,
	TextureData,
};
use crate::resources::{
	UniformBuffers,
	GeometryBuffer,
	FrameSync
};
use crate::scene::{
	Time,
	Camera,
	CameraBuilder,
	ECSContext,
	LightBuffer,
	ModelsStorage,
	ModelRegistry,
};
use crate::constants::{
	VERT,
	FRAG,
	MAX_FRAMES_IN_FLIGHT,
};
use crate::setup::{
	load_gltf_models,
	create_default_texture,
	create_default_lighting,
	spawn_from_cesium_man_instances,
	spawn_from_brain_stem_instance,
};

/// Resources whose lifetime is tied to the swapchain.
///
/// All of these depend on the swapchain extent, format or image count, and are
/// therefore destroyed and rebuilt whenever the window is resized.
///
/// ## Fields
///
/// - `swapchain` ( [Swapchain] ) - Presentation images and their views.
/// - `color_attachment` ( [ColorAttachment] ) - Multisampled color target, resolved into the swapchain image at the end of the pass.
/// - `depth_attachment` ( [DepthAttachment] ) - Depth target, sized to the swapchain extent.
#[derive(Debug)]
pub struct SwapchainDependant {
	pub swapchain:			Swapchain,
	pub color_attachment:	ColorAttachment,
	pub depth_attachment:	DepthAttachment,
}

impl SwapchainDependant {
	pub fn new(
		window:		&Window,
		instance:	&Instance,
		device:		&Device,
		gpu:		&mut GPUContext,
		surface:	vk::SurfaceKHR,
	) -> Result<Self> {
		let swapchain = Swapchain::new(
			window,
			instance,
			device,
			gpu.physical_device,
			surface,
			&mut gpu.queue_family_indices
		)?;

		let color_attachment = ColorAttachment::new(
			instance,
			device,
			gpu.physical_device,
			swapchain.vk_extent.width,
			swapchain.vk_extent.height,
			gpu.max_msaa_samples,
			swapchain.vk_format
		)?;

		let depth_attachment = DepthAttachment::new(
			instance,
			device,
			gpu.physical_device,
			swapchain.vk_extent.width,
			swapchain.vk_extent.height,
			gpu.max_msaa_samples
		)?;

		Ok(Self {
			swapchain,
			color_attachment,
			depth_attachment,
		})
	}

	#[rustfmt::skip]
    #[allow(unsafe_op_in_unsafe_fn)]
	pub unsafe fn destroy(&self, device: &Device) {
        self.depth_attachment.destroy(device);
		self.color_attachment.destroy(device);
        self.swapchain.destroy(device);
	}

	/// Rebuilds the swapchain and its attachments after a window resize.
	///
	/// The previous resources are destroyed first, so the caller must have waited on
	/// device idle beforehand.
	#[rustfmt::skip]
	pub fn recreate(
		&mut self,
		window: 			&Window,
		device:				&Device,
		instance:			&Instance,
		surface:			vk::SurfaceKHR,
		gpu:				&mut GPUContext,
	) -> Result<()> {
		unsafe { self.destroy(device) };
		*self = Self::new(window, instance, device, gpu, surface)?;
		Ok(())
	}
}

/// Resources created once and kept for the whole run of the application.
///
/// None of these depend on the swapchain, so they survive window resizes.
///
/// ## Fields
///
/// - `gpu` ( [GPUContext] ) - Selected GPU, its queues and capabilities.
/// - `descriptor_layouts` ( [DescriptorLayouts] ) - Layouts of the four descriptor sets, also needed to build the pipeline layout.
/// - `descriptors` ( [Descriptors] ) - Descriptor pool and every set allocated from it.
/// - `graphic_pipeline` ( [Pipeline] ) - Graphics pipeline and its layout, baked against the swapchain format, the depth format and the sample count.
/// - `uniform_buffers` ( [UniformBuffers] ) - One uniform buffer per swapchain image, so the count follows the image count.
/// - `command_recorder` ( [CommandRecorder] ) - Command pools, command buffers, and the per-frame scratch buffers used while recording.
/// - `geometry` ( [GeometryBuffer] ) - Vertices and indices of every loaded model, with the offsets locating each mesh.
/// - `textures` ( [TexturesStorage] ) - Every texture of every loaded model, indexed by ID.
/// - `default_texture` ( [TextureData] ) - Fallback bound wherever a material declares no texture.
/// - `camera` ( [Camera] ) - Active camera, source of the view and projection matrices.
/// - `light_buffer` ( [LightBuffer] ) - Scene lights, read by the fragment shader.
/// - `sync` ( [FrameSync] ) - Semaphores and fences ordering frames in flight.
#[derive(Debug)]
pub struct Persistent {
	pub gpu:					GPUContext,
	pub descriptor_layouts:		DescriptorLayouts,
	pub descriptors:			Descriptors,
	pub graphic_pipeline:		Pipeline,
	pub uniform_buffers:		UniformBuffers,
	pub command_recorder:		CommandRecorder,
	pub geometry_buffer:		GeometryBuffer,
	pub textures:				TexturesStorage,
	pub default_texture:		TextureData,
	pub camera:					Camera,
	pub light_buffer:			LightBuffer,
	pub sync:					FrameSync,
}

impl Persistent {
	pub fn new(
		instance: 				&Instance,
		device:					&Device,
		ecs_context:			&mut ECSContext,
		transient:				&SwapchainDependant,
		gpu:					GPUContext,
		descriptor_layouts:		DescriptorLayouts,
	) -> Result<Self> {
		let swapchain_images_count = transient.swapchain.vk_images.len();

		let graphic_pipeline = Pipeline::new(
			device,
			&[
				(VERT, vk::ShaderStageFlags::VERTEX),
				(FRAG, vk::ShaderStageFlags::FRAGMENT),
			],
			transient.swapchain.vk_format,
			transient.depth_attachment.vk_format,
			gpu.max_msaa_samples,
			&descriptor_layouts.as_slice()
		)?;

		let uniform_buffers = UniformBuffers::new(
            &instance,
            &device,
            gpu.physical_device,
            swapchain_images_count
        )?;

		let command_recorder = CommandRecorder::new(
			device,
			&gpu.queue_family_indices,
			swapchain_images_count
		)?;

		let mut models = ModelsStorage::new();
		let mut textures = TexturesStorage::new();
		let mut model_registry = ModelRegistry::new();

		load_gltf_models(
			device,
			instance,
			gpu.physical_device,
			command_recorder.setup_command_buffer,
			gpu.graphics_queue,
			&mut models,
			&mut textures,
			&mut model_registry
		)?;

		let default_texture = create_default_texture(
			instance,
			device,
			gpu.physical_device,
			command_recorder.setup_command_buffer,
			gpu.graphics_queue
		)?;

		let geometry_buffer = GeometryBuffer::new(
			instance,
			device,
			gpu.physical_device,
			&models,
			command_recorder.setup_command_buffer,
			gpu.graphics_queue
		)?;

		let light_buffer = create_default_lighting(
			instance,
			device,
			gpu.physical_device
		)?;

		let descriptors = Descriptors::new(
			device,
			ecs_context,
			&descriptor_layouts,
			&uniform_buffers.vk_buffers,
			&textures,
			&default_texture,
			&models,
			&light_buffer,
			swapchain_images_count
		)?;

		let sync = FrameSync::new(
			device,
			MAX_FRAMES_IN_FLIGHT,
			swapchain_images_count
		)?;

		// TODO: support multiple cameras
        let mut camera = CameraBuilder::new()
            .position(Vec3::new(0.0,8.0, 4.0))
            .movement_speed(10.0)
            .mouse_sensitivity(0.02)
            .build();

        camera.look_at(Vec3::new(0.0, 0.0, 0.75), None);

		// update ecs_context + spawn entities
		ecs_context.world.insert_resource(models);
		ecs_context.world.insert_resource(Time(0.0));
		spawn_from_cesium_man_instances(&mut ecs_context.world, &model_registry)?;
        spawn_from_brain_stem_instance(&mut ecs_context.world, &model_registry)?;

		Ok(Self {
			gpu,
			descriptor_layouts,
			descriptors,
			graphic_pipeline,
			uniform_buffers,
			command_recorder,
			geometry_buffer,
			textures,
			default_texture,
			camera,
			light_buffer,
			sync,
		})
	}

	#[rustfmt::skip]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn destroy(&self, device: &Device) {
		self.uniform_buffers.destroy(device);
		self.graphic_pipeline.destroy(device);
		self.textures.destroy(device);
		self.default_texture.destroy(device);
		self.light_buffer.destroy(device);
		self.descriptors.destroy(device);
		self.descriptor_layouts.destroy(device);
		self.sync.destroy(device);
		self.geometry_buffer.destroy(device);
		self.command_recorder.destroy(device);
	}
}