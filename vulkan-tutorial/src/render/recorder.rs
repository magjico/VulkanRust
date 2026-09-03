//! Here is everything that is recorded during the rendering phases
use anyhow::Result;

use vulkanalia::prelude::v1_0::*;

use super::{
	Descriptors,
	InstanceData,
	Swapchain,
	DepthAttachment,
	ColorAttachment
};

use crate::math::Mat4;
use crate::scene::ECSContext;
use crate::gpu::{
	QueueFamilyIndices,
	Pipeline
};
use crate::resources::{
	Buffers,
	create_command_pool,
	create_command_pools,
	create_setup_command_buffer,
	create_command_buffers
};
use crate::type_safety::{
	ModelId,
	MaterialId,
	MaterialSetId
};

// typing and doc import


/// Instance of a 3D object.
/// 
/// ## Fields
/// 
/// - `model_id` ( [ModelId] ) - ID of the 3D Model inside a [ModelsStorage].
/// - `world_mat` ( [Mat4] ) - 3D model (global) transformation.
/// - `ssbo_offset` ( `u32` ) - SSBO offset of this model.
#[derive(Debug)]
struct EntityInstance {
	model_id:		ModelId,
	world_mat:		Mat4,
	ssbo_offset:	u32,
}

/// A single indexed draw call, resolved from the ECS scene before recording.
///
/// One item is emitted per (model, node, primitive) group. All instances of the
/// same model share a single item, distinguished at draw time by `instance_first`
/// and `instance_count`.
///
/// ## Fields
/// 
/// - `material_set_id` ( [MaterialSetId] ) - Global index. Draw items are sorted on this field to minimise descriptor set rebinds.
/// - `model_id` ( [ModelId] ) - Owning model, used to look the material up in a [ModelsStorage].
/// - `material_id` ( Option<[MaterialId]> ) - Material index local to a owning [ModelGraph], or `None` when the primitive has no material. Used to fill the fragment push constants.
/// - `node_matrix` ( [Mat4] ) - World matrix of the node inside its model, excluding the per-instance entity transform. Used in a vertex shader.
/// - `first_index` ( `u32` ) - Offset of the first index in the shared interleaved buffer, already including both the mesh and primitive offsets.
/// - `vertex_offset` ( `u32` ) - Offset added to every index, locating this mesh's vertices in the shared interleaved buffer.
/// - `index_count` ( `u32` ) -  Number of indices to draw for this primitive.
/// - `instance_first` ( `u32` ) - Index of the first instance in [InstanceBuffer]. Passed as `firstInstance`, which offsets `gl_InstanceIndex` in the shader.
/// - `instance_count` ( `u32` ) - Number of instances sharing this draw.
#[derive(Debug)]
struct DrawItem {
	material_set_id:	MaterialSetId,
	model_id:			ModelId,
	material_id:		Option<MaterialId>,
	node_matrix:		Mat4,
	first_index:		u32,
	vertex_offset:		u32,
	index_count:		u32,
	instance_first:		u32,
	instance_count:		u32,
}

/// Owns the command pools and buffers, and records the scene into them each frame.
///
/// Recording happens in two phases: the ECS scene is first flattened into
/// `draw_list` and `instance_data`, then those are replayed into a single
/// secondary command buffer with minimal state changes.
///
/// The three working vectors are kept as fields and cleared rather than
/// reallocated, so steady-state recording performs no heap allocation.
/// 
/// ## Fields
/// 
/// - `setup_pool` ( [vk::CommandPool] ) - Pool for one-off setup work (buffer copies, image transitions) performed during asset loading.
/// - `frames_pools` ( Vec<[vk::CommandPool]> ) - One pool per swapchain image, reset at the start of the frame that uses it.
/// - `primary_command_buffers` ( Vec<[vk::CommandBuffer]> ) - Primary command buffer per swapchain image, submitted to the graphics queue.
/// - `secondary_command_buffers` ( Vec<[vk::CommandBuffer]> ) - Secondary command buffer per swapchain image, holding every draw of the frame.
/// - `setup_command_buffer` ( [vk::CommandBuffer] ) - Buffer allocated from `setup_pool`, used for transfer and layout transition commands at load time.
/// - `draw_list` ( Vec<[DrawItem]> ) - Draw calls collected for the current frame, sorted by material.
/// - `instance_data` ( Vec<[InstanceData]> ) - Per-instance transforms and skinning offsets, uploaded before recording.
/// - `sorted_entities` ( Vec<[EntityInstance]> ) - Renderable entities sorted by model, so instances of the same model form contiguous ranges.
#[derive(Debug)]
pub struct CommandRecorder {
	pub setup_pool:					vk::CommandPool,
    pub frames_pools:				Vec<vk::CommandPool>,
    pub primary_command_buffers:	Vec<vk::CommandBuffer>,
    pub secondary_command_buffers:	Vec<vk::CommandBuffer>,
    pub setup_command_buffer:		vk::CommandBuffer,

    draw_list:			Vec<DrawItem>,
    instance_data:		Vec<InstanceData>,
    sorted_entities:	Vec<EntityInstance>,
}

impl CommandRecorder {
	pub fn new(
		device: &Device,
		queue_family_indices: &mut QueueFamilyIndices,
		swapchain_images: &[vk::Image],
	) -> Result<Self> {
		let setup_pool = create_command_pool(device, queue_family_indices)?;
		let frames_pools = create_command_pools(device, queue_family_indices, swapchain_images.len())?;
		let setup_command_buffer = create_setup_command_buffer(device, setup_pool)?;
		let (primary_command_buffers, secondary_command_buffers) = create_command_buffers(device, &frames_pools)?;

		Ok(Self {
			setup_pool,
			frames_pools,
			primary_command_buffers,
			secondary_command_buffers,
			setup_command_buffer,

			draw_list:			Vec::new(),
			instance_data:		Vec::new(),
			sorted_entities:	Vec::new()
		})
	}

	#[rustfmt::skip]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn destroy(&mut self, device: &Device) {
		self.frames_pools.iter().for_each(|c| device.destroy_command_pool(*c, None));
		device.free_command_buffers(self.setup_pool, &[self.setup_command_buffer]);
		device.destroy_command_pool(self.setup_pool, None);
	}

	pub fn record(
		&mut self,
        device:				&Device,
        ecs_context:		&mut ECSContext,
        graphic_pipeline:	&Pipeline,
        buffers:			&Buffers,
        swapchain:			&Swapchain,
        color_attachment:	&ColorAttachment,
        depth_attachment:	&DepthAttachment,
        descriptors:		&Descriptors,
        msaa_samples:		vk::SampleCountFlags,
        image_index:		usize,
        frame_index:		usize,
	) -> Result<()> {

	}
}