//! Here is everything that is recorded during the rendering phases
use anyhow::{Result, anyhow};

use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::{
	KhrSynchronization2ExtensionDeviceCommands,
	KhrDynamicRenderingExtensionDeviceCommands
};

use super::{
	Descriptors,
	InstanceData,
	Swapchain,
	DepthAttachment,
	ColorAttachment,
	PushConstants
};

use crate::math::Mat4;
use crate::scene::{
	ECSContext,
	ModelsStorage,
	InstanceBuffer,
};
use crate::gpu::{
	QueueFamilyIndices,
	Pipeline
};
use crate::resources::{
	GeometryBuffer,
	create_command_pool,
	create_command_pools,
	create_setup_command_buffer,
	create_command_buffers
};
use crate::type_safety::{
	NodeId,
	ModelId,
	MaterialId,
	MaterialSetId,
};

use crate::constants::{FRAME_STRIDE};

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

    // Kept as fields and cleared rather than reallocated, so steady-state recording performs no heap allocation
    draw_list:			Vec<DrawItem>,
    instance_data:		Vec<InstanceData>,
    sorted_entities:	Vec<EntityInstance>,
}

impl CommandRecorder {
	pub fn new(
		device: &Device,
		queue_family_indices: &QueueFamilyIndices,
		swapchain_images_count: usize,
	) -> Result<Self> {
		let setup_pool = create_command_pool(device, queue_family_indices)?;
		let frames_pools = create_command_pools(device, queue_family_indices, swapchain_images_count)?;
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

    /// Destroys the command pools, which implicitly frees every command buffer
    /// allocated from them.
    ///
    /// ## Safety
    ///
    /// The caller must ensure the device is idle: destroying a pool whose buffers are
    /// still executing, or awaiting execution, is undefined behaviour.
	#[rustfmt::skip]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn destroy(&self, device: &Device) {
		self.frames_pools.iter().for_each(|c| device.destroy_command_pool(*c, None));
		device.free_command_buffers(self.setup_pool, &[self.setup_command_buffer]);
		device.destroy_command_pool(self.setup_pool, None);
	}

    /// Records the primary command buffer for one frame.
    ///
    /// Resets the frame pool, opens a dynamic rendering pass targeting the given
    /// attachments.
    ///
    /// The recorded buffer is stored in `command_buffers[image_index]` and submitted
    /// by the caller.
    ///
    /// ## Arguments
    ///
    /// - `device` ( &[Device] ) - The Vulkan device.
    /// - `ecs_context` ( &mut [ECSContext] ) - Scene to draw. Queried for renderable entities and for the [ModelsStorage] and [InstanceBuffer] resources.
    /// - `graphic_pipeline` ( &[Pipeline] ) - Pipeline and layout bound for every draw of the frame.
    /// - `buffers` ( &[Buffers] ) - Shared geometry buffer, mesh offsets and the uniform buffer of this image.
    /// - `swapchain` ( &[Swapchain] ) - Provides the render extent, the color format and the image view resolved into.
    /// - `color_attachment` ( &[ColorAttachment] ) - Multisampled color target, resolved into the swapchain image.
    /// - `depth_attachment` ( &[DepthAttachment] ) - Depth target and its format.
    /// - `descriptors` ( &[Descriptors] ) - Descriptor sets bound during recording.
    /// - `msaa_samples` ( [vk::SampleCountFlags] ) - Sample count, which must match the one the pipeline was created with.
    /// - `image_index` ( `usize` ) - Swapchain image being rendered into, selecting the pool and buffers to use.
    /// - `frame_index` ( `usize` ) - Frame in flight, selecting which region of the skinning and instance buffers to read.
	pub fn record(
		&mut self,
        device:				&Device,
        ecs_context:		&mut ECSContext,
        graphic_pipeline:	&Pipeline,
        geometry_buffer:	&GeometryBuffer,
        swapchain:			&Swapchain,
        color_attachment:	&ColorAttachment,
        depth_attachment:	&DepthAttachment,
        descriptors:		&Descriptors,
        msaa_samples:		vk::SampleCountFlags,
        image_index:		usize,
        frame_index:		usize,
	) -> Result<()> {
		// Pool
        let command_pool = self.frames_pools[image_index];
        unsafe { device.reset_command_pool(command_pool, vk::CommandPoolResetFlags::empty())? };

        // Commands
        let command_buffer = self.primary_command_buffers[image_index];

        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe { device.begin_command_buffer(command_buffer, &info)? };

        let color_clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };

        let depth_clear_value = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        };

        let render_area = vk::Rect2D::builder()
            .offset(vk::Offset2D::default())
            .extent(swapchain.vk_extent);

        let color_rendering_info = vk::RenderingAttachmentInfo::builder()
            .image_view(color_attachment.attachment().vk_image_view)
            .image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .resolve_mode(vk::ResolveModeFlags::AVERAGE)
            .resolve_image_view(swapchain.vk_image_views[image_index])
            .resolve_image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
            .clear_value(color_clear_value);

        let depth_rendering_info = vk::RenderingAttachmentInfo::builder()
            .image_view(depth_attachment.attachment.vk_image_view)
            .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(depth_clear_value);

        let rendering_info = vk::RenderingInfo::builder()
            .flags(vk::RenderingFlagsKHR::CONTENTS_SECONDARY_COMMAND_BUFFERS)
            .render_area(render_area)
            .layer_count(1)
            .color_attachments(std::slice::from_ref(&color_rendering_info))
            .depth_attachment(&depth_rendering_info);

        Self::transition_for_render(
            device,
            swapchain.vk_images[image_index],
            command_buffer
        );

        unsafe { device.cmd_begin_rendering_khr(command_buffer, &rendering_info); }

        let secondary_command_buffer = self.record_secondary_command_buffer(
            device,
            ecs_context,
            graphic_pipeline,
            geometry_buffer,
            &[swapchain.vk_format],
            depth_attachment,
            descriptors,
            msaa_samples,
            swapchain.vk_extent,
            image_index,
            frame_index
        )?;

        unsafe { 
            device.cmd_execute_commands(command_buffer, &[secondary_command_buffer]);
            device.cmd_end_rendering_khr(command_buffer);
        };

        Self::transition_for_present(
            device,
            swapchain.vk_images[image_index],
            command_buffer
        );

        unsafe { device.end_command_buffer(command_buffer)? };

        Ok(())
	}

	/// record draws inside a unique secondary command buffer
	fn record_secondary_command_buffer(
        &mut self,
        device:				&Device,
        ecs_context:		&mut ECSContext,
        graphic_pipeline:	&Pipeline,
        geometry_buffer:	&GeometryBuffer,
        swapchain_formats:	&[vk::Format],
        depth_attachment:	&DepthAttachment,
        descriptors:		&Descriptors,
        msaa_samples:		vk::SampleCountFlags,
        swapchain_extent:   vk::Extent2D,
        image_index:		usize,
        frame_index:		usize
    ) -> Result<vk::CommandBuffer> {
        self.instance_data.clear();
        self.draw_list.clear();
		self.sorted_entities.clear();

        let models = ecs_context.world.get_resource::<ModelsStorage>()
            .ok_or_else(|| anyhow!("ModelsStorage not found"))?;
		// Step 1 - Draw Sorting
		// 1.a - gather entities, sorted by model
		self.sorted_entities.extend(
			ecs_context.cached_renderable_query
				.iter(&ecs_context.world)
				.map(|(global, mesh_handle, skeleton)| {
					let ssbo_offset = skeleton
						.map(|s| s.ssbo_offset + (frame_index * FRAME_STRIDE) as u32)
						.unwrap_or(PushConstants::NO_SKIN);

					EntityInstance {
						model_id: mesh_handle.model_id,
						world_mat: global.0,
						ssbo_offset
					}
				})	
		);

		self.sorted_entities.sort_unstable_by_key(|e| e.model_id.0);

		// 1.b - instance data + model ranges
		let mut model_ranges: Vec<(ModelId, u32, u32)> = Vec::new();

		for entity in &self.sorted_entities {
			match model_ranges.last_mut() {
				Some((last_id, _, count)) if *last_id == entity.model_id => *count += 1,
				_ => model_ranges.push((entity.model_id, self.instance_data.len() as u32, 1)),
			}

			self.instance_data.push(
				InstanceData {
					model: entity.world_mat,
					ssbo_offset: entity.ssbo_offset,
					_padding: [0; 3],
				}
			)
		}

		// 1.c - one DrawItem by (model, node, primitive)
		for (model_id, instance_first, instance_count) in &model_ranges {
			let model = models.get_model(*model_id);
			let material_offset = models.get_material_offset(*model_id);

			for (i, node) in model.graph.iter().enumerate() {
				let Some(mesh) = &node.value.mesh else { continue };

				let node_id = NodeId(i);
				let node_matrix = model.get_global_matrix_of(node_id);
				let mesh_offset = geometry_buffer.get_mesh_offset(*model_id, node_id);

				for primitive in &mesh.primitives {
					self.draw_list.push(DrawItem {
						material_set_id: primitive.material_id
							.map(|id| id.to_set_id(material_offset))
							.unwrap_or(MaterialSetId(0)),
						model_id: *model_id,
						material_id: primitive.material_id, 
						node_matrix,
						first_index: mesh_offset.first_index + primitive.first_index,
						vertex_offset: mesh_offset.vertex_offset,
						index_count: primitive.index_count,
						instance_first: *instance_first,
						instance_count: *instance_count
					});
				}
			}
		}

        self.draw_list.sort_unstable_by_key(|draw_item| draw_item.material_set_id);

		// 1.d - Upload instance data
		let instance_buffer = ecs_context.world.get_resource::<InstanceBuffer>()
			.ok_or_else(|| anyhow!("InstanceBuffer not found in world"))?;
		instance_buffer.write(frame_index, &self.instance_data);

        // Step 2 - Draw Binding with state tracking
        let command_buffer = self.secondary_command_buffers[image_index];

        let mut inheritance_rendering_info = vk::CommandBufferInheritanceRenderingInfo::builder()
            .color_attachment_formats(swapchain_formats)
            .depth_attachment_format(depth_attachment.vk_format)
            .rasterization_samples(msaa_samples);

        let inheritance_info = vk::CommandBufferInheritanceInfo::builder()
            .push_next(&mut inheritance_rendering_info);

        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE)
            .inheritance_info(&inheritance_info);


        unsafe {
            device.begin_command_buffer(command_buffer, &info)?;

            let viewport = vk::Viewport::builder()
                .x(0.0)
                .y(0.0)
                .width(swapchain_extent.width as f32)
                .height(swapchain_extent.height as f32)
                .min_depth(0.0)
                .max_depth(1.0);

            let scissor = vk::Rect2D::builder()
                .offset(vk::Offset2D { x: 0, y: 0 })
                .extent(swapchain_extent);

            device.cmd_set_viewport(command_buffer, 0, &[viewport]);
            device.cmd_set_scissor(command_buffer, 0, &[scissor]);

            device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, graphic_pipeline.vk_pipeline);
            device.cmd_bind_vertex_buffers(command_buffer, 0, &[geometry_buffer.vk_buffer], &[0]);
            device.cmd_bind_index_buffer(command_buffer, geometry_buffer.vk_buffer, geometry_buffer.interleaved_offset, vk::IndexType::UINT32);
        
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                graphic_pipeline.vk_layout,
                0,
                &[descriptors.global_descriptor_sets[image_index]],
                &[]
            );

			// skin binding
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                graphic_pipeline.vk_layout,
                2,
                &[descriptors.skinning_descriptor_set],
                &[]
            );

			// entity binding
			device.cmd_bind_descriptor_sets(
				command_buffer,
				vk::PipelineBindPoint::GRAPHICS,
				graphic_pipeline.vk_layout,
				3,
				&[descriptors.instance_descriptor_set],
				&[]
			);
        }

        let mut last_material: Option<MaterialSetId> = None;

        for item in &self.draw_list {
            // state tracking
            if last_material != Some(item.material_set_id) {
                last_material = Some(item.material_set_id);

                unsafe {
                    // materials binding
                    device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        graphic_pipeline.vk_layout,
                        1,
                        &[descriptors.material_descriptor_sets[item.material_set_id.0]],
                        &[]
                    );
                }
            }

            let material = item.material_id.map(|material_id| models.get_model(item.model_id)
                .get_material(material_id));

            let push_constant = PushConstants::new(item.node_matrix, material);

            unsafe {
                let push_bytes = std::slice::from_raw_parts(
                    &push_constant as *const PushConstants as *const u8,
                    size_of::<PushConstants>()
                );
                
                let frag_offset = PushConstants::get_frag_offset();

                device.cmd_push_constants(
                    command_buffer,
                    graphic_pipeline.vk_layout,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    &push_bytes[..frag_offset as usize]
                );
                device.cmd_push_constants(
                    command_buffer,
                    graphic_pipeline.vk_layout,
                    vk::ShaderStageFlags::FRAGMENT,
                    frag_offset,
                    &push_bytes[frag_offset as usize..],
                );

                device.cmd_draw_indexed(
                    command_buffer,
                    item.index_count,
                    item.instance_count,
                    item.first_index,
                    item.vertex_offset as i32,
                    item.instance_first
                );
            }
        } 

        unsafe { device.end_command_buffer(command_buffer)?; }
        Ok(command_buffer)
    }

	fn transition_for_render(
        device: &Device,
        swapchain_image: vk::Image,
        command_buffer: vk::CommandBuffer,
    ) {
        let barrier = vk::ImageMemoryBarrier2::builder()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .src_access_mask(vk::AccessFlags2::empty())
            .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
            .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE) // TODO: if blending then need to access READ aswell.
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .image(swapchain_image)
            .subresource_range(vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .base_array_layer(0)
                .layer_count(1)
                .level_count(1)
                .build()
            );

        let barriers = [barrier];
        let dependency_info = vk::DependencyInfo::builder()
            .image_memory_barriers(&barriers);

        unsafe { device.cmd_pipeline_barrier2_khr(command_buffer, &dependency_info) };
    }

	fn transition_for_present(
        device: &Device,
        swapchain_image: vk::Image,
        command_buffer: vk::CommandBuffer,
    ) {
        let barrier = vk::ImageMemoryBarrier2::builder()
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags2::empty())
            .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
            .image(swapchain_image)
            .subresource_range(vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .base_array_layer(0)
                .layer_count(1)
                .level_count(1)
                .build()
            );
        
        let barriers = [barrier];
        let dependency_info = vk::DependencyInfo::builder()
            .image_memory_barriers(&barriers);

        unsafe { device.cmd_pipeline_barrier2_khr(command_buffer, &dependency_info) };
    }

    /// Rebuilds the per-frame command pools and buffers for a new swapchain image count.
    ///
    /// Only needed when a swapchain recreation changes the number of images, since the
    /// pools and buffers are otherwise reset and reused every frame. Destroying the old
    /// pools implicitly frees the command buffers allocated from them.
    ///
    /// ## Arguments
    ///
    /// - `device` ( &[Device] ) - The Vulkan device.
    /// - `queue_family_indices` ( &[QueueFamilyIndices] ) - Source of the graphics family index the new pools are created on.
    /// - `swapchain_images_count` ( `usize` ) - New image count, and therefore the number of pools and buffers to allocate.
    ///
    /// ## Safety
    ///
    /// The caller must have waited on device idle: the pools being destroyed may still
    /// hold buffers submitted to the queue.
    pub fn resize_frame_resources(
        &mut self,
        device: &Device,
        queue_family_indices: &QueueFamilyIndices,
        swapchain_images_count: usize
    ) -> Result<()> {
        unsafe {
            self.frames_pools.iter().for_each(|pool| device.destroy_command_pool(*pool, None));
        }

        self.frames_pools = create_command_pools(device, queue_family_indices, swapchain_images_count)?;
        let (primary, secondary) = create_command_buffers(device, &self.frames_pools)?;
        self.primary_command_buffers = primary;
        self.secondary_command_buffers = secondary;

        Ok(())
    }
}