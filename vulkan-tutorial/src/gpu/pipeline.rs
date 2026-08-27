use anyhow::Result;

use vulkanalia::prelude::v1_0::*;

use super::ShaderStagesBuilder;
use crate::scene::Vertex;
use crate::render::PushConstants;

#[derive(Debug)]
pub struct Pipeline {
    pub vk_pipeline: vk::Pipeline,
    pub vk_layout: vk::PipelineLayout,
}

impl Pipeline {
    pub fn new(
        device:					&Device,
        shaders:				&[(&[u8], vk::ShaderStageFlags)],
        swapchain_extent:		vk::Extent2D,
        swapchain_format:		vk::Format,
        depth_format:			vk::Format,
        msaa_samples:			vk::SampleCountFlags,
        set_layouts:			&[vk::DescriptorSetLayout]
    ) -> Result<Self> {
		let stages_builder = ShaderStagesBuilder::new(device, shaders)?;
		let stages = stages_builder.build();

		// Vertex Input State
		let binding_descriptions = &[Vertex::binding_description()];
		let attribute_descriptions = Vertex::attribute_descriptions();
		let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
			.vertex_binding_descriptions(binding_descriptions)
			.vertex_attribute_descriptions(&attribute_descriptions);

		// Input Assembly State
		let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
			.topology(vk::PrimitiveTopology::TRIANGLE_LIST)
			.primitive_restart_enable(false);


		// Viewport State
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

		let viewports = &[viewport];
		let scissors = &[scissor];
		let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
			.viewports(viewports)
			.scissors(scissors);

		// Rasterization State
		let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
			.depth_clamp_enable(false)
			.rasterizer_discard_enable(false)
			.polygon_mode(vk::PolygonMode::FILL)
			.line_width(1.0)
			.cull_mode(vk::CullModeFlags::BACK)
			.front_face(vk::FrontFace::COUNTER_CLOCKWISE)
			.depth_bias_enable(false);

		// Multisample State
		let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
			.sample_shading_enable(true)
			// minimum fraction for shading, closer to one is smoother.
			.min_sample_shading(0.2)
			.rasterization_samples(msaa_samples);

		// Depth Stencil State
		let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder()
			.depth_test_enable(true)
			.depth_write_enable(true)
			.depth_compare_op(vk::CompareOp::LESS)
			.depth_bounds_test_enable(false)
			.stencil_test_enable(false);

		// Color Blend State
		let attachment = vk::PipelineColorBlendAttachmentState::builder()
			.color_write_mask(vk::ColorComponentFlags::all())
			.blend_enable(true)
			.src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
			.dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
			.color_blend_op(vk::BlendOp::ADD)
			.src_alpha_blend_factor(vk::BlendFactor::ONE)
			.dst_alpha_blend_factor(vk::BlendFactor::ZERO)
			.alpha_blend_op(vk::BlendOp::ADD);

		let attachments = &[attachment];
		let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
			.logic_op_enable(false)
			.logic_op(vk::LogicOp::COPY)
			.attachments(attachments)
			.blend_constants([0.0, 0.0, 0.0, 0.0]);

		// Push Constant
		// TODO: this should change to take into account for the n shaders
		let frag_offset = PushConstants::get_frag_offset();
		let frag_size = PushConstants::get_frag_size();

		let vert_push_constant_range = vk::PushConstantRange::builder()
			.stage_flags(vk::ShaderStageFlags::VERTEX)
			.offset(0)
			.size(frag_offset);

		let frag_push_constant_range = vk::PushConstantRange::builder()
			.stage_flags(vk::ShaderStageFlags::FRAGMENT)
			.offset(frag_offset)
			.size(frag_size);

		// Layout
		let push_constant_ranges = &[vert_push_constant_range, frag_push_constant_range];
		let layout_info = vk::PipelineLayoutCreateInfo::builder()
			.push_constant_ranges(push_constant_ranges)
			.set_layouts(set_layouts);

		let vk_layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };

		// dynamic rendering
		let color_formats = &[swapchain_format];

		let mut pipeline_rendering_info = vk::PipelineRenderingCreateInfoKHR::builder()
			.color_attachment_formats(color_formats)
			.depth_attachment_format(depth_format);

		// Create
		let info = vk::GraphicsPipelineCreateInfo::builder()
			.stages(&stages)
			.vertex_input_state(&vertex_input_state)
			.input_assembly_state(&input_assembly_state)
			.viewport_state(&viewport_state)
			.rasterization_state(&rasterization_state)
			.multisample_state(&multisample_state)
			.depth_stencil_state(&depth_stencil_state)
			.color_blend_state(&color_blend_state)
			.layout(vk_layout)
			.push_next(&mut pipeline_rendering_info);

		let vk_pipeline = unsafe { device
			.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)?
			.0[0]
		};

		// Cleanup
		unsafe { stages_builder.destroy(device) };

		Ok(Self { vk_pipeline, vk_layout })
    }

	#[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn destroy(&self, device: &Device) {
        device.destroy_pipeline(self.vk_pipeline, None);
        device.destroy_pipeline_layout(self.vk_layout, None);
    }
}
