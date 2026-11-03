use anyhow::Result;

use vulkanalia::prelude::v1_0::*;

use super::{ShaderStagesBuilder, create_shader_module};
use crate::scene::Vertex;

#[derive(Debug)]
pub struct Pipeline {
    pub vk_pipeline: vk::Pipeline,
    pub vk_layout: vk::PipelineLayout,
}

impl Pipeline {
	/// Returns a builder for a graphics pipeline.
    #[inline]
    pub fn graphics_builder<'a>(
        shaders: &'a [(&'a [u8], vk::ShaderStageFlags)],
		set_layouts: &'a [vk::DescriptorSetLayout],
        color_format: vk::Format,
        depth_format: vk::Format,
    ) -> GraphicsPipelineBuilder<'a> {
        GraphicsPipelineBuilder::new(shaders, set_layouts, color_format, depth_format)
    }

	/// Create a compute pipeline
	pub fn new_compute(
		device: &Device,
		compute_shader: &[u8],
		set_layouts: &[vk::DescriptorSetLayout],
	) -> Result<Self> {
		let shader_module = create_shader_module(device, compute_shader)?;

        let stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(b"main\0");

		let layout_info = vk::PipelineLayoutCreateInfo::builder()
			.set_layouts(set_layouts);
		let vk_layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };

		let info = vk::ComputePipelineCreateInfo::builder()
            .stage(stage)
            .layout(vk_layout);

        let vk_pipeline = unsafe { device
			.create_compute_pipelines(vk::PipelineCache::null(), &[info], None)?
			.0[0]
		};

		unsafe { device.destroy_shader_module(shader_module, None) };

        Ok(Self { vk_pipeline, vk_layout })
	}

	/// Destroys the pipeline and its layout.
    ///
    /// ## Safety
    ///
    /// The caller must ensure the device is idle: the pipeline may still be bound by
    /// a command buffer awaiting execution.
	#[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn destroy(&self, device: &Device) {
        device.destroy_pipeline(self.vk_pipeline, None);
        device.destroy_pipeline_layout(self.vk_layout, None);
    }
}

/// Builder for the engine's graphics pipelines.
///
/// ## Fields
///
/// - `shaders` ( &\[(&\[`u8`], [vk::ShaderStageFlags])] ) - SPIR-V bytecode paired with the stage it feeds. The modules are created, used, and destroyed inside [build].
/// - `set_layouts` ( &\[[vk::DescriptorSetLayout]] ) - Descriptor set layouts in binding order: position 0 becomes set 0, and so on.
/// - `push_constant_ranges` ( &\[[vk::PushConstantRange]] ) - Push constant ranges per stage. Empty by default, since not every pass uses them.
/// - `samples` ( [vk::SampleCountFlags] ) - Multisampling count. Must match the colour attachment the pass renders into.
/// - `color_format` ( [vk::Format] ) - Format of the colour attachment, baked into the pipeline by dynamic rendering. Must match the swapchain.
/// - `depth_format` ( [vk::Format] ) - Format of the depth attachment, likewise baked in.
/// - `depth_compare_op` ( [vk::CompareOp] ) - Depth test operator. `LESS` for geometry, `LESS_OR_EQUAL` for a far-plane pass, whose fragments sit exactly at the cleared depth value.
/// - `cull_mode` ( [vk::CullModeFlags] ) - Which faces the rasterizer discards. `BACK` for closed meshes, where back-facing triangles are hidden anyway; `NONE` for passes whose geometry has no meaningful winding.
/// - `min_sample_shading` ( Option<`f32`> ) - Minimum fraction of samples the fragment shader runs for, smoothing aliasing inside triangles rather than only at their edges. `None` shades once per pixel, which is enough for a background pass with no high-frequency detail.
/// - `read_vertex_input` ( `bool` ) - Whether the pipeline reads [Vertex] from a bound buffer. False for passes that generate their geometry from `gl_VertexIndex`.
/// - `depth_write` ( `bool` ) - Whether fragments write to the depth buffer. False for a background pass, which must not occlude anything drawn after it.
/// - `alpha_blending_enabled` ( `bool` ) - Whether fragments blend with what is already in the attachment, weighted by source alpha. Off for opaque passes, which overwrite instead.
pub struct GraphicsPipelineBuilder<'a> {
	shaders:				&'a [(&'a [u8], vk::ShaderStageFlags)],
	set_layouts:			&'a [vk::DescriptorSetLayout],
	push_constant_ranges:	&'a [vk::PushConstantRange],
	samples:				vk::SampleCountFlags,
	color_format:			vk::Format,
	depth_format:			vk::Format,
	depth_compare_op:		vk::CompareOp,
	cull_mode:				vk::CullModeFlags,
	min_sample_shading:		Option<f32>,
	read_vertex_input:		bool,
	depth_write:			bool,
	alpha_blending_enabled:	bool,
}

impl<'a> GraphicsPipelineBuilder<'a> {
	pub fn new(
		shaders:		&'a [(&'a [u8], vk::ShaderStageFlags)],
		set_layouts:	&'a [vk::DescriptorSetLayout],
		color_format:	vk::Format,
		depth_format:	vk::Format,
	) -> Self {
		Self {
			shaders,
			set_layouts,
			color_format,
			depth_format,
			push_constant_ranges: &[],
			samples: vk::SampleCountFlags::_1,
			depth_compare_op: vk::CompareOp::LESS,
			cull_mode: vk::CullModeFlags::BACK,
			min_sample_shading: None,
			read_vertex_input: true,
			depth_write: true,
			alpha_blending_enabled: false,
		}
	}

	#[inline]
	pub fn push_constants(mut self, ranges: &'a [vk::PushConstantRange]) -> Self {
		self.push_constant_ranges = ranges;
		self
	}

	#[inline]
	pub fn samples(mut self, samples: vk::SampleCountFlags) -> Self {
		self.samples = samples;
		self
	}

	/// Run the fragment shader multiple times per pixel, smoothing aliasing inside
    /// triangles rather than only at their edges. The fraction is the minimum
    /// portion of samples shaded: closer to one is smoother and more expensive.
    #[inline]
    pub fn min_sample_shading(mut self, min: f32) -> Self {
        self.min_sample_shading = Some(min);
        self
    }

	/// false => vertices are generated in the shader from `gl_VertexIndex`, so no buffer
    /// is bound and the vertex input state stays empty.
	#[inline]
	pub fn read_vertex_input(mut self, activate: bool) -> Self {
		self.read_vertex_input = activate;
		self
	}

	/// Test against depth without writing to it, as a background pass must.
	#[inline]
	pub fn depth_read_only(mut self) -> Self {
		self.depth_write = false;
		self.depth_compare_op = vk::CompareOp::LESS_OR_EQUAL;
		self
	}

	/// Which faces the rasterizer discards. A fullscreen pass has no meaningful
    /// winding, so it disables culling rather than matching the convention.
    #[inline]
    pub fn cull_mode(mut self, mode: vk::CullModeFlags) -> Self {
        self.cull_mode = mode;
        self
    }

	/// Alpha blending, as transparent materials need. Off by default, since an
    /// opaque pass writes straight to the attachment.
    #[inline]
    pub fn alpha_blending_enabled(mut self, activate: bool) -> Self {
        self.alpha_blending_enabled = activate;
        self
    }

	pub fn build(self, device: &Device) -> Result<Pipeline> {
		let stages_builder = ShaderStagesBuilder::new(device, self.shaders)?;
		let stages = stages_builder.build();

		// 1 - Vertex Input State
		let binding_descriptions = &[Vertex::binding_description()];
		let attribute_descriptions = Vertex::attribute_descriptions();
		let vertex_input_state = if self.read_vertex_input {
			vk::PipelineVertexInputStateCreateInfo::builder()
				.vertex_binding_descriptions(binding_descriptions)
				.vertex_attribute_descriptions(&attribute_descriptions)
		} else {
			vk::PipelineVertexInputStateCreateInfo::builder()
		};
 
		// 2 - Pipeline State
		let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
			.topology(vk::PrimitiveTopology::TRIANGLE_LIST)
			.primitive_restart_enable(false);

		let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
			.viewport_count(1)
			.scissor_count(1);

		let dynamic_state = vk::PipelineDynamicStateCreateInfo::builder()
    		.dynamic_states(&[
				vk::DynamicState::VIEWPORT,
				vk::DynamicState::SCISSOR,
			]);

		let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
			.depth_clamp_enable(false)
			.rasterizer_discard_enable(false)
			.polygon_mode(vk::PolygonMode::FILL)
			.line_width(1.0)
			.cull_mode(self.cull_mode)
			.front_face(vk::FrontFace::COUNTER_CLOCKWISE)
			.depth_bias_enable(false);

		let multisample_state = match self.min_sample_shading {
			Some(min) => vk::PipelineMultisampleStateCreateInfo::builder()
				.sample_shading_enable(true)
				.min_sample_shading(min),
			None => vk::PipelineMultisampleStateCreateInfo::builder()
				.sample_shading_enable(false),
		}
			.rasterization_samples(self.samples);

		let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder()
			.depth_test_enable(true)
			.depth_write_enable(self.depth_write)
			.depth_compare_op(self.depth_compare_op)
			.depth_bounds_test_enable(false)
			.stencil_test_enable(false);
		
		// 3 - Color Blend State
		let attachment = if self.alpha_blending_enabled {
			vk::PipelineColorBlendAttachmentState::builder()
				.color_write_mask(vk::ColorComponentFlags::all())
				.blend_enable(true)
				.src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
				.dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
				.color_blend_op(vk::BlendOp::ADD)
				.src_alpha_blend_factor(vk::BlendFactor::ONE)
				.dst_alpha_blend_factor(vk::BlendFactor::ZERO)
				.alpha_blend_op(vk::BlendOp::ADD)
		} else {
			vk::PipelineColorBlendAttachmentState::builder()
				.color_write_mask(vk::ColorComponentFlags::all())
				.blend_enable(false)
		};
		let attachments = &[attachment];

		let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
			.logic_op_enable(false)
			.logic_op(vk::LogicOp::COPY)
			.attachments(attachments)
			.blend_constants([0.0, 0.0, 0.0, 0.0]);

		// 4 - Vulkan Layout
		let layout_info = vk::PipelineLayoutCreateInfo::builder()
			.push_constant_ranges(self.push_constant_ranges)
			.set_layouts(self.set_layouts);

		let vk_layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };

		// 5 - Dynamic Rendering
		let color_formats = &[self.color_format];

		let mut pipeline_rendering_info = vk::PipelineRenderingCreateInfoKHR::builder()
			.color_attachment_formats(color_formats)
			.depth_attachment_format(self.depth_format);

		// 6 - Create
		let info = vk::GraphicsPipelineCreateInfo::builder()
			.stages(&stages)
			.vertex_input_state(&vertex_input_state)
			.input_assembly_state(&input_assembly_state)
			.viewport_state(&viewport_state)
			.rasterization_state(&rasterization_state)
			.multisample_state(&multisample_state)
			.depth_stencil_state(&depth_stencil_state)
			.color_blend_state(&color_blend_state)
			.dynamic_state(&dynamic_state)
			.layout(vk_layout)
			.push_next(&mut pipeline_rendering_info);

		let vk_pipeline = unsafe { device
			.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)?
			.0[0]
		};

		// 7 - Cleanup
		unsafe { stages_builder.destroy(device) };

		Ok(Pipeline { vk_pipeline, vk_layout })
	}

}