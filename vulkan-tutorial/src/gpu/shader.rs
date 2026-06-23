use anyhow::Result;

use vulkanalia::bytecode::Bytecode;
use vulkanalia::prelude::v1_0::*;

use crate::scene::Vertex;

//===========================================
// Descriptor Set
//===========================================

/// create **UBO** descriptor set layout.
pub fn create_global_descriptor_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout> {
    // Base-color binding
    let ubo_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::VERTEX);

    let bindings = &[ubo_binding];
    let info = vk::DescriptorSetLayoutCreateInfo::builder()
        .bindings(bindings);

    let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&info, None)? };

    Ok(descriptor_set_layout)
}

/// create **skin** descriptor set layout.
pub fn create_skin_descriptor_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout> {
    // Animation joint binding
    let joint_matrices_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::VERTEX);

    let bindings = &[joint_matrices_binding];
    let info = vk::DescriptorSetLayoutCreateInfo::builder()
        .bindings(bindings);

    let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&info, None)? };

    Ok(descriptor_set_layout)
}

/// create material set layout in this order:
/// base color, metallic-roughness, normal map, occlusion map, emissive map.
pub fn create_material_descriptor_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout> {
    let bindings: Vec<vk::DescriptorSetLayoutBinding>  = (0..5u32)
        .map(|i| *vk::DescriptorSetLayoutBinding::builder()
            .binding(i)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        ).collect();
    
    let info = vk::DescriptorSetLayoutCreateInfo::builder()
        .bindings(&bindings);

    let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&info, None)? };
    Ok(descriptor_set_layout)

}

//===========================================
// Shader pipeline and module
//===========================================

fn create_shader_module(device: &Device, bytecode: &[u8]) -> Result<vk::ShaderModule> {
    let bytecode = Bytecode::new(bytecode).unwrap();
    let info = vk::ShaderModuleCreateInfo::builder()
        .code(bytecode.code())
        .code_size(bytecode.code_size());
    Ok( unsafe { device.create_shader_module(&info, None)? })
}

pub fn create_pipeline(
    device: &Device,
    vert: &[u8],
    frag: &[u8],
    swapchain_extent: vk::Extent2D,
    swapchain_format: vk::Format,
    depth_format: vk::Format,
    msaa_samples: vk::SampleCountFlags,
    global_set_layout: vk::DescriptorSetLayout,
    material_set_layout: vk::DescriptorSetLayout,
    skin_set_layout: vk::DescriptorSetLayout,
) -> Result<(vk::Pipeline, vk::PipelineLayout)> {
    // Stages
    let vert_shader_module = create_shader_module(device, &vert[..])?;
    let frag_shader_module = create_shader_module(device, &frag[..])?;

    let vert_stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(vert_shader_module)
        .name(b"main\0");

    let frag_stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::FRAGMENT)
        .module(frag_shader_module)
        .name(b"main\0");

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
        .src_alpha_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .alpha_blend_op(vk::BlendOp::ADD);

    let attachments = &[attachment];
    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
        .logic_op_enable(false)
        .logic_op(vk::LogicOp::COPY)
        .attachments(attachments)
        .blend_constants([0.0, 0.0, 0.0, 0.0]);

    // Constant Push
    let vert_push_constant_range = vk::PushConstantRange::builder()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .offset(0)
        .size(68 /* Mat4 (16 x 4 bytes float) + 4 bytes int*/);

    let frag_push_constant_range = vk::PushConstantRange::builder()
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        .offset(68)
        .size(4 /* 4 bytes float */);

    // Layout
    let set_layouts = &[global_set_layout, material_set_layout, skin_set_layout];
    let push_constant_ranges = &[vert_push_constant_range, frag_push_constant_range];
    let layout_info = vk::PipelineLayoutCreateInfo::builder()
        .push_constant_ranges(push_constant_ranges)
        .set_layouts(set_layouts);

    let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };

    // dynamic rendering
    let color_formats = &[swapchain_format];

    let mut pipeline_rendering_info = vk::PipelineRenderingCreateInfoKHR::builder()
        .color_attachment_formats(color_formats)
        .depth_attachment_format(depth_format);

    // Create
    let stages = &[vert_stage, frag_stage];
    let info = vk::GraphicsPipelineCreateInfo::builder()
        .stages(stages)
        .vertex_input_state(&vertex_input_state)
        .input_assembly_state(&input_assembly_state)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization_state)
        .multisample_state(&multisample_state)
        .depth_stencil_state(&depth_stencil_state)
        .color_blend_state(&color_blend_state)
        .layout(pipeline_layout)
        .push_next(&mut pipeline_rendering_info);

    let pipeline = unsafe { device
        .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)?
        .0[0]
    };

    // Cleanup
    unsafe {
        device.destroy_shader_module(vert_shader_module, None);
        device.destroy_shader_module(frag_shader_module, None);
    }

    Ok((pipeline, pipeline_layout))
}