use anyhow::Result;

use vulkanalia::prelude::v1_0::*;

use crate::render::get_depth_format;

//=========================================================
// Render Pass
//=========================================================

pub fn create_render_pass(
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    swapchain_format: vk::Format,
    msaa_samples: vk::SampleCountFlags,
) -> Result<vk::RenderPass> {
    // Attachments
    let color_attachment = vk::AttachmentDescription::builder()
        .format(swapchain_format)
        .samples(msaa_samples)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    let depth_stencil_attachment = vk::AttachmentDescription::builder()
        .format(get_depth_format(instance, physical_device)?)
        .samples(msaa_samples)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::DONT_CARE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

	let color_resolve_attachment = vk::AttachmentDescription::builder()
		.format(swapchain_format)
		.samples(vk::SampleCountFlags::_1)
		.load_op(vk::AttachmentLoadOp::DONT_CARE)
		.store_op(vk::AttachmentStoreOp::STORE)
		.stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
		.stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
		.initial_layout(vk::ImageLayout::UNDEFINED)
		.final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

    // Subpasses
    let color_attachment_ref = vk::AttachmentReference::builder()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    let stencil_attachment_ref = vk::AttachmentReference::builder()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

	let color_resolve_attachment_ref = vk::AttachmentReference::builder()
		.attachment(2)
		.layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    let color_attachments = &[color_attachment_ref];
	let resolve_attachments = &[color_resolve_attachment_ref];
    let subpass = vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(color_attachments)
        .depth_stencil_attachment(&stencil_attachment_ref)
		.resolve_attachments(resolve_attachments);

    // Dependencies
    let dependency = vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
            | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
        )
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
            | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
        )
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE
            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
        );

    // Create

    let attachments = &[color_attachment, depth_stencil_attachment, color_resolve_attachment];
    let subpasses = &[subpass];
    let dependencies = &[dependency];
    let info = vk::RenderPassCreateInfo::builder()
        .attachments(attachments)
        .subpasses(subpasses)
        .dependencies(dependencies);

    let render_pass = unsafe { device.create_render_pass(&info, None)? };

    Ok(render_pass)
}