use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use super::{shader, wrappers::Vertex};

// pub static VERTICES: [Vertex; 4] = [
//     // Vertex::new(vec3(-0.5, -0.5, 0.0), vec3(1.0, 0.0, 0.0)),
//     // Vertex::new(vec3(0.5, -0.5, 0.0), vec3(0.0, 1.0, 0.0)),
//     // Vertex::new(vec3(0.5, 0.5, 0.0), vec3(0.0, 0.0, 1.0)),
//     // Vertex::new(vec3(-0.5, 0.5, 0.0), vec3(1.0, 1.0, 1.0)),
//     Vertex::new(vec3(-0.5, -0.5, 0.0)),
//     Vertex::new(vec3(0.5, -0.5, 0.0)),
//     Vertex::new(vec3(0.5, 0.5, 0.0)),
//     Vertex::new(vec3(-0.5, 0.5, 0.0)),
// ];

pub static INDICES : &[u16] = &[0, 1, 2, 2, 3, 0];

pub unsafe fn create_pipeline(
    device: &Device, 
    swapchain_extent: vk::Extent2D, 
    set_layouts: &[vk::DescriptorSetLayout],
    render_pass: vk::RenderPass,
) -> Result<(vk::PipelineLayout, vk::Pipeline)> {

    let vert = include_bytes!("../../../shaders_compiled/vert.spv");
    let frag = include_bytes!("../../../shaders_compiled/frag.spv");

    let vert_shader_module = shader::create_shader_module(device, vert)?;
    let frag_shader_module = shader::create_shader_module(device, frag)?;

    let binding_description: &[vk::VertexInputBindingDescription; 1] 
        = &[Vertex::binding_description()];
    let attribute_descriptions = &Vertex::attribute_descriptions();
    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
        .vertex_binding_descriptions(binding_description)
        .vertex_attribute_descriptions(attribute_descriptions);

    let vert_stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(vert_shader_module)
        .name(b"main\0");

    let frag_stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::FRAGMENT)
        .module(frag_shader_module)
        .name(b"main\0");

    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

    let viewport = vk::Viewport::builder()
        .x(0.0)
        .y(0.0)
        .width(swapchain_extent.width as f32)
        .height(swapchain_extent.height as f32)
        .min_depth(0.0)
        .max_depth(1.0);

    let scissor = vk::Rect2D::builder()
        .offset(vk::Offset2D {x: 0, y: 0})
        .extent(swapchain_extent);

    let viewports = &[viewport];
    let scissors = &[scissor];
    let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
        .viewports(viewports)
        .scissors(scissors);

    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .depth_bias_enable(false);

    let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlags::_1);

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
    let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS)
        .depth_bounds_test_enable(false)
        .min_depth_bounds(0.0)
        .max_depth_bounds(1.0)
        .stencil_test_enable(false);

    let layout_info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(set_layouts);
    log::info!("Create pipeline with layout of {} descriptors.", layout_info.set_layout_count);

    let pipeline_layout: vk::PipelineLayout = device.create_pipeline_layout(&layout_info, None)?;

    let stages = &[vert_stage, frag_stage];
    let info = vk::GraphicsPipelineCreateInfo::builder()
        .stages(stages)
        .vertex_input_state(&vertex_input_state)
        .input_assembly_state(&input_assembly_state)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization_state)
        .multisample_state(&multisample_state)
        .color_blend_state(&color_blend_state)
        .depth_stencil_state(&depth_stencil_state)
        .layout(pipeline_layout)
        .render_pass(render_pass)
        .subpass(0)
        .base_pipeline_handle(vk::Pipeline::null())
        .base_pipeline_index(-1);

    let pipeline: vk::Pipeline = device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)?.0[0];

    shader::destroy_shader_module(device, vert_shader_module);
    shader::destroy_shader_module(device, frag_shader_module);

    Ok((pipeline_layout, pipeline))
}

pub unsafe fn destroy_pipeline(device: &Device, pipeline: vk::Pipeline, pipeline_layout: vk::PipelineLayout) {
    device.destroy_pipeline(pipeline, None);
    device.destroy_pipeline_layout(pipeline_layout, None);
}
