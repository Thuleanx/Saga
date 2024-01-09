use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use super::pipeline::INDICES;
use super::queue_families::QueueFamilyIndices;

pub unsafe fn create_command_buffers(
    device: &Device,
    command_pool: vk::CommandPool,
    swapchain_extent: vk::Extent2D,
    render_pass: vk::RenderPass,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    framebuffers: &Vec<vk::Framebuffer>,
    vertex_buffer: vk::Buffer,
    index_buffer: vk::Buffer,
    descriptor_sets: &Vec<vk::DescriptorSet>,
) -> Result<Vec<vk::CommandBuffer>> {
    // Allocate

    let allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(framebuffers.len() as u32);

    let command_buffers = device.allocate_command_buffers(&allocate_info)?;

    // Commands

    for (i, command_buffer) in command_buffers.iter().enumerate() {
        let info = vk::CommandBufferBeginInfo::builder();

        device.begin_command_buffer(*command_buffer, &info)?;

        let render_area = vk::Rect2D::builder()
            .offset(vk::Offset2D::default())
            .extent(swapchain_extent);

        let color_clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };

        let clear_values = &[color_clear_value];
        let info = vk::RenderPassBeginInfo::builder()
            .render_pass(render_pass)
            .framebuffer(framebuffers[i])
            .render_area(render_area)
            .clear_values(clear_values);

        device.cmd_begin_render_pass(*command_buffer, &info, vk::SubpassContents::INLINE);
        device.cmd_bind_pipeline(*command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline);

        // Bind Vertex buffer
        {
            let first_binding = 0;
            let memory_offsets = [0];
            device.cmd_bind_vertex_buffers(*command_buffer, first_binding, &[vertex_buffer], &memory_offsets);
            device.cmd_bind_index_buffer(*command_buffer, index_buffer, memory_offsets[0], vk::IndexType::UINT16);
        }

        // Bind Descriptor Set
        {
            device.cmd_bind_descriptor_sets(*command_buffer, vk::PipelineBindPoint::GRAPHICS, 
                                            pipeline_layout, 0, &[descriptor_sets[i]], &[]);

        }

        device.cmd_draw_indexed(*command_buffer, INDICES.len() as u32, 1, 0, 0, 0);
        device.cmd_end_render_pass(*command_buffer);

        device.end_command_buffer(*command_buffer)?;
    }


    Ok(command_buffers)
}

pub unsafe fn create_command_pool(
    instance: &Instance, device: &Device, 
    surface: vk::SurfaceKHR, physical_device: vk::PhysicalDevice
) -> Result<vk::CommandPool> {

    let indices = QueueFamilyIndices::get(instance, surface, physical_device)?;

    let info = vk::CommandPoolCreateInfo::builder()
        .flags(vk::CommandPoolCreateFlags::empty()) // Optional.
        .queue_family_index(indices.graphics);

    let command_pool = device.create_command_pool(&info, None)?;

    Ok(command_pool)
}

pub unsafe fn destroy_command_pool(device: &Device, command_pool: vk::CommandPool) {
    device.destroy_command_pool(command_pool, None);
}
