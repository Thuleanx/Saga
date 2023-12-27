use anyhow::Result;
use std::mem::size_of;
use vulkanalia::prelude::v1_0::*;

use cgmath::{Matrix4, Deg, point3, vec3};
use crate::saga::Camera;

use super::{appdata::AppData, App};
use std::ptr::copy_nonoverlapping as memcpy;
type Mat4 = cgmath::Matrix4<f32>;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct UniformBufferObject {
    pub model: Mat4,
    pub  view: Mat4,
    pub proj: Mat4,
}

pub unsafe fn create_descriptor_set_layout(
    device: &Device,
    data: &mut AppData,
) -> Result<()> {

    let ubo_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::VERTEX);

    let ubo_bindings = &[ubo_binding];
    let info = vk::DescriptorSetLayoutCreateInfo::builder()
        .bindings(ubo_bindings);

    data.descriptor_set_layout = device.create_descriptor_set_layout(&info, None)?;


    Ok(())
}

pub unsafe fn create_uniform_buffers(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    use super::vertex;

    data.uniform_buffers.clear();
    data.uniform_buffers_memory.clear();

    for _ in 0..data.swapchain_images.len() {
        let (uniform_buffer, uniform_buffer_memory) = vertex::create_buffer(
            instance,
            device,
            data,
            size_of::<UniformBufferObject>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        )?;

        data.uniform_buffers.push(uniform_buffer);
        data.uniform_buffers_memory.push(uniform_buffer_memory);
    }

    Ok(())
}

pub unsafe fn destroy_uniform_buffers(app: &App) {
    app.data.uniform_buffers
        .iter()
        .for_each(|b| app.device.destroy_buffer(*b, None));
    app.data.uniform_buffers_memory
        .iter()
        .for_each(|b| app.device.free_memory(*b, None));
}

pub unsafe fn update_uniform_buffer(app: &App, image_index: usize) -> Result<()> {
    let time = app.start.elapsed().as_secs_f32();

    let model = Matrix4::from_axis_angle(
        vec3(0.0, 0.0, 1.0), 
        Deg(90.0) * time
    );

    /*
    let view = Matrix4::look_at_rh(
        point3(2.0, 2.0, 2.0), 
        point3(0.0, 0.0, 0.9), 
        vec3(0.0, 0.0, 1.0)
    );

    let mut proj = cgmath::perspective(
        Deg(45.0),
        app.data.swapchain_extent.width as f32 / app.data.swapchain_extent.height as f32,
        0.1, 10.0
    );

    // since cmath is designed for OpenGL, the Y coordinate of the the clip coordinate is
    // inverted as opposed to 
    proj[1][1] *= -1.0;
    */

    let view = app.data.camera.get_cached_view_matrix();
    let proj = app.data.camera.get_cached_projection_matrix();
    
    let ubo = UniformBufferObject { model, view, proj };

    let memory = app.device.map_memory(
        app.data.uniform_buffers_memory[image_index], 
        0,
        size_of::<UniformBufferObject>() as u64,
        vk::MemoryMapFlags::empty()
    )?;

    memcpy(&ubo, memory.cast(), 1);

    app.device.unmap_memory(app.data.uniform_buffers_memory[image_index]);

    Ok(())
}

pub unsafe fn destroy_descriptor_set_layout(app: &App) {
    app.device.destroy_descriptor_set_layout(app.data.descriptor_set_layout, None);
}
