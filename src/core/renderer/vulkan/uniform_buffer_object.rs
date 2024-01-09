use anyhow::Result;
use std::{mem::size_of, time::Instant};
use vulkanalia::prelude::v1_0::*;

use cgmath::{Matrix4, Deg, vec3};
use crate::saga::Camera;

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
) -> Result<vk::DescriptorSetLayout> {

    let ubo_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::VERTEX);

    let ubo_bindings = &[ubo_binding];
    let info = vk::DescriptorSetLayoutCreateInfo::builder()
        .bindings(ubo_bindings);

    let descriptor_set_layout = device.create_descriptor_set_layout(&info, None)?;

    Ok(descriptor_set_layout)
}

pub unsafe fn create_uniform_buffers(
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    uniform_buffers: &mut Vec<vk::Buffer>,
    uniform_buffers_memory: &mut Vec<vk::DeviceMemory>,
    swapchain_images: &Vec<vk::Image>,
) -> Result<()> {
    use super::vertex;

    uniform_buffers.clear();
    uniform_buffers_memory.clear();

    for _ in 0..swapchain_images.len() {
        let (uniform_buffer, uniform_buffer_memory) = vertex::create_buffer(
            instance,
            device,
            physical_device,
            size_of::<UniformBufferObject>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        )?;

        uniform_buffers.push(uniform_buffer);
        uniform_buffers_memory.push(uniform_buffer_memory);
    }

    Ok(())
}

pub unsafe fn destroy_uniform_buffers(
    device: &Device,
    uniform_buffers: &mut Vec<vk::Buffer>) {

    uniform_buffers
        .iter()
        .for_each(|b| device.destroy_buffer(*b, None));
}

pub unsafe fn destroy_uniform_buffers_memory(
    device: &Device,
    uniform_buffers_memory: &mut Vec<vk::DeviceMemory>) {

    uniform_buffers_memory
        .iter()
        .for_each(|b| device.free_memory(*b, None));
}

pub unsafe fn update_uniform_buffer(
    device: &Device, 
    image_index: usize, 
    start_time: Instant,
    uniform_buffers_memory: &mut Vec<vk::DeviceMemory>,
    camera: &dyn Camera,
) -> Result<()> {
    let time = start_time.elapsed().as_secs_f32();

    let model = Matrix4::from_axis_angle(
        vec3(0.0, 0.0, 1.0), 
        Deg(90.0) * time
    );

    let view = camera.get_cached_view_matrix();
    let proj = camera.get_cached_projection_matrix();
    
    let ubo = UniformBufferObject { model, view, proj };

    let memory = device.map_memory(
        uniform_buffers_memory[image_index], 
        0,
        size_of::<UniformBufferObject>() as u64,
        vk::MemoryMapFlags::empty()
    )?;

    memcpy(&ubo, memory.cast(), 1);

    device.unmap_memory(uniform_buffers_memory[image_index]);

    Ok(())
}

pub unsafe fn destroy_descriptor_set_layout(
    device: &Device, 
    descriptor_set_layout: vk::DescriptorSetLayout
) -> () {
    device.destroy_descriptor_set_layout(descriptor_set_layout, None);
}
