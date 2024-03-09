use anyhow::Result;
use std::mem::size_of;
use vulkanalia::vk::{self};
use vulkanalia::prelude::v1_0::*;

use super::super::buffers::{create_buffer, copy_buffer};

type Vec3 = cgmath::Vector3<f32>;
type Vec2 = cgmath::Vector2<f32>;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    pub pos: Vec3,
    pub uv: Vec2
    // color: Vec3,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct VertexBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

impl Vertex {
    pub const fn new(
        pos: Vec3, 
        uv: Vec2
        // color: Vec3,
    ) -> Self {
        Self {
            pos, 
            uv
            // color
        }
    }

    pub fn binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()
    }

    pub fn attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
        let pos = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0)
            .build();
        let uv = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32_SFLOAT)
            .offset((size_of::<Vec3>()) as u32)
            .build();
        // let color = vk::VertexInputAttributeDescription::builder()
        //     .binding(0)
        //     .location(1)
        //     .format(vk::Format::R32G32B32_SFLOAT)
        //     .offset(size_of::<Vec3>() as u32)
        //     .build();
        // [pos, color]
        [pos, uv]
    }
}

impl VertexBuffer {
    pub(in crate::core::graphics) unsafe fn create(
        instance: &Instance, 
        device: &Device,
        physical_device: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
        graphics_queue: vk::Queue,
        vertices: &[Vertex],
    ) -> Result<Self> {
        use std::ptr::copy_nonoverlapping as memcpy;

        let size: u64 = (size_of::<Vertex>() * vertices.len()) as u64;

        let memory_property_flags = vk::MemoryPropertyFlags::HOST_COHERENT | 
            vk::MemoryPropertyFlags::HOST_VISIBLE;

        let (staging_buffer, staging_buffer_memory) = create_buffer(
            instance, device, physical_device,
            size, vk::BufferUsageFlags::TRANSFER_SRC, 
            memory_property_flags)?;

        let memory_offset: u64 = 0;
        let gpu_memory = device.map_memory(
            staging_buffer_memory,
            memory_offset, 
            size, 
            vk::MemoryMapFlags::empty()
        )?;

        memcpy(vertices.as_ptr(), gpu_memory.cast(), vertices.len());

        device.unmap_memory(staging_buffer_memory);

        let (vertex_buffer, vertex_buffer_memory) = create_buffer(
            instance, device, physical_device,
            size, vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER, 
            vk::MemoryPropertyFlags::DEVICE_LOCAL)?;

        copy_buffer(device, staging_buffer, vertex_buffer, size, command_pool, graphics_queue)?;

        device.destroy_buffer(staging_buffer, None);
        device.free_memory(staging_buffer_memory, None);

        Ok(Self {
            buffer : vertex_buffer,
            memory : vertex_buffer_memory
        })
    }
    pub unsafe fn destroy(buffer: VertexBuffer, device: &Device) {
        device.destroy_buffer(buffer.buffer, None);
        device.free_memory(buffer.memory, None);
    }
}

impl VertexBuffer {
    pub unsafe fn bind(&self, device: &Device, command_buffer: vk::CommandBuffer, first_binding : u32, memory_offset: u64) {
        device.cmd_bind_vertex_buffers(
            command_buffer, 
            first_binding, 
            &[self.buffer], 
            &[memory_offset]
        );
    }
}
