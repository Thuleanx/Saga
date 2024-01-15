use anyhow::Result;
use std::mem::size_of;
use vulkanalia::vk::{self};
use vulkanalia::prelude::v1_0::*;

use crate::core::graphics::vulkan::buffers::{create_buffer, copy_buffer};

pub struct Index(u16);

#[derive(Copy, Clone, Debug, Default)]
pub struct IndexBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

impl IndexBuffer {
    pub unsafe fn create(
        instance: &Instance, 
        device: &Device, 
        physical_device: vk::PhysicalDevice, 
        command_pool: vk::CommandPool,
        graphics_queue: vk::Queue,
        indices: &[u16],
    ) -> Result<IndexBuffer> {
        use std::ptr::copy_nonoverlapping as memcpy;

        let size: u64 = (size_of::<Index>() * indices.len()) as u64;

        let memory_property_flags = vk::MemoryPropertyFlags::HOST_COHERENT | 
                                    vk::MemoryPropertyFlags::HOST_VISIBLE;
        let (staging_buffer, staging_buffer_memory) = create_buffer(
            instance, device, physical_device, size, 
            vk::BufferUsageFlags::TRANSFER_SRC, 
            memory_property_flags)?;

        let memory_offset: u64 = 0;
        let gpu_memory = device.map_memory(
            staging_buffer_memory,
            memory_offset, 
            size, 
            vk::MemoryMapFlags::empty()
        )?;

        memcpy(indices.as_ptr(), gpu_memory.cast(), indices.len());

        device.unmap_memory(staging_buffer_memory);

        let (index_buffer, index_buffer_memory) = create_buffer(
            instance, device, physical_device,
            size, vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER, 
            vk::MemoryPropertyFlags::DEVICE_LOCAL)?;


        copy_buffer(device, staging_buffer, index_buffer, size, 
                    command_pool, graphics_queue)?;

        device.destroy_buffer(staging_buffer, None);
        device.free_memory(staging_buffer_memory, None);

        Ok(IndexBuffer {
            buffer: index_buffer, 
            memory: index_buffer_memory
        })
    }

    pub unsafe fn unload(buffer: &IndexBuffer, device: &Device) {
        device.destroy_buffer(buffer.buffer, None);
        device.free_memory(buffer.memory, None);
    }

    pub unsafe fn bind(&self, device: &Device, command_buffer: vk::CommandBuffer, memory_offset: u64) {
        device.cmd_bind_index_buffer(
            command_buffer, 
            self.buffer,
            memory_offset,
            vk::IndexType::UINT16
        );
    }
}

