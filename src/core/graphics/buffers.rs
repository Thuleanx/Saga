use anyhow::{anyhow, Result};
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::{self};

use super::command_buffers::{begin_single_time_commands, end_single_time_commands};

type Vec2 = cgmath::Vector2<f32>;
type Vec3 = cgmath::Vector3<f32>;

pub unsafe fn create_buffer(
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    let buffer_info = vk::BufferCreateInfo::builder()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .flags(vk::BufferCreateFlags::empty());

    let buffer = device.create_buffer(&buffer_info, None)?;

    let memory_requirements: vk::MemoryRequirements = device.get_buffer_memory_requirements(buffer);

    let memory_type_index: u32 =
        get_memory_type_index(instance, properties, memory_requirements, physical_device)?;

    let memory_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(memory_requirements.size)
        .memory_type_index(memory_type_index);

    let buffer_memory = device.allocate_memory(&memory_info, None)?;

    let buffer_offset: u64 = 0;
    device.bind_buffer_memory(buffer, buffer_memory, buffer_offset)?;

    Ok((buffer, buffer_memory))
}

pub unsafe fn get_memory_type_index(
    instance: &Instance,
    properties: vk::MemoryPropertyFlags,
    requirements: vk::MemoryRequirements,
    physical_device: vk::PhysicalDevice,
) -> Result<u32> {
    let memory_properties: vk::PhysicalDeviceMemoryProperties =
        instance.get_physical_device_memory_properties(physical_device);

    (0..memory_properties.memory_type_count)
        .find(|memory_type| {
            let is_memory_type_suitable = (requirements.memory_type_bits & (1 << memory_type)) != 0;
            let actual_memory_type: vk::MemoryType =
                memory_properties.memory_types[*memory_type as usize];

            is_memory_type_suitable && actual_memory_type.property_flags.contains(properties)
        })
        .ok_or_else(|| anyhow!("Failed to find suitable memory type"))
}

pub unsafe fn copy_buffer(
    device: &Device,
    source: vk::Buffer,
    destination: vk::Buffer,
    size: vk::DeviceSize,
    command_pool: vk::CommandPool,
    graphics_queue: vk::Queue,
) -> Result<()> {
    let command_buffer = begin_single_time_commands(device, command_pool)?;

    let regions = vk::BufferCopy::builder().size(size);
    device.cmd_copy_buffer(command_buffer, source, destination, &[regions]);

    end_single_time_commands(device, graphics_queue, command_pool, command_buffer)?;

    Ok(())
}
