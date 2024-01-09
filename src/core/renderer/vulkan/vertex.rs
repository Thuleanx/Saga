use std::mem::size_of;
use anyhow::{Result, anyhow};
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::{self, HasBuilder};

use super::pipeline::{VERTICES, INDICES};

type Vec2 = cgmath::Vector2<f32>;
type Vec3 = cgmath::Vector3<f32>;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    pos: Vec2,
    color: Vec3,
}

impl Vertex {
    pub const fn new(pos: Vec2, color: Vec3) -> Self {
        Self {pos, color}
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
            .format(vk::Format::R32G32_SFLOAT)
            .offset(0)
            .build();
        let color = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(size_of::<Vec2>() as u32)
            .build();
        [pos, color]
    }
}

pub unsafe fn create_vertex_buffer(instance: &Instance, 
    device: &Device, 
    physical_device: vk::PhysicalDevice,
    command_pool: vk::CommandPool,
    graphics_queue: vk::Queue,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    use std::ptr::copy_nonoverlapping as memcpy;

    let size: u64 = (size_of::<Vertex>() * VERTICES.len()) as u64;

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

    memcpy(VERTICES.as_ptr(), gpu_memory.cast(), VERTICES.len());

    device.unmap_memory(staging_buffer_memory);

    let (vertex_buffer, vertex_buffer_memory) = create_buffer(
        instance, device, physical_device,
        size, vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER, 
        vk::MemoryPropertyFlags::DEVICE_LOCAL)?;

    copy_buffer(device, staging_buffer, vertex_buffer, size, command_pool, graphics_queue)?;

    device.destroy_buffer(staging_buffer, None);
    device.free_memory(staging_buffer_memory, None);

    Ok((vertex_buffer, vertex_buffer_memory))
}

pub unsafe fn create_index_buffer(
    instance: &Instance, 
    device: &Device, 
    physical_device: vk::PhysicalDevice, 
    command_pool: vk::CommandPool,
    graphics_queue: vk::Queue,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    use std::ptr::copy_nonoverlapping as memcpy;

    let size: u64 = (size_of::<u16>() * INDICES.len()) as u64;

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

    memcpy(INDICES.as_ptr(), gpu_memory.cast(), INDICES.len());

    device.unmap_memory(staging_buffer_memory);

    let (index_buffer, index_buffer_memory) = create_buffer(
        instance, device, physical_device,
        size, vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER, 
        vk::MemoryPropertyFlags::DEVICE_LOCAL)?;


    copy_buffer(device, staging_buffer, index_buffer, size, 
                command_pool, graphics_queue)?;

    device.destroy_buffer(staging_buffer, None);
    device.free_memory(staging_buffer_memory, None);

    Ok((index_buffer, index_buffer_memory))
}

unsafe fn get_memory_type_index(
    instance: &Instance,
    properties: vk::MemoryPropertyFlags,
    requirements: vk::MemoryRequirements,
    physical_device: vk::PhysicalDevice,
) -> Result<u32> {
    let memory_properties: vk::PhysicalDeviceMemoryProperties = instance.get_physical_device_memory_properties(physical_device);

    (0..memory_properties.memory_type_count)
        .find(|memory_type| {
            let is_memory_type_suitable = (requirements.memory_type_bits & (1 << memory_type)) != 0;
            let actual_memory_type: vk::MemoryType = memory_properties.memory_types[*memory_type as usize];

            is_memory_type_suitable && actual_memory_type.property_flags.contains(properties)
        })
        .ok_or_else(|| anyhow!("Failed to find suitable memory type"))
}


/**
 * Copy the contents of one buffer to another
 */
unsafe fn copy_buffer(
    device: &Device,
    source: vk::Buffer,
    destination: vk::Buffer,
    size: vk::DeviceSize,
    command_pool: vk::CommandPool,
    graphics_queue: vk::Queue,
) -> Result<()> {

    let info = vk::CommandBufferAllocateInfo::builder()
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(command_pool)
        .command_buffer_count(1);

    let command_buffer = device.allocate_command_buffers(&info)?[0];

    let begin_info = vk::CommandBufferBeginInfo::builder()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    device.begin_command_buffer(command_buffer, &begin_info)?;

    let regions = vk::BufferCopy::builder().size(size);
    device.cmd_copy_buffer(command_buffer, source, destination, &[regions]);

    device.end_command_buffer(command_buffer)?;

    let command_buffers = &[command_buffer];
    let submit_info = vk::SubmitInfo::builder()
        .command_buffers(command_buffers);

    device.queue_submit(graphics_queue, &[submit_info], vk::Fence::null())?;

    // can optimize by using a fence
    device.queue_wait_idle(graphics_queue)?;

    device.free_command_buffers(command_pool, command_buffers);

    Ok(())
}

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

    let memory_type_index : u32 = get_memory_type_index(
        instance, 
        properties,
        memory_requirements,
        physical_device,
    )?;

    let memory_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(memory_requirements.size)
        .memory_type_index(memory_type_index);

    let buffer_memory = device.allocate_memory(&memory_info, None)?;

    let buffer_offset: u64 = 0;
    device.bind_buffer_memory(buffer, buffer_memory, buffer_offset)?;

    Ok((buffer, buffer_memory))
}

pub unsafe fn destroy_buffer_and_free_memory(device: &Device, buffer: vk::Buffer, memory: vk::DeviceMemory) {
    device.destroy_buffer(buffer, None);
    device.free_memory(memory, None);
}

