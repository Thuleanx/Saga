use anyhow::{anyhow, Result};
use std::{fs::File, path::Path};
use vulkanalia::{
    vk::{self, DeviceV1_0, HasBuilder},
    Device, Instance,
};

use crate::core::graphics::{
    buffers::{create_buffer, get_memory_type_index},
    command_buffers::{begin_single_time_commands, end_single_time_commands},
};

pub struct Image {
    width: u32,
    height: u32,
    pixels: Vec<u8>, // this is in [r, g, b, a] tuples
}

impl Image {
    pub fn load(filepath: &Path) -> Result<Self> {
        let image = File::open(filepath)?;

        let decoder = png::Decoder::new(image);

        let mut reader = decoder.read_info()?;
        let mut pixels = vec![0; reader.info().raw_bytes()];
        reader.next_frame(&mut pixels)?;

        let size = reader.info().raw_bytes() as u64;
        let (width, height) = reader.info().size();

        Ok(Image {
            width,
            height,
            pixels,
        })
    }
}

pub struct LoadedImage {
    image: vk::Image,
    memory: vk::DeviceMemory,
    image_view: vk::ImageView,
}

impl LoadedImage {
    pub fn get_image_view(&self) -> vk::ImageView {
        self.image_view
    }
}

impl LoadedImage {
    pub unsafe fn load_into_memory(
        image: &Image,
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        graphics_queue: vk::Queue,
        command_pool: vk::CommandPool,
    ) -> Result<Self> {
        use std::ptr::copy_nonoverlapping as memcpy;
        let size = image.pixels.len();

        let (staging_buffer, staging_buffer_memory) = create_buffer(
            instance,
            device,
            physical_device,
            size as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        )?;

        let memory = device.map_memory(
            staging_buffer_memory,
            0,
            size as u64,
            vk::MemoryMapFlags::empty(),
        )?;

        memcpy(image.pixels.as_ptr(), memory.cast(), size);

        device.unmap_memory(staging_buffer_memory);

        let (texture_image, texture_image_memory) =
            create_vk_image(instance, device, physical_device, image.width, image.height)?;

        transition_image_layout(
            device,
            graphics_queue,
            command_pool,
            texture_image,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        )?;

        copy_buffer_to_image(
            device,
            graphics_queue,
            command_pool,
            staging_buffer,
            texture_image,
            image.width,
            image.height,
        )?;

        transition_image_layout(
            device,
            graphics_queue,
            command_pool,
            texture_image,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        )?;

        device.destroy_buffer(staging_buffer, None);
        device.free_memory(staging_buffer_memory, None);

        let texture_image_view =
            create_image_view(device, texture_image, vk::Format::R8G8B8A8_SRGB)?;

        Ok(Self {
            image: texture_image,
            memory: texture_image_memory,
            image_view: texture_image_view,
        })
    }

    pub unsafe fn destroy(&mut self, device: &Device) {
        device.destroy_image(self.image, None);
        device.free_memory(self.memory, None);
        device.destroy_image_view(self.image_view, None);
    }
}

pub unsafe fn create_vk_image(
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    width: u32,
    height: u32,
) -> Result<(vk::Image, vk::DeviceMemory)> {
    let info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::_2D)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .format(vk::Format::R8G8B8A8_SRGB)
        .tiling(vk::ImageTiling::OPTIMAL)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .samples(vk::SampleCountFlags::_1)
        .flags(vk::ImageCreateFlags::empty());

    let texture_image = device.create_image(&info, None)?;

    let requirements = device.get_image_memory_requirements(texture_image);

    let info = vk::MemoryAllocateInfo::builder()
        .allocation_size(requirements.size)
        .memory_type_index(get_memory_type_index(
            instance,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            requirements,
            physical_device,
        )?);

    let texture_image_memory = device.allocate_memory(&info, None)?;

    device.bind_image_memory(texture_image, texture_image_memory, 0)?;

    Ok((texture_image, texture_image_memory))
}

pub unsafe fn transition_image_layout(
    device: &Device,
    graphics_queue: vk::Queue,
    command_pool: vk::CommandPool,
    image: vk::Image,
    format: vk::Format,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) -> Result<()> {
    let command_buffer = begin_single_time_commands(device, command_pool)?;

    let (src_access_mask, dst_access_mask, src_stage_mask, dst_stage_mask) =
        match (old_layout, new_layout) {
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
            ),
            (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
            ),
            _ => return Err(anyhow!("Unsupported image layout transition!")),
        };

    let subresource = vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1);

    let barrier = vk::ImageMemoryBarrier::builder()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(subresource)
        .src_access_mask(src_access_mask)
        .dst_access_mask(dst_access_mask);

    device.cmd_pipeline_barrier(
        command_buffer,
        src_stage_mask, // TODO
        dst_stage_mask, // TODO
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[barrier],
    );

    end_single_time_commands(device, graphics_queue, command_pool, command_buffer)?;

    Ok(())
}

pub unsafe fn copy_buffer_to_image(
    device: &Device,
    graphics_queue: vk::Queue,
    command_pool: vk::CommandPool,
    buffer: vk::Buffer,
    image: vk::Image,
    width: u32,
    height: u32,
) -> Result<()> {
    let command_buffer = begin_single_time_commands(device, command_pool)?;

    let subresource = vk::ImageSubresourceLayers::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .mip_level(0)
        .base_array_layer(0)
        .layer_count(1);

    let region = vk::BufferImageCopy::builder()
        .buffer_offset(0)
        .buffer_row_length(0) // means they are tightly packed in memory
        .buffer_image_height(0) // means they are tightly packed in memory
        .image_subresource(subresource)
        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
        .image_extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        });

    device.cmd_copy_buffer_to_image(
        command_buffer,
        buffer,
        image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        &[region],
    );

    end_single_time_commands(device, graphics_queue, command_pool, command_buffer)?;

    Ok(())
}

pub unsafe fn create_image_view(
    device: &Device,
    image: vk::Image,
    format: vk::Format,
) -> Result<vk::ImageView> {
    let subresource_range = vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1);

    let info = vk::ImageViewCreateInfo::builder()
        .image(image)
        .view_type(vk::ImageViewType::_2D)
        .format(format)
        .subresource_range(subresource_range);

    let image_view = device.create_image_view(&info, None)?;

    Ok(image_view)
}
