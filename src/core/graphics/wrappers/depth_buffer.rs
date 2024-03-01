use anyhow::{anyhow, Result};
use vulkanalia::prelude::v1_0::*;
use vulkanalia::{vk, Device, Instance};

use crate::core::graphics::swapchain::Swapchain;

use super::create_image_view;
use super::image::{create_vk_image, transition_image_layout};

pub struct DepthBuffer {
    image: vk::Image,
    image_memory: vk::DeviceMemory,
    image_view: vk::ImageView,
}

impl DepthBuffer {
    pub fn get_image_view(&self) -> vk::ImageView {
        self.image_view
    }
}

impl DepthBuffer {
    pub unsafe fn new(
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        swapchain: &Swapchain,
        graphics_queue: vk::Queue,
        command_pool: vk::CommandPool,
    ) -> Result<Self> {

        let format = get_depth_format(instance, physical_device)?;
        let extent = swapchain.get_extent();
        let (depth_image, depth_image_memory) = create_vk_image(
            instance,
            device,
            physical_device,
            extent.width,
            extent.height,
            format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let depth_image_view =
            create_image_view(device, depth_image, format, vk::ImageAspectFlags::DEPTH)?;

        // This is optional but included for completeness
        // It's taken care of in the render pass when we render
        transition_image_layout(
            device,
            graphics_queue,
            command_pool,
            depth_image,
            format,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        )?;

        Ok(Self {
            image: depth_image,
            image_memory: depth_image_memory,
            image_view: depth_image_view,
        })
    }

    pub unsafe fn destroy(&self, device: &Device) {
        device.destroy_image(self.image, None);
        device.destroy_image_view(self.image_view, None);
        device.free_memory(self.image_memory, None);
    }
}

pub unsafe fn get_depth_format(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<vk::Format> {
    let candidates = &[
        vk::Format::D32_SFLOAT,
        vk::Format::D32_SFLOAT_S8_UINT,
        vk::Format::D24_UNORM_S8_UINT,
    ];

    get_supported_format(
        instance,
        physical_device,
        candidates,
        vk::ImageTiling::OPTIMAL,
        vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
    )
}

pub unsafe fn get_supported_format(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    candidates: &[vk::Format],
    tiling: vk::ImageTiling,
    features: vk::FormatFeatureFlags,
) -> Result<vk::Format> {
    candidates
        .iter()
        .cloned()
        .find(|f| {
            let properties = instance.get_physical_device_format_properties(physical_device, *f);

            match tiling {
                vk::ImageTiling::LINEAR => properties.linear_tiling_features.contains(features),
                vk::ImageTiling::OPTIMAL => properties.optimal_tiling_features.contains(features),
                _ => false,
            }
        })
        .ok_or_else(|| anyhow!("Failed to find supported format!"))
}
