use anyhow::Result;
use std::collections::HashSet;
use vulkanalia::prelude::v1_0::*;

use super::queue_families::QueueFamilyIndices;
use super::validation_layers::*;

use crate::core::config::{DEVICE_EXTENSIONS, PORTABILITY_MACOS_VERSION};

pub unsafe fn create_logical_device(
    entry: &Entry,
    instance: &Instance,
    window_surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
)-> Result<(Device, vk::Queue, vk::Queue)> {

    let indices = QueueFamilyIndices::get(instance, window_surface, physical_device)?;

    let mut unique_indices = HashSet::new();
    unique_indices.insert(indices.graphics);
    unique_indices.insert(indices.present);

    let queue_priorities = &[1.0];
    let queue_infos = unique_indices
        .iter()
        .map(|i| {
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(*i)
                .queue_priorities(queue_priorities)
        })
        .collect::<Vec<_>>();

    let layers = get_validation_layers(entry)?;

    let mut extensions = DEVICE_EXTENSIONS
        .iter()
        .map(|n| n.as_ptr())
        .collect::<Vec<_>>();

    // Required by Vulkan SDK on macOS since 1.3.216.
    if cfg!(target_os = "macos") && entry.version()? >= PORTABILITY_MACOS_VERSION {
        extensions.push(vk::KHR_PORTABILITY_SUBSET_EXTENSION.name.as_ptr());
    }

    let features = vk::PhysicalDeviceFeatures::builder()
        .sampler_anisotropy(true); // allows for anisotropy to be used on image samplers

    let info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_infos)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions)
        .enabled_features(&features);

    let device = instance.create_device(physical_device, &info, None)?;

    let graphics_queue = device.get_device_queue(indices.graphics, 0);
    let present_queue = device.get_device_queue(indices.present, 0);

    Ok((device, graphics_queue, present_queue))
}

pub unsafe fn destroy_logical_device(
    device: &Device) {

    device.destroy_device(None);
}
