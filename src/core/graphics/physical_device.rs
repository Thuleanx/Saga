use anyhow::{anyhow, Result};
use log::*;
use std::collections::HashSet;
use vulkanalia::prelude::v1_0::*;

use super::errors::SuitabilityError;
use super::queue_families::QueueFamilyIndices;
use crate::core::config::DEVICE_EXTENSIONS;

pub unsafe fn pick_physical_device(
    instance: &Instance,
    window_surface: vk::SurfaceKHR,
) -> Result<vk::PhysicalDevice> {
    for physical_device in instance.enumerate_physical_devices()? {
        let properties = instance.get_physical_device_properties(physical_device);

        if let Err(error) = check_physical_device(instance, window_surface, physical_device) {
            warn!(
                "Skipping physical device (`{}`): {}",
                properties.device_name, error
            );
        } else {
            info!("Selected physical device (`{}`).", properties.device_name);
            return Ok(physical_device);
        }
    }

    Err(anyhow!("Failed to find suitable physical device."))
}

/// Check a physical device to see if supports everything we need
unsafe fn check_physical_device(
    instance: &Instance,
    window_surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
) -> Result<()> {
    let properties = instance.get_physical_device_properties(physical_device);
    if properties.device_type != vk::PhysicalDeviceType::DISCRETE_GPU {
        return Err(anyhow!(SuitabilityError(
            "Only discrete GPUs are supported."
        )));
    }

    let features = instance.get_physical_device_features(physical_device);
    if features.sampler_anisotropy != vk::TRUE {
        return Err(anyhow!(SuitabilityError("No sampler anisotropy.")));
    }
    if features.geometry_shader != vk::TRUE {
        return Err(anyhow!(SuitabilityError(
            "Missing geometry shader support."
        )));
    }

    QueueFamilyIndices::get(instance, window_surface, physical_device)?;
    check_physical_device_extensions(instance, physical_device)?;

    // let support = SwapchainSupport::get(instance, data, physical_device)?;
    // if support.formats.is_empty() || support.present_modes.is_empty() {
    //     return Err(anyhow!(SuitabilityError("Insufficient swapchain support")));
    // }

    Ok(())
}

unsafe fn check_physical_device_extensions(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<()> {
    let extensions = instance
        .enumerate_device_extension_properties(physical_device, None)?
        .iter()
        .map(|e| e.extension_name)
        .collect::<HashSet<_>>();

    if DEVICE_EXTENSIONS.iter().all(|e| extensions.contains(e)) {
        Ok(())
    } else {
        Err(anyhow!(SuitabilityError(
            "Missing required device extensions."
        )))
    }
}
