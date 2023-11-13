use anyhow::Result;
use vulkanalia::prelude::v1_0::*;
use vulkanalia::window as vk_window;

use winit::window::Window;
use log::*;

use super::appdata::AppData;
use super::config::PORTABILITY_MACOS_VERSION;
use super::{validation_layers, App};

/// Create an instance of Vulkan with added checks and features:
/// - flags to enable portability extensions for MacOS
/// - application info with the Saga engine version
/// - validation layers
pub unsafe fn create_instance(window: &Window, entry: &Entry, data: &mut AppData) -> Result<Instance> {
    // Optional
    let application_info = vk::ApplicationInfo::builder()
        .application_name(b"Saga Engine\0")
        .application_version(vk::make_version(0, 0, 0))
        .engine_name(b"Saga\0")
        .engine_version(vk::make_version(0, 0, 0))
        .api_version(vk::make_version(1, 0, 0));

    // Required: We convert global extensions into c strings and pass it onto the Vulkan instance
    let mut extensions: Vec<*const i8> = vk_window::get_required_instance_extensions(window)
        .iter()
        .map(|e| e.as_ptr())
        .collect();

    let flags = if cfg!(target_os = "macos") && entry.version()? >= PORTABILITY_MACOS_VERSION {
        info!("Enabling extensions for macOS portability.");
        extensions.push(vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_EXTENSION.name.as_ptr());
        extensions.push(vk::KHR_PORTABILITY_ENUMERATION_EXTENSION.name.as_ptr());
        vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
    } else {
        vk::InstanceCreateFlags::empty()
    };

    let instance = validation_layers::create_instance_with_debug(
        entry, data, application_info, extensions, flags)?;
    Ok(instance)
}

pub unsafe fn destroy_instance(app: &App) {
    app.instance.destroy_instance(None);
}
