use anyhow::Result;
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::KhrSurfaceExtension;
use vulkanalia::window as vk_window;
use winit::window::Window;

pub unsafe fn create_window_surface(
    instance: &Instance,
    window: &Window,
) -> Result<vk::SurfaceKHR> {
    let surface = vk_window::create_surface(&instance, &window, &window)?;
    Ok(surface)
}

/// ## Note:
/// Make sure surface is destroyed before instance.
pub unsafe fn destroy_window_surface(instance: &Instance, window_surface: vk::SurfaceKHR) {
    instance.destroy_surface_khr(window_surface, None);
}
