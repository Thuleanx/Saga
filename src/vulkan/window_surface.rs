use anyhow::Result;
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::KhrSurfaceExtension;
use vulkanalia::window as vk_window;
use winit::window::Window;


use super::app::App;
use super::appdata::AppData;

pub unsafe fn create_window_surface(
    instance: &Instance,
    window: &Window,
    data: &mut AppData,
) -> Result<()> {
    data.surface = vk_window::create_surface(&instance, &window, &window)?;
    Ok(())
}

/// ## Note:
/// Make sure surface is destroyed before instance.
pub unsafe fn destroy_window_surface(app: &App) {
    app.instance.destroy_surface_khr(app.data.surface, None);
}
