use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use super::App;
use super::config::MAX_FRAMES_IN_FLIGHT;
use super::appdata::AppData;

pub unsafe fn create_sync_objects(device: &Device, data: &mut AppData) -> Result<()> {
    let semaphore_info = vk::SemaphoreCreateInfo::builder();
    let fence_info = vk::FenceCreateInfo::builder()
        .flags(vk::FenceCreateFlags::SIGNALED);

    data.image_available_semaphores = (0..MAX_FRAMES_IN_FLIGHT)
        .map(|i| {
            device.create_semaphore(&semaphore_info, None)
        })
        .collect::<Result<Vec<_>, _>>()?;

    data.render_finished_semaphores = (0..MAX_FRAMES_IN_FLIGHT)
        .map(|i| {
            device.create_semaphore(&semaphore_info, None)
        })
        .collect::<Result<Vec<_>, _>>()?;

    data.in_flight_fences = (0..MAX_FRAMES_IN_FLIGHT)
        .map(|i| {
            device.create_fence(&fence_info, None)
        })
        .collect::<Result<Vec<_>, _>>()?;

    data.images_in_flight = data.swapchain_images
        .iter()
        .map(|_| vk::Fence::null())
        .collect();

    Ok(())
}

pub unsafe fn destroy_sync_objects(app: &App) {
    app.data.render_finished_semaphores
        .iter()
        .for_each(|s| app.device.destroy_semaphore(*s, None));
    app.data.image_available_semaphores
        .iter()
        .for_each(|s| app.device.destroy_semaphore(*s, None));
    app.data.in_flight_fences
        .iter()
        .for_each(|f| app.device.destroy_fence(*f, None));
}
