use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use crate::core::config::MAX_FRAMES_IN_FLIGHT;

pub unsafe fn create_sync_objects(device: &Device, swapchain_images: &Vec<vk::Image>) 
    -> Result<(Vec<vk::Semaphore>, Vec<vk::Semaphore>, Vec<vk::Fence>, Vec<vk::Fence>)> {
    let semaphore_info = vk::SemaphoreCreateInfo::builder();
    let fence_info = vk::FenceCreateInfo::builder()
        .flags(vk::FenceCreateFlags::SIGNALED);

    let image_available_semaphores = (0..MAX_FRAMES_IN_FLIGHT)
        .map(|i| {
            device.create_semaphore(&semaphore_info, None)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let render_finished_semaphores = (0..MAX_FRAMES_IN_FLIGHT)
        .map(|i| {
            device.create_semaphore(&semaphore_info, None)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let in_flight_fences = (0..MAX_FRAMES_IN_FLIGHT)
        .map(|i| {
            device.create_fence(&fence_info, None)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let images_in_flight : Vec<vk::Fence> = swapchain_images
        .iter()
        .map(|_| vk::Fence::null())
        .collect();

    Ok((image_available_semaphores, 
        render_finished_semaphores, 
        in_flight_fences,
        images_in_flight))
}

pub unsafe fn destroy_semaphores(device: &Device, semaphores: &Vec<vk::Semaphore>) {
    semaphores.iter()
        .for_each(|s| device.destroy_semaphore(*s, None));
}

pub unsafe fn destroy_fences(device: &Device, fences: &Vec<vk::Fence>) {
    fences.iter()
        .for_each(|f| device.destroy_fence(*f, None));
}
