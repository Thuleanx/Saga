use anyhow::{anyhow, Result};
use vulkanalia::prelude::v1_0::*;

use crate::core::config::MAX_FRAMES_IN_FLIGHT;

pub struct GraphicsBarriers {
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    images_in_flight: Vec<vk::Fence>,
}

impl GraphicsBarriers {
    pub fn new(device: &Device, swapchain_images: &[vk::Image]) -> Result<Self> {
        let (image_available_semaphores, render_finished_semaphores,
             in_flight_fences, images_in_flight) 
            = unsafe {create_sync_objects(device, swapchain_images)?};
        Ok(Self { 
            image_available_semaphores, 
            render_finished_semaphores, 
            in_flight_fences, 
            images_in_flight 
        })
    }

    pub fn get_length(&self) -> usize { self.in_flight_fences.len() }

    pub fn wait_for_in_flight_fence(&self, 
        device: &Device, frame_index: usize
    ) -> Result<()> {

        let in_flight_fence = self.in_flight_fences.get(frame_index);
        match in_flight_fence {
            Some(in_flight_fence) => unsafe {
                device.wait_for_fences(
                    &[in_flight_fence.clone()]
                    , true, u64::MAX)?;
            }
            None => {
                return Err(anyhow!("Frame index {} > Number of in flight fences {}", 
                                   frame_index, self.in_flight_fences.len()));
            }
        }
        Ok(())
    }

    pub fn wait_for_image_in_flight(
        &self, device: &Device, frame_index: usize
    ) -> Result<()> {
        match self.images_in_flight.get(frame_index) {
            Some(image_in_flight) => {
                if !image_in_flight.is_null() {
                    unsafe {device.wait_for_fences(
                        &[image_in_flight.clone()],
                        true,
                        u64::MAX,
                    )?; }
                }
            },
            None => {
                return Err(anyhow!("Frame index {} > Number of images in flight {}",
                                   frame_index, self.images_in_flight.len()));
            }
        }
        Ok(())
    }

    pub fn slot_in_flight_fence_to_image_in_flight(
        &mut self, 
        in_flight_fence_index: usize, 
        image_in_flight_index: usize
    ) {
        if in_flight_fence_index >= self.in_flight_fences.len() ||
            image_in_flight_index >= self.images_in_flight.len() {
            return;
        }
        self.images_in_flight[image_in_flight_index] =
            self.in_flight_fences[in_flight_fence_index];
    }

    pub fn get_image_available_semaphore(&self, frame_index: usize) -> Option<&vk::Semaphore> {
        self.image_available_semaphores.get(frame_index)
    }

    pub fn get_image_available_semaphore_unchecked(&self, frame_index: usize) -> vk::Semaphore {
        self.image_available_semaphores[frame_index]
    }

    pub fn get_render_finished_semaphores_unchecked(&self, frame_index: usize) -> vk::Semaphore {
        self.render_finished_semaphores[frame_index]
    }

    pub fn get_in_flight_fence_unchecked(&self, frame_index: usize) -> vk::Fence {
        self.in_flight_fences[frame_index]
    }

    pub fn reset_images_in_flight(&mut self) {
        self.images_in_flight.fill(vk::Fence::null());
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            destroy_semaphores(device, &self.render_finished_semaphores);
            destroy_semaphores(device, &self.image_available_semaphores);
            destroy_fences(device, &self.in_flight_fences);
        }
    }
}

pub unsafe fn create_sync_objects(device: &Device, swapchain_images: &[vk::Image]) 
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

pub unsafe fn destroy_semaphores(device: &Device, semaphores: &[vk::Semaphore]) {
    semaphores.iter()
        .for_each(|s| device.destroy_semaphore(*s, None));
}

pub unsafe fn destroy_fences(device: &Device, fences: &[vk::Fence]) {
    fences.iter()
        .for_each(|f| device.destroy_fence(*f, None));
}
