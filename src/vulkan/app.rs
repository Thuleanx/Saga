use anyhow::{anyhow, Result};

use vulkanalia::prelude::v1_0::*;
use vulkanalia::loader::{LibloadingLoader, LIBRARY};
use winit::window::Window;
use vulkanalia::vk::{KhrSwapchainExtension, Fence};
use super::appdata::AppData;

use super::config::MAX_FRAMES_IN_FLIGHT;
use std::time::Instant;

use super::{
    command_buffers,
    framebuffer,
    instance, 
    logical_device,
    physical_device, 
    pipeline,
    renderpass,
    swapchain,
    sync_objects,
    validation_layers,
    window_surface, 
    vertex, 
    uniform_buffer_object, descriptor_set,
};

/// Our Vulkan app.
#[derive(Clone, Debug)]
pub struct App {
    pub entry: Entry,
    pub instance: Instance,
    pub data: AppData,
    pub device: Device,
    pub frame: usize,
    pub resized: bool,
    pub start: Instant,
}

impl App {
    /// Creates our Vulkan app.
    pub unsafe fn create(window: &Window) -> Result<Self> {
        let loader = LibloadingLoader::new(LIBRARY)?;
        let entry = Entry::new(loader).map_err(|b| anyhow!("{}", b))?;
        let mut data = AppData::default();
        let instance = instance::create_instance(window, &entry, &mut data)?;

        window_surface::create_window_surface(&instance, window, &mut data)?;
        physical_device::pick_physical_device(&instance, &mut data)?;

        let device = logical_device::create_logical_device(&entry, &instance, &mut data)?;
        swapchain::create_swapchain(window, &instance, &device, &mut data)?;
        swapchain::create_swapchain_image_views(&device, &mut data)?;
        renderpass::create_render_pass(&instance, &device, &mut data)?;

        uniform_buffer_object::create_descriptor_set_layout(&device, &mut data)?;
        pipeline::create_pipeline(&device, &mut data)?;

        framebuffer::create_framebuffers(&device, &mut data)?;
        command_buffers::create_command_pool(&instance, &device, &mut data)?;
        vertex::create_vertex_buffer(&instance, &device, &mut data)?;
        vertex::create_index_buffer(&instance, &device, &mut data)?;
        uniform_buffer_object::create_uniform_buffers(&instance, &device, &mut data)?;
        descriptor_set::create_descriptor_pool(&device, &mut data)?;
        descriptor_set::create_descriptor_sets(&device, &mut data)?;
        command_buffers::create_command_buffers(&device, &mut data)?;
        sync_objects::create_sync_objects(&device, &mut data)?;

        Ok(Self { entry, instance, data, device, frame: 0, resized: false, start: Instant::now() })
    }

    /// Renders a frame for our Vulkan app.
    pub unsafe fn render(&mut self, window: &Window) -> Result<()> {
        let in_flight_fence : Fence = self.data.in_flight_fences[self.frame];

        self.device.wait_for_fences(
            &[in_flight_fence],
            true,
            u64::MAX,
        )?;

        let result = self.device.acquire_next_image_khr(
            self.data.swapchain,
            u64::MAX,
            self.data.image_available_semaphores[self.frame],
            vk::Fence::null(),
        );

        let image_index = match result {
            Ok((image_index, _)) => image_index as usize,
            Err(vk::ErrorCode::OUT_OF_DATE_KHR) => return self.recreate_swapchain(window),
            Err(e) => return Err(anyhow!(e)),
        };

        let image_in_flight : Fence = self.data.images_in_flight[image_index];
        if !image_in_flight.is_null() {
            self.device.wait_for_fences(
                &[image_in_flight],
                true,
                u64::MAX,
            )?;
        }

        self.data.images_in_flight[image_index] = in_flight_fence;

        uniform_buffer_object::update_uniform_buffer(self, image_index)?;

        let wait_semaphores = &[self.data.image_available_semaphores[self.frame]];
        let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = &[self.data.command_buffers[image_index]];
        let signal_semaphores = &[self.data.render_finished_semaphores[self.frame]];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores);

        self.device.reset_fences(&[in_flight_fence])?;

        self.device.queue_submit(
            self.data.graphics_queue,
            &[submit_info], 
            in_flight_fence)?;

        let swapchains = &[self.data.swapchain];
        let image_indices = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);
        
        let result = self.device.queue_present_khr(self.data.present_queue, &present_info);
        let changed = result == Ok(vk::SuccessCode::SUBOPTIMAL_KHR) ||
            result == Err(vk::ErrorCode::OUT_OF_DATE_KHR);

        if self.resized || changed {
            self.resized = false;
            self.recreate_swapchain(window)?;
        } else if let Err(e) = result {
            return Err(anyhow!(e));
        }

        self.frame = (self.frame + 1) % MAX_FRAMES_IN_FLIGHT;

        Ok(())
    }

    /// Destroys our Vulkan app.
    pub unsafe fn destroy(&mut self) {
        self.destroy_swapchain();
        uniform_buffer_object::destroy_descriptor_set_layout(self);
        vertex::destroy_vertex_buffer(self);
        vertex::destroy_index_buffer(self);

        sync_objects::destroy_sync_objects(self);
        command_buffers::destroy_command_pool(self);

        logical_device::destroy_logical_device(self);
        physical_device::destroy_physical_device(self);
        window_surface::destroy_window_surface(self);

        validation_layers::destroy_debug_messenger(self);
        instance::destroy_instance(self);
    }

    pub unsafe fn recreate_swapchain(&mut self, window: &Window) -> Result<()> {
        self.device.device_wait_idle()?;
        self.destroy_swapchain();
        swapchain::create_swapchain(window, &self.instance, &self.device, &mut self.data)?;
        swapchain::create_swapchain_image_views(&self.device, &mut self.data)?;
        renderpass::create_render_pass(&self.instance, &self.device, &mut self.data)?;
        pipeline::create_pipeline(&self.device, &mut self.data)?;
        framebuffer::create_framebuffers(&self.device, &mut self.data)?;
        uniform_buffer_object::create_uniform_buffers(&self.instance, &self.device, &mut self.data)?;
        descriptor_set::create_descriptor_pool(&self.device, &mut self.data)?;
        descriptor_set::create_descriptor_sets(&self.device, &mut self.data)?;
        command_buffers::create_command_buffers(&self.device, &mut self.data)?;
        self.data
            .images_in_flight
            .resize(self.data.swapchain_images.len(), vk::Fence::null());

        Ok(())
    }

    unsafe fn destroy_swapchain(&mut self) {
        self.device.free_command_buffers(self.data.command_pool, &self.data.command_buffers);
        uniform_buffer_object::destroy_uniform_buffers(self);
        descriptor_set::destroy_descriptor_pool(self);
        self.data.framebuffers
            .iter()
            .for_each(|f| self.device.destroy_framebuffer(*f, None));
        self.device.destroy_pipeline(self.data.pipeline, None);
        self.device.destroy_pipeline_layout(self.data.pipeline_layout, None);
        self.device.destroy_render_pass(self.data.render_pass, None);
        self.data.swapchain_image_views
            .iter()
            .for_each(|v| self.device.destroy_image_view(*v, None));
        self.device.destroy_swapchain_khr(self.data.swapchain, None);
    }

}
