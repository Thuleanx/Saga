use anyhow::{anyhow, Result};

use vulkanalia::prelude::v1_0::*;
use vulkanalia::loader::{LibloadingLoader, LIBRARY};
use winit::window::Window;
use vulkanalia::vk::{KhrSwapchainExtension, Fence};
use crate::saga::PerspectiveCameraBuilder;

use super::appdata::AppData;

use crate::core::config::MAX_FRAMES_IN_FLIGHT;
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

        let size = window.inner_size();
        data.camera = {
            let mut camera_builder = PerspectiveCameraBuilder::default();
            camera_builder
                .set_width(size.width)
                .set_height(size.height);
            camera_builder.build()
        };

        let (instance, optional_messenger) = instance::create_instance(window, &entry)?;
        if let Some(messenger) = optional_messenger {
            data.messenger = messenger;
        }

        data.surface = window_surface::create_window_surface(&instance, window)?;
        data.physical_device = physical_device::pick_physical_device(&instance, data.surface)?;

        let device;
        (device, data.graphics_queue, data.present_queue) = 
            logical_device::create_logical_device(&entry, &instance, data.surface, data.physical_device)?;

        (data.swapchain, data.swapchain_images, data.swapchain_format, data.swapchain_extent)
            = swapchain::create_swapchain(window, &instance, &device, data.surface, data.physical_device)?;

        data.swapchain_image_views = swapchain::create_swapchain_image_views(&device, &data.swapchain_images, data.swapchain_format)?;

        data.render_pass = renderpass::create_render_pass(&instance, &device, data.swapchain_format)?;

        data.descriptor_set_layout = uniform_buffer_object::create_descriptor_set_layout(&device)?;

        (data.pipeline_layout, data.pipeline) = 
            pipeline::create_pipeline(&device, data.swapchain_extent, data.descriptor_set_layout, data.render_pass)?;

        data.framebuffers = framebuffer::create_framebuffers(
            &device, 
            &data.swapchain_image_views,
            data.render_pass,
            data.swapchain_extent
        )?;

        data.command_pool = command_buffers::create_command_pool(
            &instance, 
            &device, 
            data.surface, 
            data.physical_device,
        )?;

        (data.vertex_buffer, data.vertex_buffer_memory) = 
            vertex::create_vertex_buffer(&instance, &device, data.physical_device, 
                                         data.command_pool, data.graphics_queue)?;
        (data.index_buffer, data.index_buffer_memory) =
            vertex::create_index_buffer(&instance, &device, data.physical_device, 
                                        data.command_pool, data.graphics_queue)?;

        uniform_buffer_object::create_uniform_buffers(&instance, &device, data.physical_device,
            &mut data.uniform_buffers, &mut data.uniform_buffers_memory, &data.swapchain_images)?;

        data.descriptor_pool = descriptor_set::create_descriptor_pool(
            &device, data.swapchain_images.len() as u32)?;

        data.descriptor_sets = descriptor_set::create_descriptor_sets(
            &device, data.descriptor_set_layout, 
            data.descriptor_pool, data.swapchain_images.len(), 
            &data.uniform_buffers, &data.descriptor_sets)?;

        data.command_buffers = command_buffers::create_command_buffers(
            &device, 
            data.command_pool, data.swapchain_extent, data.render_pass, 
            data.pipeline, data.pipeline_layout, &data.framebuffers, 
            data.vertex_buffer, data.index_buffer, &data.descriptor_sets,
        )?;

        (data.image_available_semaphores, data.render_finished_semaphores, 
         data.in_flight_fences, data.images_in_flight) = sync_objects::create_sync_objects(&device, &data.swapchain_images)?;

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

        uniform_buffer_object::update_uniform_buffer(
            &self.device,
            image_index, 
            self.start,
            &mut self.data.uniform_buffers_memory, 
            &self.data.camera
        )?;

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

            let size = window.inner_size();
            self.data.camera.set_width(size.width);
            self.data.camera.set_height(size.height);
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
        uniform_buffer_object::destroy_descriptor_set_layout(
            &self.device, self.data.descriptor_set_layout);

        vertex::destroy_buffer_and_free_memory(&self.device, 
                                               self.data.vertex_buffer, 
                                               self.data.vertex_buffer_memory);
        vertex::destroy_buffer_and_free_memory(&self.device, 
                                               self.data.index_buffer, 
                                               self.data.index_buffer_memory);

        sync_objects::destroy_semaphores(&self.device, &self.data.render_finished_semaphores);
        sync_objects::destroy_semaphores(&self.device, &self.data.image_available_semaphores);
        sync_objects::destroy_fences(&self.device, &self.data.in_flight_fences);

        command_buffers::destroy_command_pool(&self.device, self.data.command_pool);

        logical_device::destroy_logical_device(&self.device);
        window_surface::destroy_window_surface(&self.instance, self.data.surface);

        validation_layers::destroy_debug_messenger(&self.instance, self.data.messenger);
        instance::destroy_instance(&self.instance);
    }

    pub unsafe fn recreate_swapchain(&mut self, window: &Window) -> Result<()> {
        self.device.device_wait_idle()?;
        self.destroy_swapchain();
        (self.data.swapchain, self.data.swapchain_images, self.data.swapchain_format, self.data.swapchain_extent)
            = swapchain::create_swapchain(window, &self.instance, &self.device, self.data.surface, self.data.physical_device)?;

        self.data.swapchain_image_views = swapchain::create_swapchain_image_views(
            &self.device, &self.data.swapchain_images, self.data.swapchain_format)?;

        self.data.render_pass = renderpass::create_render_pass(&self.instance, &self.device, self.data.swapchain_format)?;
        (self.data.pipeline_layout, self.data.pipeline) = 
            pipeline::create_pipeline(&self.device, 
                                      self.data.swapchain_extent, 
                                      self.data.descriptor_set_layout, 
                                      self.data.render_pass)?;

        self.data.framebuffers = framebuffer::create_framebuffers(
            &self.device, 
            &self.data.swapchain_image_views,
            self.data.render_pass,
            self.data.swapchain_extent
        )?;

        uniform_buffer_object::create_uniform_buffers(
            &self.instance, 
            &self.device,
            self.data.physical_device,
            &mut self.data.uniform_buffers, 
            &mut self.data.uniform_buffers_memory, 
            &self.data.swapchain_images
        )?;

        self.data.descriptor_pool = descriptor_set::create_descriptor_pool(
            &self.device, self.data.swapchain_images.len() as u32)?;

        self.data.descriptor_sets = descriptor_set::create_descriptor_sets(
            &self.device, self.data.descriptor_set_layout, 
            self.data.descriptor_pool, self.data.swapchain_images.len(), 
            &self.data.uniform_buffers, &self.data.descriptor_sets)?;

        self.data.command_buffers = command_buffers::create_command_buffers(
            &self.device, 
            self.data.command_pool, self.data.swapchain_extent, self.data.render_pass, 
            self.data.pipeline, self.data.pipeline_layout, &self.data.framebuffers, 
            self.data.vertex_buffer, self.data.index_buffer, &self.data.descriptor_sets,
        )?;

        self.data
            .images_in_flight
            .resize(self.data.swapchain_images.len(), vk::Fence::null());

        Ok(())
    }

    unsafe fn destroy_swapchain(&mut self) {
        self.device.free_command_buffers(self.data.command_pool, &self.data.command_buffers);

        uniform_buffer_object::destroy_uniform_buffers(&self.device, &mut self.data.uniform_buffers);
        uniform_buffer_object::destroy_uniform_buffers_memory(&self.device, &mut self.data.uniform_buffers_memory);

        descriptor_set::destroy_descriptor_pool(&self.device, self.data.descriptor_pool);

        self.data.framebuffers
            .iter()
            .for_each(|f| self.device.destroy_framebuffer(*f, None));

        pipeline::destroy_pipeline(&self.device, self.data.pipeline, self.data.pipeline_layout);
        renderpass::destroy_render_pass(&self.device, self.data.render_pass);

        self.data.swapchain_image_views
            .iter()
            .for_each(|v| self.device.destroy_image_view(*v, None));
        self.device.destroy_swapchain_khr(self.data.swapchain, None);
    }

}
