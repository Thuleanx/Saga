use anyhow::{anyhow, Result};
use cgmath::vec3;
use log::info;
use tobj::{self};
use std::{path::Path, fmt::Debug, time::Instant};
use vulkanalia::{prelude::v1_0::*, vk::{DebugUtilsMessengerEXT, KhrSwapchainExtension}};
use winit::window::Window;
use vulkanalia::loader::{LibloadingLoader, LIBRARY};
use crate::core::{graphics::renderpass, config::MAX_FRAMES_IN_FLIGHT};

use super::{
    wrappers::{VertexBuffer, IndexBuffer, Vertex, uniform_buffer}, 
    instance, window_surface, physical_device, logical_device, descriptor, 
    swapchain::{Swapchain, self}, pipeline, framebuffer, command_buffers::{self, record_command_buffers}, sync_objects::GraphicsBarriers, validation_layers};

pub use uniform_buffer::UniformBufferSeries;

type Vec3 = cgmath::Vector3<f32>;
type Index = u16;

pub struct CPUMesh {
    vertices: Vec<Vertex>,
    indices: Vec<Index>,
}

pub struct GPUMesh {
    triangles_count: usize,
    vertex_buffer: VertexBuffer,
    index_buffer: IndexBuffer,
}

impl GPUMesh {
    pub unsafe fn bind(&self, graphics: &Graphics, command_buffer: vk::CommandBuffer) {
        self.bind_manual(graphics.get_device(), command_buffer);
    }

    pub unsafe fn draw(&self, graphics: &Graphics, command_buffer: vk::CommandBuffer) {
        self.draw_manual(graphics.get_device(), command_buffer)
    }

    pub(super) unsafe fn bind_manual(&self, device: &Device, command_buffer: vk::CommandBuffer) {
        let first_binding = 0;
        let memory_offset = 0;

        unsafe { self.vertex_buffer.bind(device, command_buffer, 
                                         first_binding, memory_offset); }
        unsafe { self.index_buffer.bind(device, command_buffer, 
                                        memory_offset); }
    }

    pub(super) unsafe fn draw_manual(&self, device: &Device, command_buffer: vk::CommandBuffer) {
        unsafe { device.cmd_draw_indexed(
            command_buffer, 
            (self.triangles_count * 3) as u32, 
            1, 
            0, 
            0, 
            0
        ); }
    }
}

#[derive(Copy, Clone)]
pub enum GraphicsEvent {
    Initialize,
    SwapchainDestroy,
    SwapchainRecreate,
    Destroy,
}

pub struct Graphics {
    instance: Instance,
    entry: Entry,
    device: Device,
    current_frame: usize,
    resized: bool,

    messenger: vk::DebugUtilsMessengerEXT,
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    command_pool: vk::CommandPool,

    graphics_barriers: GraphicsBarriers,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: descriptor::Pool,

    // on swapchain
    swapchain: Swapchain,

    pipeline: vk::Pipeline,
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    framebuffers: Vec<vk::Framebuffer>,

    // on mesh change
    command_buffers: Vec<vk::CommandBuffer>,

    start: Instant,
}

impl Graphics {
    pub fn create(window: &Window) -> Result<Self> {
        let size = window.inner_size();
        let loader = unsafe { LibloadingLoader::new(LIBRARY)? };
        let entry = unsafe { Entry::new(loader) }.map_err(|b| anyhow!("{}", b))?;

        let (instance, optional_messenger) 
            = unsafe {instance::create_instance(window, &entry)?};
        let surface: vk::SurfaceKHR 
            = unsafe {window_surface::create_window_surface(&instance, window)?};
        let physical_device: vk::PhysicalDevice 
            = unsafe {physical_device::pick_physical_device(&instance, surface)}?;
        let (device, graphics_queue, present_queue)
            = unsafe {logical_device::create_logical_device(&entry, &instance, surface, physical_device)?};
        let swapchain: Swapchain 
            = unsafe {swapchain::Swapchain::new(window, &instance, &device, surface, physical_device)?};
        let render_pass
            = unsafe {renderpass::create_render_pass(&instance, &device, swapchain.get_format())?};
        let descriptor_set_layout: vk::DescriptorSetLayout 
            = unsafe {descriptor::layout::create(
                &device,
                &[
                    descriptor::layout::UBODescription {
                        binding: 0, 
                        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                        descriptor_count: 1,
                        stage_flags: vk::ShaderStageFlags::VERTEX
                    },
                ])?
            };
        let (pipeline_layout, pipeline)
            = unsafe { pipeline::create_pipeline(
                &device, swapchain.get_extent(),
                descriptor_set_layout, render_pass)?
            };
        let framebuffers
            = unsafe { framebuffer::create_framebuffers(
                &device, swapchain.get_image_views(), render_pass, swapchain.get_extent())?
            };
        let command_pool
            = unsafe { command_buffers::create_command_pool(&instance, &device, 
                                                surface, physical_device)? };
        let graphics_barriers 
            = GraphicsBarriers::new(&device, swapchain.get_images() )?;
        let command_buffers
            = unsafe { command_buffers::allocate_command_buffers(
                    &device, command_pool, swapchain.get_length() as u32)?};

        let descriptor_pool = unsafe {
            descriptor::pool::create(
                &device, 
                vk::DescriptorType::UNIFORM_BUFFER, 
                swapchain.get_length() as u32
            )?
        };

        Ok(Self { 
            instance, entry, device, 
            current_frame: 0, 
            resized: false, 
            messenger: match optional_messenger {
                Some(messenger) => messenger,
                None => DebugUtilsMessengerEXT::default()
            }, 
            physical_device, surface, graphics_queue, present_queue, 
            command_pool, graphics_barriers, 
            descriptor_set_layout, descriptor_pool, swapchain, pipeline, 
            render_pass, pipeline_layout, framebuffers, command_buffers, 
            start: Instant::now()
        })
    }
}

pub enum StartRenderResult {
    Normal(Result<usize>),
    ShouldRecreateSwapchain,
}


impl Graphics {
    pub fn get_start_time(&self) -> Instant { self.start }

    pub(super) fn get_device(&self) -> &Device { &self.device }

    pub unsafe fn record_command_buffers<F>(
        &self, record_function: F
    ) -> Result<()> where F : Fn(&Self, vk::CommandBuffer, usize) -> (){
        record_command_buffers(
            &self.device, 
            &self.command_buffers, 
            self.swapchain.get_extent(), 
            self.render_pass, 
            self.pipeline, 
            &self.framebuffers, 
            record_function, self
        )?;

        Ok(())
    }

    pub unsafe fn start_render(
        &mut self, 
        window: &Window, 
    ) -> StartRenderResult {

        match self.graphics_barriers.wait_for_in_flight_fence(&self.device, self.current_frame) {
            Ok(it) => it,
            Err(e) => return StartRenderResult::Normal(Err(e)),
        };

        let result = 
        match self.graphics_barriers.get_image_available_semaphore(self.current_frame) {
            Some(semaphore) => {
                self.device.acquire_next_image_khr(
                    self.swapchain.get_chain(),
                    u64::MAX,
                    semaphore.clone(),
                    vk::Fence::null(),
                )
            }
            None => return StartRenderResult::Normal(Err(anyhow!(
                "Current frame index {} is out of range of available semaphores structure {}", 
                self.current_frame, self.graphics_barriers.get_length()))),
        };

        let image_index = match result {
            Ok((image_index, _)) => image_index as usize,
            Err(vk::ErrorCode::OUT_OF_DATE_KHR) => return StartRenderResult::ShouldRecreateSwapchain,
            Err(e) => return StartRenderResult::Normal(Err(anyhow!(e))),
        };

        match self.graphics_barriers.wait_for_image_in_flight(&self.device, image_index) {
            Ok(it) => it,
            Err(e) => return StartRenderResult::Normal(Err(e)),
        };

        self.graphics_barriers.slot_in_flight_fence_to_image_in_flight(
            self.current_frame, image_index);

        StartRenderResult::Normal(Ok(image_index))
    }

    pub unsafe fn end_render(
        &mut self, 
        window: &Window, 
        image_index: usize, 
    ) -> Result<bool> {

        let command_buffers = &[self.command_buffers[image_index]];
        let wait_semaphores = &[self.graphics_barriers
            .get_image_available_semaphore_unchecked(self.current_frame)];
        let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = &[self.graphics_barriers
            .get_render_finished_semaphores_unchecked(self.current_frame)];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores);

        let in_flight_fence = self.graphics_barriers
            .get_in_flight_fence_unchecked(self.current_frame);
        self.device.reset_fences(&[in_flight_fence])?;

        self.device.queue_submit(self.graphics_queue, &[submit_info], in_flight_fence)?;

        let swapchains = &[self.swapchain.get_chain()];
        let image_indices = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);

        let result = self.device.queue_present_khr(
            self.present_queue, &present_info);
        let present_queue_changed = result == Ok(vk::SuccessCode::SUBOPTIMAL_KHR) ||
            result == Err(vk::ErrorCode::OUT_OF_DATE_KHR);

        let should_recreate_swapchain = self.resized || present_queue_changed;
        if should_recreate_swapchain {
            self.resized = false;
        } else if let Err(e) = result {
            return Err(anyhow!(e));
        }

        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;

        Ok(should_recreate_swapchain)
    }

    pub unsafe fn device_wait_idle(&self) -> Result<()> {
        self.device.device_wait_idle()?;
        Ok(())
    }

    pub unsafe fn recreate_swapchain(
        &mut self, 
        window: &Window, 
    ) -> Result<()> {
        unsafe {
            self.destroy_swapchain();
            self.swapchain = unsafe {swapchain::Swapchain::new(window,
                &self.instance, &self.device, 
                self.surface, self.physical_device)?};
            self.render_pass = unsafe {renderpass::create_render_pass(
                &self.instance, &self.device, self.swapchain.get_format())?};
            (self.pipeline_layout, self.pipeline)
                = unsafe { pipeline::create_pipeline(
                    &self.device, self.swapchain.get_extent(),
                    self.descriptor_set_layout, self.render_pass)?
                };
            self.framebuffers
                = unsafe { framebuffer::create_framebuffers(
                    &self.device, self.swapchain.get_image_views(), 
                    self.render_pass, self.swapchain.get_extent())?
                };
            self.command_buffers = unsafe { 
                command_buffers::allocate_command_buffers(
                    &self.device, self.command_pool, 
                    self.swapchain.get_length() as u32)?
            };
        }

        Ok(())
    }

    pub unsafe fn continue_after_swapchain_construction(&mut self) {
        self.graphics_barriers.reset_images_in_flight();
    }

    pub fn trigger_resize(&mut self) {
        self.resized = true;
    }

    pub unsafe fn free_command_buffers(&self) {
        self.device.free_command_buffers(self.command_pool, &self.command_buffers);
    }

    pub unsafe fn free_descriptor_sets(&self, descriptor_sets: &[vk::DescriptorSet]) -> Result<()> {
        descriptor::set::free(&self.device, self.descriptor_pool, descriptor_sets)
    }

    unsafe fn destroy_swapchain(&mut self) {
        unsafe { 
            framebuffer::destroy_framebuffers(&self.device, &self.framebuffers);
            pipeline::destroy_pipeline(&self.device, self.pipeline, self.pipeline_layout);
            renderpass::destroy_render_pass(&self.device, self.render_pass);
            self.swapchain.destroy(&self.device); 
        }
    }

    pub fn destroy(&mut self) {
        unsafe { 
            self.destroy_swapchain(); 
            descriptor::layout::destroy(&self.device, self.descriptor_set_layout);
            descriptor::pool::destroy(&self.device, &self.descriptor_pool);

            self.graphics_barriers.destroy(&self.device);

            command_buffers::destroy_command_pool(&self.device, self.command_pool);

            logical_device::destroy_logical_device(&self.device);
            window_surface::destroy_window_surface(&self.instance, self.surface);

            validation_layers::destroy_debug_messenger(&self.instance, self.messenger);
            instance::destroy_instance(&self.instance);
        }
    }
}

impl Graphics {
    pub fn load<P>(&self, path: P) -> Vec<CPUMesh> where P : AsRef<Path> + Debug {
        let loaded_obj_result = tobj::load_obj(path.as_ref(), &tobj::GPU_LOAD_OPTIONS);

        info!("Loading mesh at {:?}", path);

        let error_message = std::format!("Failed to load Obj file: {}", path.as_ref().display());
        let (models, materials) = loaded_obj_result.expect(
            &error_message
        );

        let mut results : Vec<CPUMesh> = vec![];

        for (model_index, model) in models.iter().enumerate() {
            let mesh = &model.mesh;

            let mut vertices : Vec<Vertex> = vec![];
            let mut indices: Vec<Index> = vec![];

            info!("Mesh has {} faces {} indices", mesh.face_arities.len(), mesh.indices.len());

            static INDICES_PER_FACE : usize = 3;

            for face in 0..mesh.indices.len() / INDICES_PER_FACE {
                let face_start = face * INDICES_PER_FACE;
                let face_end = face_start + INDICES_PER_FACE;
                for index in mesh.indices[face_start..face_end].iter() {
                    indices.push((*index) as Index);
                }
            }

            for v in 0..mesh.positions.len() / 3 {
                let pos : Vec3 = vec3(
                    mesh.positions[3 * v],
                    mesh.positions[3 * v + 1],
                    mesh.positions[3 * v + 2]
                );

                vertices.push(
                    Vertex::new(
                        pos,
                        // pos
                    )
                );
            }

            info!("Loaded mesh with {} vertices and {} faces", vertices.len(), indices.len() / 3);

            results.push(
                CPUMesh{
                    vertices,
                    indices,
                }
            );

        }


        results
    }

    pub unsafe fn load_into_gpu(&self, mesh: &CPUMesh) -> Result<GPUMesh> {
        let vertex_buffer = VertexBuffer::create(
            &self.instance,
            &self.device,
            self.physical_device,
            self.command_pool, 
            self.graphics_queue, 
            &mesh.vertices
        )?;

        let index_buffer = IndexBuffer::create(
            &self.instance,
            &self.device,
            self.physical_device,
            self.command_pool, 
            self.graphics_queue, 
            &mesh.indices
        )?;

        let triangles_count : usize = mesh.indices.len() / 3;

        Ok(GPUMesh { triangles_count, vertex_buffer, index_buffer })
    }

    pub unsafe fn unload_from_gpu(&self, mesh: &GPUMesh) -> Result<()> {
        VertexBuffer::destroy(mesh.vertex_buffer, &self.device);
        IndexBuffer::destroy(mesh.index_buffer, &self.device);
        Ok(())
    }

    pub unsafe fn create_uniform_buffer_series<T>(&self) -> Result<UniformBufferSeries> {
        uniform_buffer::create_series::<T>(
            &self.instance, 
            &self.device, 
            self.physical_device, 
            self.swapchain.get_length()
        )
    }

    pub unsafe fn update_uniform_buffer_series<T>(&self, uniform_buffer_series: &UniformBufferSeries, image_index : usize, data: &T) -> Result<()> {
        uniform_buffer::update_uniform_buffer_series(
            &self.device, 
            data, 
            uniform_buffer_series, 
            image_index
        )
    }

    pub unsafe fn bind_descriptor_set(&self, command_buffer: vk::CommandBuffer, descriptor_set: vk::DescriptorSet) {
        unsafe {
            self.device.cmd_bind_descriptor_sets(command_buffer, vk::PipelineBindPoint::GRAPHICS, 
                self.pipeline_layout, 0, &[descriptor_set], &[]);
        }
    }

    pub unsafe fn destroy_uniform_buffer_series(&self, uniform_buffers: &UniformBufferSeries) {
        uniform_buffer::destroy_series(&self.device, uniform_buffers);
    }

    pub unsafe fn create_descriptor_sets<T>(&self, 
        uniform_buffer_series: &UniformBufferSeries
    ) -> Result<Vec<vk::DescriptorSet>> {
        Ok(unsafe {
            descriptor::set::create::<T>(
                &self.device, 
                &self.descriptor_pool,
                self.descriptor_set_layout,
                uniform_buffer_series.get_buffers()
            )?
        })
    }
}

