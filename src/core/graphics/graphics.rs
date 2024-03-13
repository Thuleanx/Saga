use super::abstraction::descriptor_allocator::DescriptorAllocator;
use super::abstraction::descriptor_writer::DescriptorWriter;
use super::wrappers::{bind_sampler_to_descriptor_sets, DepthBuffer};
use super::{
    command_buffers::{self, record_command_buffers},
    descriptor, framebuffer, instance, logical_device, physical_device, pipeline,
    swapchain::{self, Swapchain},
    sync_objects::GraphicsBarriers,
    validation_layers, window_surface,
    wrappers::{uniform_buffer, IndexBuffer, Vertex, VertexBuffer},
};
use crate::core::{config::MAX_FRAMES_IN_FLIGHT, graphics::renderpass};
use anyhow::{anyhow, Result};
use cgmath::{vec2, vec3};
use log::{info, trace};
use std::{fmt::Debug, path::Path, time::Instant};
use tobj::{self};
use vulkanalia::loader::{LibloadingLoader, LIBRARY};
use vulkanalia::{
    prelude::v1_0::*,
    vk::{DebugUtilsMessengerEXT, KhrSwapchainExtension},
};
use winit::window::Window;

type Vec3 = cgmath::Vector3<f32>;
type Vec2 = cgmath::Vector2<f32>;
type Index = u16;

pub use super::wrappers::{Image, ImageSampler, LoadedImage};
pub use uniform_buffer::UniformBufferSeries;

pub struct CPUMesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<Index>,
}

impl CPUMesh {
    pub fn get_simple_plane() -> Self {
        CPUMesh {
            vertices: vec![
                Vertex::new(cgmath::vec3(-0.5, -0.5, 0.0), cgmath::vec2(0.0, 1.0)),
                Vertex::new(cgmath::vec3(0.5, -0.5, 0.0), cgmath::vec2(1.0, 1.0)),
                Vertex::new(cgmath::vec3(-0.5, 0.5, 0.0), cgmath::vec2(0.0, 0.0)),
                Vertex::new(cgmath::vec3(0.5, 0.5, 0.0), cgmath::vec2(1.0, 0.0)),
            ],
            indices: vec![
                0, 1, 2, 2, 1, 3
            ],
        }
    }
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

        unsafe {
            self.vertex_buffer
                .bind(device, command_buffer, first_binding, memory_offset); }
        unsafe {
            self.index_buffer
                .bind(device, command_buffer, memory_offset);
        }
    }

    pub(super) unsafe fn draw_manual(&self, device: &Device, command_buffer: vk::CommandBuffer) {
        unsafe {
            device.cmd_draw_indexed(
                command_buffer,
                (self.triangles_count * 3) as u32,
                1,
                0,
                0,
                0,
            );
        }
    }
}

#[derive(bevy_ecs::system::Resource)]
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

    pub mesh_descriptor_set_layout: vk::DescriptorSetLayout,
    pub global_descriptor_set_layout: vk::DescriptorSetLayout,

    pub global_descriptor_allocator: DescriptorAllocator,
    pub mesh_descriptor_allocator: DescriptorAllocator,
    pub descriptor_writer: DescriptorWriter,
    pub global_descriptor_sets: Vec<vk::DescriptorSet>,

    // on swapchain
    pub swapchain: Swapchain,
    depth_buffer: DepthBuffer,

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

        let (instance, optional_messenger) = unsafe { instance::create_instance(window, &entry)? };
        let surface: vk::SurfaceKHR =
            unsafe { window_surface::create_window_surface(&instance, window)? };
        let physical_device: vk::PhysicalDevice =
            unsafe { physical_device::pick_physical_device(&instance, surface) }?;
        let (device, graphics_queue, present_queue) = unsafe {
            logical_device::create_logical_device(&entry, &instance, surface, physical_device)?
        };
        let swapchain: Swapchain = unsafe {
            swapchain::Swapchain::new(window, &instance, &device, surface, physical_device)?
        };

        let command_pool = unsafe {
            command_buffers::create_command_pool(&instance, &device, surface, physical_device)?
        };
        let depth_buffer: DepthBuffer = unsafe {
            DepthBuffer::new(
                &instance,
                &device,
                physical_device,
                &swapchain,
                graphics_queue,
                command_pool,
            )?
        };

        let render_pass = unsafe {
            renderpass::create_render_pass(
                &instance,
                &device,
                physical_device,
                swapchain.get_format(),
            )?
        };

        let global_descriptor_set_layout: vk::DescriptorSetLayout = unsafe {
            descriptor::layout::create(
                &device,
                &[descriptor::layout::DescriptorInfo {
                    binding: 0,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::VERTEX,
                }],
            )?
        };

        let mesh_descriptor_set_layout: vk::DescriptorSetLayout = unsafe {
            descriptor::layout::create(
                &device,
                &[
                    descriptor::layout::DescriptorInfo {
                        binding: 0,
                        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                        descriptor_count: 1,
                        stage_flags: vk::ShaderStageFlags::VERTEX,
                    },
                    descriptor::layout::DescriptorInfo {
                        binding: 1,
                        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        descriptor_count: 1,
                        stage_flags: vk::ShaderStageFlags::FRAGMENT,
                    },
                    descriptor::layout::DescriptorInfo {
                        binding: 2,
                        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                        descriptor_count: 1,
                        stage_flags: vk::ShaderStageFlags::FRAGMENT,
                    },
                ],
            )?
        };

        let (pipeline_layout, pipeline) = unsafe {
            pipeline::create_pipeline(
                &device,
                swapchain.get_extent(),
                &[global_descriptor_set_layout, mesh_descriptor_set_layout],
                render_pass,
            )?
        };
        let framebuffers = unsafe {
            framebuffer::create_framebuffers(
                &device,
                swapchain.get_image_views(),
                render_pass,
                &depth_buffer,
                swapchain.get_extent(),
            )?
        };
        let graphics_barriers = GraphicsBarriers::new(&device, swapchain.get_images())?;

        let command_buffers = unsafe {
            command_buffers::allocate_command_buffers(
                &device,
                command_pool,
                swapchain.get_length() as u32,
            )?
        };

        let mut global_descriptor_allocator = DescriptorAllocator::new(
            &device,
            &[descriptor::pool::PoolDescription {
                type_: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
            }],
            swapchain.get_length() as u32,
            1024,
        );

        let mesh_descriptor_allocator = DescriptorAllocator::new(
            &device,
            &[
                descriptor::pool::PoolDescription {
                    type_: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: 2,
                },
                descriptor::pool::PoolDescription {
                    type_: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: 1,
                },
            ],
            swapchain.get_length() as u32,
            1024,
        );

        let descriptor_writer = DescriptorWriter::default();

        let global_descriptor_sets = unsafe {
            global_descriptor_allocator.allocate(
                &device,
                global_descriptor_set_layout,
                swapchain.get_length(),
            )?
        };

        Ok(Self {
            instance,
            entry,
            device,
            current_frame: 0,
            resized: false,
            messenger: match optional_messenger {
                Some(messenger) => messenger,
                None => DebugUtilsMessengerEXT::default(),
            },
            physical_device,
            surface,
            graphics_queue,
            present_queue,
            command_pool,
            graphics_barriers,
            mesh_descriptor_set_layout,
            global_descriptor_set_layout,
            swapchain,
            depth_buffer,
            pipeline,
            render_pass,
            pipeline_layout,
            framebuffers,
            command_buffers,
            start: Instant::now(),
            global_descriptor_allocator,
            mesh_descriptor_allocator,
            descriptor_writer,
            global_descriptor_sets,
        })
    }
}

pub enum StartRenderResult {
    Normal(Result<usize>),
    ShouldRecreateSwapchain,
}

impl Graphics {
    pub fn get_start_time(&self) -> Instant {
        self.start
    }

    pub fn get_device(&self) -> &Device {
        &self.device
    }

    pub fn get_swapchain_extent(&self) -> vk::Extent2D {
        self.swapchain.get_extent()
    }

    pub unsafe fn record_command_buffers<F>(&self, record_function: F) -> Result<()>
    where
        F: Fn(&Self, vk::CommandBuffer, usize) -> (),
    {
        record_command_buffers(
            &self.device,
            &self.command_buffers,
            self.swapchain.get_extent(),
            self.render_pass,
            self.pipeline,
            &self.framebuffers,
            record_function,
            self,
        )?;

        Ok(())
    }

    pub unsafe fn start_render(&mut self, window: &Window) -> StartRenderResult {
        match self
            .graphics_barriers
            .wait_for_in_flight_fence(&self.device, self.current_frame)
        {
            Ok(it) => it,
            Err(e) => return StartRenderResult::Normal(Err(e)),
        };

        let result =
            match self
                .graphics_barriers
                .get_image_available_semaphore(self.current_frame)
            {
                Some(semaphore) => self.device.acquire_next_image_khr(
                    self.swapchain.get_chain(),
                    u64::MAX,
                    semaphore.clone(),
                    vk::Fence::null(),
                ),
                None => {
                    return StartRenderResult::Normal(Err(anyhow!(
                "Current frame index {} is out of range of available semaphores structure {}", 
                self.current_frame, self.graphics_barriers.get_length())))
                }
            };

        let image_index = match result {
            Ok((image_index, _)) => image_index as usize,
            Err(vk::ErrorCode::OUT_OF_DATE_KHR) => {
                trace!("Image index grab error");
                return StartRenderResult::ShouldRecreateSwapchain;
            }
            Err(e) => return StartRenderResult::Normal(Err(anyhow!(e))),
        };

        match self
            .graphics_barriers
            .wait_for_image_in_flight(&self.device, image_index)
        {
            Ok(it) => it,
            Err(e) => return StartRenderResult::Normal(Err(e)),
        };

        self.graphics_barriers
            .slot_in_flight_fence_to_image_in_flight(self.current_frame, image_index);

        StartRenderResult::Normal(Ok(image_index))
    }

    pub unsafe fn end_render(&mut self, window: &Window, image_index: usize) -> Result<bool> {
        let command_buffers = &[self.command_buffers[image_index]];
        let wait_semaphores = &[self
            .graphics_barriers
            .get_image_available_semaphore_unchecked(self.current_frame)];
        let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = &[self
            .graphics_barriers
            .get_render_finished_semaphores_unchecked(self.current_frame)];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores);

        let in_flight_fence = self
            .graphics_barriers
            .get_in_flight_fence_unchecked(self.current_frame);
        self.device.reset_fences(&[in_flight_fence])?;

        self.device
            .queue_submit(self.graphics_queue, &[submit_info], in_flight_fence)?;

        let swapchains = &[self.swapchain.get_chain()];
        let image_indices = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);

        let result = self
            .device
            .queue_present_khr(self.present_queue, &present_info);
        let present_queue_changed = result == Ok(vk::SuccessCode::SUBOPTIMAL_KHR)
            || result == Err(vk::ErrorCode::OUT_OF_DATE_KHR);

        let should_recreate_swapchain = self.resized || present_queue_changed;
        if should_recreate_swapchain {
            trace!(
                "Swapchain recreation queued resized: {} suboptimal {} out of date {}",
                self.resized,
                result == Ok(vk::SuccessCode::SUBOPTIMAL_KHR),
                result == Err(vk::ErrorCode::OUT_OF_DATE_KHR)
            );
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

    pub unsafe fn recreate_swapchain(&mut self, window: &Window) -> Result<()> {
        unsafe {
            self.destroy_swapchain();
            self.swapchain = unsafe {
                swapchain::Swapchain::new(
                    window,
                    &self.instance,
                    &self.device,
                    self.surface,
                    self.physical_device,
                )?
            };
            self.depth_buffer = unsafe {
                DepthBuffer::new(
                    &self.instance,
                    &self.device,
                    self.physical_device,
                    &self.swapchain,
                    self.graphics_queue,
                    self.command_pool,
                )?
            };
            self.render_pass = unsafe {
                renderpass::create_render_pass(
                    &self.instance,
                    &self.device,
                    self.physical_device,
                    self.swapchain.get_format(),
                )?
            };
            (self.pipeline_layout, self.pipeline) = unsafe {
                pipeline::create_pipeline(
                    &self.device,
                    self.swapchain.get_extent(),
                    &[
                        self.global_descriptor_set_layout,
                        self.mesh_descriptor_set_layout,
                    ],
                    self.render_pass,
                )?
            };
            self.framebuffers = unsafe {
                framebuffer::create_framebuffers(
                    &self.device,
                    self.swapchain.get_image_views(),
                    self.render_pass,
                    &self.depth_buffer,
                    self.swapchain.get_extent(),
                )?
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
        self.device
            .free_command_buffers(self.command_pool, &self.command_buffers);
    }

    unsafe fn destroy_swapchain(&mut self) {
        unsafe {
            framebuffer::destroy_framebuffers(&self.device, &self.framebuffers);
            pipeline::destroy_pipeline(&self.device, self.pipeline, self.pipeline_layout);
            renderpass::destroy_render_pass(&self.device, self.render_pass);
            self.depth_buffer.destroy(&self.device);
            self.swapchain.destroy(&self.device);
        }
    }

    pub fn destroy(&mut self) {
        unsafe {
            self.destroy_swapchain();
            descriptor::layout::destroy(&self.device, self.mesh_descriptor_set_layout);
            descriptor::layout::destroy(&self.device, self.global_descriptor_set_layout);

            self.global_descriptor_allocator.destroy(&self.device);
            self.mesh_descriptor_allocator.destroy(&self.device);

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
    pub fn load<P>(&self, path: P) -> Vec<CPUMesh>
    where
        P: AsRef<Path> + Debug,
    {
        let loaded_obj_result = tobj::load_obj(path.as_ref(), &tobj::GPU_LOAD_OPTIONS);

        info!("Loading mesh at {:?}", path);

        let error_message = std::format!("Failed to load Obj file: {}", path.as_ref().display());
        let (models, materials) = loaded_obj_result.expect(&error_message);

        let mut results: Vec<CPUMesh> = vec![];

        for (model_index, model) in models.iter().enumerate() {
            let mesh = &model.mesh;

            let mut vertices: Vec<Vertex> = vec![];
            let mut indices: Vec<Index> = vec![];

            info!(
                "Mesh has {} faces {} indices",
                mesh.face_arities.len(),
                mesh.indices.len()
            );

            static INDICES_PER_FACE: usize = 3;

            for face in 0..mesh.indices.len() / INDICES_PER_FACE {
                let face_start = face * INDICES_PER_FACE;
                let face_end = face_start + INDICES_PER_FACE;
                for index in mesh.indices[face_start..face_end].iter() {
                    indices.push((*index) as Index);
                }
            }

            for v in 0..mesh.positions.len() / 3 {
                let pos: Vec3 = vec3(
                    mesh.positions[3 * v],
                    mesh.positions[3 * v + 1],
                    mesh.positions[3 * v + 2],
                );

                let uv: Vec2 = vec2(mesh.texcoords[2 * v], 1.0 - mesh.texcoords[2 * v + 1]);
                // info!(
                //     "Vertex pos ({}, {}, {}) ({}, {})",
                //     pos.x, pos.y, pos.z, uv.x, uv.y
                // );

                vertices.push(Vertex::new(pos, uv));
            }

            info!(
                "Loaded mesh with {} vertices and {} faces",
                vertices.len(),
                indices.len() / 3
            );

            results.push(CPUMesh { vertices, indices });
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
            &mesh.vertices,
        )?;

        let index_buffer = IndexBuffer::create(
            &self.instance,
            &self.device,
            self.physical_device,
            self.command_pool,
            self.graphics_queue,
            &mesh.indices,
        )?;

        let triangles_count: usize = mesh.indices.len() / 3;

        Ok(GPUMesh {
            triangles_count,
            vertex_buffer,
            index_buffer,
        })
    }

    pub unsafe fn unload_from_gpu(&self, mesh: &GPUMesh) -> Result<()> {
        VertexBuffer::destroy(mesh.vertex_buffer, &self.device);
        IndexBuffer::destroy(mesh.index_buffer, &self.device);
        Ok(())
    }

    pub unsafe fn load_texture_to_gpu(&self, image: &Image) -> Result<LoadedImage> {
        let loaded_image = unsafe {
            LoadedImage::load_into_memory(
                image,
                &self.instance,
                &self.device,
                self.physical_device,
                self.graphics_queue,
                self.command_pool,
            )?
        };

        Ok(loaded_image)
    }

    pub unsafe fn unload_texture_from_gpu(&self, loaded_image: &LoadedImage) -> Result<()> {
        loaded_image.destroy(&self.device);
        Ok(())
    }

    pub unsafe fn unload_sampler_from_gpu(&self, image_sampler: &ImageSampler) -> Result<()> {
        image_sampler.destroy(&self.device);
        Ok(())
    }

    pub unsafe fn create_image_sampler(&self) -> Result<ImageSampler> {
        ImageSampler::create(&self.device)
    }

    pub unsafe fn create_uniform_buffer_series<T>(&self) -> Result<UniformBufferSeries> {
        uniform_buffer::create_series::<T>(
            &self.instance,
            &self.device,
            self.physical_device,
            self.swapchain.get_length(),
        )
    }

    pub unsafe fn update_uniform_buffer_series<T>(
        &self,
        uniform_buffer_series: &UniformBufferSeries,
        image_index: usize,
        data: &T,
    ) -> Result<()> {
        uniform_buffer::update_uniform_buffer_series(
            &self.device,
            data,
            uniform_buffer_series,
            image_index,
        )
    }

    pub unsafe fn bind_descriptor_set(
        &self,
        command_buffer: vk::CommandBuffer,
        descriptor_sets: &[vk::DescriptorSet],
        first_set: u32,
    ) {
        unsafe {
            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                first_set,
                descriptor_sets,
                &[],
            );
        }
    }

    pub unsafe fn bind_image_sampler(
        &self,
        descriptor_sets: &[vk::DescriptorSet],
        sampler: &ImageSampler,
        image: &LoadedImage,
        binding: u32,
    ) {
        bind_sampler_to_descriptor_sets(&self.device, sampler, image, descriptor_sets, binding);
    }

    pub unsafe fn destroy_uniform_buffer_series(&self, uniform_buffers: &UniformBufferSeries) {
        uniform_buffer::destroy_series(&self.device, uniform_buffers);
    }

    pub unsafe fn bind_uniform_buffer<T>(
        &self,
        uniform_buffers: &UniformBufferSeries,
        descriptor_sets: &[vk::DescriptorSet],
        binding: u32,
    ) {
        uniform_buffers.bind_to_descriptor_sets::<T>(&self.device, descriptor_sets, binding)
    }

    pub unsafe fn reset_command_buffers(&self) -> Result<()> {
        self.command_buffers.iter().for_each(|command_buffer: &vk::CommandBuffer| {
            command_buffers::reset_command_buffer(&self.device, command_buffer.clone()).unwrap()
        });
        Ok(())
    }
}

pub mod graphics_utility {
    use anyhow::Result;
    use std::{fmt::Debug, path::Path};

    use crate::core::graphics::{
        graphics::Index,
        wrappers::{uniform_buffer, IndexBuffer, Vertex, VertexBuffer},
    };

    use super::{
        CPUMesh, GPUMesh, Graphics, Image, ImageSampler, LoadedImage, UniformBufferSeries,
    };

    pub fn descriptor_writer_write(graphics: &mut Graphics) {
        graphics.descriptor_writer.write(&graphics.device);
    }

    impl UniformBufferSeries {
        pub unsafe fn create_from_graphics<T>(graphics: &Graphics) -> Result<Self> {
            uniform_buffer::create_series::<T>(
                &graphics.instance,
                &graphics.device,
                graphics.physical_device,
                graphics.swapchain.get_length(),
            )
        }

        pub unsafe fn destroy_uniform_buffer_series(&self, graphics: &Graphics) {
            uniform_buffer::destroy_series(&graphics.device, self);
        }
    }

    impl CPUMesh {
        pub unsafe fn load_from_obj<P>(graphics: &Graphics, path: P) -> Vec<Self>
        where
            P: AsRef<Path> + Debug,
        {
            let loaded_obj_result = tobj::load_obj(path.as_ref(), &tobj::GPU_LOAD_OPTIONS);

            log::info!("Loading mesh at {:?}", path);

            let error_message =
                std::format!("Failed to load Obj file: {}", path.as_ref().display());
            let (models, materials) = loaded_obj_result.expect(&error_message);

            let mut results: Vec<CPUMesh> = vec![];

            for (model_index, model) in models.iter().enumerate() {
                let mesh = &model.mesh;

                let mut vertices: Vec<Vertex> = vec![];
                let mut indices: Vec<Index> = vec![];

                log::info!(
                    "Mesh has {} faces {} indices",
                    mesh.face_arities.len(),
                    mesh.indices.len()
                );

                static INDICES_PER_FACE: usize = 3;

                for face in 0..mesh.indices.len() / INDICES_PER_FACE {
                    let face_start = face * INDICES_PER_FACE;
                    let face_end = face_start + INDICES_PER_FACE;
                    for index in mesh.indices[face_start..face_end].iter() {
                        indices.push((*index) as Index);
                    }
                }

                for v in 0..mesh.positions.len() / 3 {
                    let pos = cgmath::vec3(
                        mesh.positions[3 * v],
                        mesh.positions[3 * v + 1],
                        mesh.positions[3 * v + 2],
                    );

                    let uv = cgmath::vec2(mesh.texcoords[2 * v], 1.0 - mesh.texcoords[2 * v + 1]);

                    vertices.push(Vertex::new(pos, uv));
                }

                log::info!(
                    "Loaded mesh with {} vertices and {} faces",
                    vertices.len(),
                    indices.len() / 3
                );

                results.push(CPUMesh { vertices, indices });
            }

            results
        }
    }

    impl GPUMesh {
        pub unsafe fn create(graphics: &Graphics, mesh: &CPUMesh) -> Result<Self> {
            let vertex_buffer = VertexBuffer::create(
                &graphics.instance,
                &graphics.device,
                graphics.physical_device,
                graphics.command_pool,
                graphics.graphics_queue,
                &mesh.vertices,
            )?;

            let index_buffer = IndexBuffer::create(
                &graphics.instance,
                &graphics.device,
                graphics.physical_device,
                graphics.command_pool,
                graphics.graphics_queue,
                &mesh.indices,
            )?;

            let triangles_count: usize = mesh.indices.len() / 3;

            Ok(Self {
                triangles_count,
                vertex_buffer,
                index_buffer,
            })
        }

        pub unsafe fn destroy(&self, graphics: &Graphics) {
            VertexBuffer::destroy(self.vertex_buffer, &graphics.device);
            IndexBuffer::destroy(self.index_buffer, &graphics.device);
        }
    }

    impl LoadedImage {
        pub unsafe fn create(graphics: &Graphics, image: &Image) -> Result<Self> {
            LoadedImage::load_into_memory(
                &image,
                &graphics.instance,
                &graphics.device,
                graphics.physical_device,
                graphics.graphics_queue,
                graphics.command_pool,
            )
        }

        pub unsafe fn destroy_with_graphics(&self, graphics: &Graphics) {
            self.destroy(&graphics.device)
        }
    }

    impl ImageSampler {
        pub unsafe fn create_from_graphics(graphics: &Graphics) -> Result<Self> {
            Self::create(&graphics.device)
        }

        pub unsafe fn destroy_with_graphics(&self, graphics: &Graphics) {
            self.destroy(&graphics.device)
        }
    }
}
