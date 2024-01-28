use anyhow::Result;
use cgmath::vec3;
use tobj::{self};
use std::{path::Path, fmt::{Display, Debug}};
use vulkanalia::prelude::v1_0::*;

use super::wrappers::{UniformBufferSeries, VertexBuffer, IndexBuffer, Vertex};
use super::descriptor;

type Vec3 = cgmath::Vector3<f32>;
type Index = u16;

struct CPUMesh {
    vertices: Vec<Vertex>,
    indices: Vec<Index>,
}

struct GPUMesh {
    vertex_buffer: VertexBuffer,
    index_buffer: IndexBuffer,
}

struct Graphics {
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

    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    images_in_flight: Vec<vk::Fence>,

    // on swapchain
    swapchain: Swapchain,

    pipeline: vk::Pipeline,
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    framebuffers: Vec<vk::Framebuffer>,
    descriptor_pool : descriptor::Pool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_sets : Vec<vk::DescriptorSet>,

    // on mesh change
    command_buffers: Vec<vk::CommandBuffer>,
    uniform_buffer_series: UniformBufferSeries,
}


impl Graphics {
    pub fn load<P>(&self, path: P) -> Vec<CPUMesh> where P : AsRef<Path> + Display + Debug {
        let loaded_obj_result = tobj::load_obj(&path, &tobj::GPU_LOAD_OPTIONS);

        let error_message = std::format!("Failed to load Obj file: {}", path);
        let (models, materials) = loaded_obj_result.expect(
            &error_message
        );

        let mut results : Vec<CPUMesh> = vec![];

        for (model_index, model) in models.iter().enumerate() {
            let mesh = &model.mesh;

            let mut vertices : Vec<Vertex> = vec![];
            let mut indices: Vec<Index> = vec![];

            let mut next_face = 0;

            for face in 0..mesh.face_arities.len() {
                let face_end = next_face + mesh.face_arities[face] as usize;
                let face_indices: Vec<_> = mesh.indices[next_face..face_end].iter().collect();

                for index in mesh.indices[next_face..face_end].iter() {
                    indices.push((*index) as Index);
                }

                next_face = face_end;
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

        Ok(GPUMesh { vertex_buffer, index_buffer })
    }
    pub unsafe fn unload_from_gpu(&self, mesh: &GPUMesh) -> Result<()> {
        VertexBuffer::destroy(mesh.vertex_buffer, &self.device);
        IndexBuffer::destroy(mesh.index_buffer, &self.device);
        Ok(())
    }
}
