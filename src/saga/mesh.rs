use std::path::Path;

use super::{
    common_traits::{HasPosition, HasRotation},
    PerspectiveCamera,
};
use crate::core::graphics::{GPUMesh, Graphics, Image, ImageSampler, LoadedImage};
use anyhow::Result;
use cgmath::{One, Quaternion, Zero};
use vulkanalia::prelude::v1_0::*;

type Vec3 = cgmath::Vector3<f32>;
type Quat = cgmath::Quaternion<f32>;

pub struct Mesh {
    position: Vec3,
    rotation: Quat,
    gpu_mesh: GPUMesh,
    texture: LoadedImage,
    texture_sampler: ImageSampler,
}

impl Mesh {
    pub unsafe fn bind(&self, graphics: &Graphics, command_buffer: vk::CommandBuffer) {
        self.gpu_mesh.bind(graphics, command_buffer);
        // self.texture_sampler.bind_image_sampler(&graphics, &self.texture, 1);
    }

    pub unsafe fn draw(&self, graphics: &Graphics, command_buffer: vk::CommandBuffer) {
        self.gpu_mesh.draw(graphics, command_buffer);
    }

    pub unsafe fn unload(&self, graphics: &Graphics) -> Result<()> {
        graphics.unload_from_gpu(&self.gpu_mesh)?;
        graphics.unload_texture_from_gpu(&self.texture)?;
        graphics.unload_sampler_from_gpu(&self.texture_sampler)?;
        Ok(())
    }
}

impl HasPosition for Mesh {
    fn get_position(&self) -> Vec3 {
        self.position
    }
    fn set_position(&mut self, new_position: Vec3) -> () {
        self.position = new_position;
    }
}

impl HasRotation for Mesh {
    fn get_rotation(&self) -> Quat {
        self.rotation
    }
    fn set_rotation(&mut self, new_rotation: Quat) -> () {
        self.rotation = new_rotation
    }
}

pub struct MeshBuilder<'a> {
    mesh_path: &'a Path,
    texture_path: &'a Path,
    position: Option<Vec3>,
    rotation: Option<Quat>,
}

impl<'a> MeshBuilder<'a> {
    pub fn new(mesh_path: &'a Path, texture_path: &'a Path) -> Self {
        MeshBuilder {
            mesh_path,
            texture_path,
            position: None,
            rotation: None,
        }
    }
}

impl MeshBuilder<'_> {
    pub fn build(&self, graphics: &Graphics, camera: &PerspectiveCamera) -> Result<Mesh> {
        let cpu_mesh = graphics.load(&self.mesh_path);

        // discard all but the first mesh

        let gpu_mesh = unsafe { graphics.load_into_gpu(&cpu_mesh[0])? };

        let position: Vec3 = self.position.unwrap_or_else(Vec3::zero);
        let rotation: Quat = self.rotation.unwrap_or_else(Quaternion::one);

        let texture = Image::load(&self.texture_path)?;
        let loaded_texture = unsafe { graphics.load_texture_to_gpu(&texture)? };
        let texture_sampler = unsafe { graphics.create_image_sampler()? };


        unsafe {
            texture_sampler.bind_image_sampler(&graphics, &loaded_texture, 1);
        }

        Ok(Mesh {
            position,
            rotation,
            gpu_mesh,
            texture: loaded_texture,
            texture_sampler,
        })
    }
}
