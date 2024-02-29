use crate::core::graphics::{Graphics, UniformBufferSeries};
use anyhow::Result;
use cgmath::{point3, vec3, Deg, Matrix, Matrix4, One, Transform};

use super::{
    common_traits::{HasPosition, HasRotation},
    spectator::Spectator,
};

use vulkanalia::prelude::v1_0::*;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct UniformBufferObject {
    pub model: Mat4,
    pub view: Mat4,
    pub proj: Mat4,
}

type Vec3 = cgmath::Vector3<f32>;
type Mat3 = cgmath::Matrix3<f32>;
type Mat4 = cgmath::Matrix4<f32>;
type Quat = cgmath::Quaternion<f32>;

pub trait Camera: HasPosition + HasRotation {
    fn get_cached_projection_matrix(&self) -> Mat4;
    fn get_cached_view_matrix(&self) -> Mat4;

    fn calculate_projection_matrix(&self) -> Mat4;
    fn calculate_view_matrix(&self) -> Mat4 {
        let look: Vec3 = self.get_forward();
        let up: Vec3 = self.get_up();
        let right: Vec3 = look.cross(up);

        // this is row major
        //  in view space, camera captures in the opposite direction of the look
        //  vector by convention
        let rotation_matrix: Mat4 = Mat4::new(
            right.x, up.x, -look.x, 0.0, right.y, up.y, -look.y, 0.0, right.z, up.z, -look.z, 0.0,
            0.0, 0.0, 0.0, 1.0,
        );

        let translation_matrix: Mat4 = Mat4::new(
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            -self.get_position().x,
            -self.get_position().y,
            -self.get_position().z,
            1.0,
        );

        rotation_matrix * translation_matrix
    }
}

#[derive(Clone, Debug)]
pub struct PerspectiveCamera {
    position: Vec3,
    rotation: Quat,
    field_of_view: f32,
    far_plane_distance: f32,
    near_plane_distance: f32,
    width: u32,
    height: u32,
    view: Mat4,
    projection: Mat4,
    uniform_buffers: UniformBufferSeries,
    descriptor_sets: Vec<vk::DescriptorSet>,
}

impl PerspectiveCamera {
    pub fn set_width(&mut self, width: u32) -> () {
        self.width = width;
        self.projection = self.calculate_projection_matrix();
    }

    pub fn set_height(&mut self, height: u32) -> () {
        self.height = height;
        self.projection = self.calculate_projection_matrix();
    }

    pub unsafe fn bind(
        &self,
        graphics: &Graphics,
        command_buffer: vk::CommandBuffer,
        index: usize,
    ) {
        graphics.bind_descriptor_set(command_buffer, self.descriptor_sets[index])
    }

    pub unsafe fn before_swapchain_recreate(&mut self, graphics: &Graphics) -> Result<()> {
        graphics.destroy_uniform_buffer_series(&self.uniform_buffers);
        graphics.free_descriptor_sets(&self.descriptor_sets)?;

        Ok(())
    }

    pub unsafe fn after_swapchain_recreate(&mut self, graphics: &Graphics) -> Result<()> {
        self.uniform_buffers =
            unsafe { graphics.create_uniform_buffer_series::<UniformBufferObject>()? };

        self.descriptor_sets = unsafe { graphics.create_descriptor_sets()? };

        unsafe { graphics.bind_uniform_buffer::<UniformBufferObject>(&self.uniform_buffers, &self.descriptor_sets, 0) }

        Ok(())
    }

    pub unsafe fn update_uniform_buffer_series(
        &self,
        graphics: &Graphics,
        image_index: usize,
    ) -> Result<()> {
        let time = graphics.get_start_time().elapsed().as_secs_f32();

        let model = Matrix4::from_axis_angle(vec3(0.0, 0.0, 1.0), Deg(90.0 * time))
            * Matrix4::from_axis_angle(vec3(1.0, 0.0, 0.0), Deg(90.0));

        let view = self.get_cached_view_matrix();
        let proj = self.get_cached_projection_matrix();

        let ubo = UniformBufferObject { model, view, proj };

        graphics.update_uniform_buffer_series(&self.uniform_buffers, image_index, &ubo)
    }

    pub unsafe fn unload(&self, graphics: &Graphics) -> Result<()> {
        unsafe {
            graphics.destroy_uniform_buffer_series(&self.uniform_buffers);
            graphics.free_descriptor_sets(&self.descriptor_sets)?;
        }
        Ok(())
    }
}

impl Default for PerspectiveCamera {
    fn default() -> Self {
        Self {
            position: vec3(0.0, 0.0, 0.0),
            rotation: Quat::one(),
            field_of_view: Default::default(),
            far_plane_distance: Default::default(),
            near_plane_distance: Default::default(),
            width: Default::default(),
            height: Default::default(),
            view: Mat4::one(),
            projection: Mat4::one(),
            uniform_buffers: UniformBufferSeries::default(),
            descriptor_sets: vec![],
        }
    }
}

impl Spectator for PerspectiveCamera {
    fn get_movement_speed(&self) -> f32 {
        3.0
    }
}

pub struct PerspectiveCameraBuilder {
    position: Vec3,
    rotation: Quat,
    field_of_view: f32,
    far_plane_distance: f32,
    near_plane_distance: f32,
    width: u32,
    height: u32,
}

impl PerspectiveCameraBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_position(&mut self, new_position: Vec3) -> &mut Self {
        self.position = new_position;
        self
    }

    pub fn set_rotation(&mut self, new_rotation: Quat) -> &mut Self {
        self.rotation = new_rotation;
        self
    }

    pub fn set_field_of_view(&mut self, new_field_of_view: f32) -> &mut Self {
        self.field_of_view = new_field_of_view;
        self
    }

    pub fn set_far_plane_distance(&mut self, new_far_plane_distance: f32) -> &mut Self {
        self.far_plane_distance = new_far_plane_distance;
        self
    }
    pub fn set_near_plane_distance(&mut self, new_near_plane_distance: f32) -> &mut Self {
        self.near_plane_distance = new_near_plane_distance;
        self
    }

    pub fn set_width(&mut self, new_width: u32) -> &mut Self {
        self.width = new_width;
        self
    }

    pub fn set_height(&mut self, new_height: u32) -> &mut Self {
        self.height = new_height;
        self
    }

    pub fn build(&self, graphics: &Graphics) -> Result<PerspectiveCamera> {
        let uniform_buffers =
            unsafe { graphics.create_uniform_buffer_series::<UniformBufferObject>()? };

        let descriptor_sets: Vec<vk::DescriptorSet> = unsafe { graphics.create_descriptor_sets()? };

        unsafe { graphics.bind_uniform_buffer::<UniformBufferObject>(&uniform_buffers, &descriptor_sets, 0) }

        let mut builder_result = PerspectiveCamera {
            position: self.position,
            rotation: self.rotation,
            field_of_view: self.field_of_view,
            far_plane_distance: self.far_plane_distance,
            near_plane_distance: self.near_plane_distance,
            width: self.width,
            height: self.height,
            view: Mat4::one(),
            projection: Mat4::one(),
            uniform_buffers,
            descriptor_sets,
        };

        builder_result.view = builder_result.calculate_view_matrix();
        builder_result.projection = builder_result.calculate_projection_matrix();

        Ok(builder_result)
    }
}

impl Default for PerspectiveCameraBuilder {
    fn default() -> Self {
        Self {
            position: vec3(4.0, 4.0, 4.0),
            rotation: Mat3::look_at_rh(
                point3(0.0, 0.0, 0.0),
                point3(2.0, 2.0, 1.1),
                vec3(0.0, 0.0, 1.0),
            )
            .transpose()
            .into(),
            field_of_view: (45.0 as f32).to_radians(),
            far_plane_distance: 10.0,
            near_plane_distance: 0.1,
            width: 1920,
            height: 1080,
        }
    }
}

impl HasPosition for PerspectiveCamera {
    fn get_position(&self) -> Vec3 {
        self.position
    }
    fn set_position(&mut self, new_position: Vec3) -> () {
        self.position = new_position;
        self.view = self.calculate_view_matrix();
    }
}

impl HasRotation for PerspectiveCamera {
    fn get_rotation(&self) -> Quat {
        self.rotation
    }
    fn set_rotation(&mut self, new_rotation: Quat) -> () {
        self.rotation = new_rotation;
        self.view = self.calculate_view_matrix();
    }
}

impl Camera for PerspectiveCamera {
    fn get_cached_projection_matrix(&self) -> Mat4 {
        self.projection
    }
    fn get_cached_view_matrix(&self) -> Mat4 {
        self.view
    }

    fn calculate_projection_matrix(&self) -> Mat4 {
        // this matrix transforms the typical view space into an intermediate space
        // where the forward direction is the direction that the camera
        // fulstrum captures, and the up direction is the downwards camera direction
        let intermediate_matrix: Mat4 = Mat4::new(
            1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        );

        let inverse_tan_half_fov: f32 = 1.0 / (self.field_of_view / 2.0).tan();
        let inverse_aspect_ratio: f32 = (self.width as f32) / (self.height as f32);

        let projection_matrix_c0r0: f32 = inverse_tan_half_fov * inverse_aspect_ratio;
        let projection_matrix_c1r1: f32 = inverse_tan_half_fov;
        let projection_matrix_c2r2: f32 =
            self.far_plane_distance / (self.far_plane_distance - self.near_plane_distance);
        let projection_matrix_c2r3: f32 = 1.0;
        let projection_matrix_c3r2: f32 = -self.far_plane_distance * self.near_plane_distance
            / (self.far_plane_distance - self.near_plane_distance);

        // according to the following:
        // https://johannesugb.github.io/gpu-programming/setting-up-a-proper-vulkan-projection-matrix/
        // this should transform the intermediate space into clip space in vulkan
        let projection_matrix: Mat4 = Mat4::new(
            projection_matrix_c0r0,
            0.0,
            0.0,
            0.0,
            0.0,
            projection_matrix_c1r1,
            0.0,
            0.0,
            0.0,
            0.0,
            projection_matrix_c2r2,
            projection_matrix_c2r3,
            0.0,
            0.0,
            projection_matrix_c3r2,
            0.0,
        );

        projection_matrix * intermediate_matrix
    }
}
