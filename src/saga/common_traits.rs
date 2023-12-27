use cgmath::vec3;

use super::input::Input;

type Vec3 = cgmath::Vector3<f32>;
type Mat4 = cgmath::Matrix4<f32>;
type Quat = cgmath::Quaternion<f32>;

pub trait HasPosition {
    fn get_position(&self) -> Vec3;
    fn set_position(&mut self, new_position: Vec3) -> ();
}

pub trait HasOrientation {
    fn get_rotation(&self) -> Quat;
    fn set_rotation(&mut self, new_rotation: Quat) -> ();

    fn get_forward(&self) -> Vec3 {
        self.get_rotation() * vec3(0.0, 0.0, 1.0)
    }

    fn get_right(&self) -> Vec3 {
        self.get_rotation() * vec3(-1.0, 0.0, 0.0)
    }

    fn get_up(&self) -> Vec3 {
        self.get_rotation() * vec3(0.0, 1.0, 0.0)
    }
}

pub trait GameObject {
    fn start(&mut self, input: &Input) -> (); 
    fn update(&mut self, input: &Input, delta_time: f32) -> ();
    fn end(&mut self, input: &Input) -> ();
}
