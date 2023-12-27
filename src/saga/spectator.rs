use super::{common_traits::{HasOrientation, HasPosition, GameObject}, input::Input};

use cgmath::{InnerSpace, Zero};
use winit::event::VirtualKeyCode as Key;

type Vec3 = cgmath::Vector3<f32>;

pub trait Spectator : GameObject + HasPosition + HasOrientation {
    fn get_movement_speed(&self) -> f32;
}

impl<T: Spectator> GameObject for T {
    fn start(&mut self, input: &Input) -> () {
    }

    fn update(&mut self, input: &Input, delta_time: f32) -> () {
        let horizontal_axis : f32 = 
            (if input.is_key_down(Key::D) {1.0} else {0.0}) -
            (if input.is_key_down(Key::A) {1.0} else {0.0});

        let forward_axis: f32 = 
            (if input.is_key_down(Key::W) {1.0} else {0.0}) -
            (if input.is_key_down(Key::S) {1.0} else {0.0});

        let vertical_axis : f32 = 
            (if input.is_key_down(Key::E) {1.0} else {0.0}) -
            (if input.is_key_down(Key::Q) {1.0} else {0.0});

        let movement_direction : Vec3 = {
            let direction = 
                horizontal_axis * self.get_right() + 
                vertical_axis * self.get_up() +
                forward_axis * self.get_forward();
            if direction.is_zero() {
                direction.normalize();
            }
            direction
        };

        let move_amount : Vec3 = 
            movement_direction * delta_time * self.get_movement_speed();

        if !move_amount.is_zero() {
            self.set_position(self.get_position() + move_amount);
        }
    }

    fn end(&mut self, input: &Input) -> () {

    }
}
