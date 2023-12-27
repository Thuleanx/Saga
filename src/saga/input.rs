use std::collections::HashSet;
use winit::event::KeyboardInput;

pub use winit::event::VirtualKeyCode as Key;

type Vec2 = cgmath::Vector2<f32>;

#[derive(Clone, Debug)]
pub struct Input {
    key_pressed: HashSet<usize>,
    button_pressed: HashSet<usize>,
    mouse_position: Vec2,
}

impl Default for Input {
    fn default() -> Self {
        Self {
            key_pressed : HashSet::default(),
            button_pressed : HashSet::default(),
            mouse_position: Vec2::new(0.0, 0.0),
        }
    }
}

impl Input {
    pub fn handle_key_event(&mut self, key_input: KeyboardInput) -> () {
        if let KeyboardInput {
            virtual_keycode: Some(keycode),
            state,
            ..
        } = key_input
        {
            match state {
                winit::event::ElementState::Pressed => {
                    self.key_pressed.insert(keycode as usize);
                }
                winit::event::ElementState::Released => {
                    self.key_pressed.remove(&(keycode as usize));
                }
            }
        }
    }

    pub fn is_key_down(&self, keycode: Key) -> bool {
        self.key_pressed.contains(&(keycode as usize))
    }
}
