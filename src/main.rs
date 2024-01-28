#![allow(
    dead_code,
    unused_variables,
    clippy::too_many_arguments,
    clippy::unnecessary_wraps
    )]

mod saga;
mod core;
mod datastructures;
mod gameworld;
mod ecs;

use crate::core::graphics::App;

use anyhow::Result;
use saga::common_traits::GameObject;
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

// Instance
use vulkanalia::prelude::v1_0::*;

fn main() -> Result<()> {
    pretty_env_logger::init();

    // Window
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Saga Engine")
        .with_inner_size(LogicalSize::new(1024, 768))
        .build(&event_loop)?;

    // App
    let mut app = unsafe { App::create(&window)? };
    let mut destroying : bool = false;
    let mut minimized : bool = false;

    let mut last_frame_time = app.start.elapsed().as_secs_f32();

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        // input handling
        if destroying {
            return;
        }
        match event {
            Event::WindowEvent {
                // Note this deeply nested pattern match
                event: WindowEvent::KeyboardInput {
                    input:key_ev,
                    ..
                },
                ..
            } => {
                app.data.input.handle_key_event(key_ev);
            },
            Event::WindowEvent {
                event: WindowEvent::MouseInput { state, button, .. },
                ..
            } => {
                // input.handle_mouse_button(state, button);
            }
            Event::WindowEvent {
                event: WindowEvent::CursorMoved { position, .. },
                ..
            } => {
                // input.handle_mouse_move(position);
            },
            _ => ()
        }
        match event {
            // Render a frame if our Vulkan app is not being destroyed.
            Event::MainEventsCleared if !destroying && !minimized => {
                let current_time = app.start.elapsed().as_secs_f32();
                (app.data.camera).update(&app.data.input, current_time - last_frame_time);
                last_frame_time = current_time;
                unsafe { app.render(&window)}.unwrap()
            },
            Event::WindowEvent { event: WindowEvent::Resized(size), .. } => {
                if size.width == 0 || size.height == 0 {
                    minimized = true;
                } else {
                    minimized = false;
                    app.resized = true;
                }
            },
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                // Destroy our Vulkan app.
                destroying = true;
                *control_flow = ControlFlow::Exit;
                unsafe { app.device.device_wait_idle().unwrap(); }
                unsafe { app.destroy(); }
            },
            _ => {}
        }
    });
}

