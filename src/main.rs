#![allow(
    dead_code,
    unused_variables,
    clippy::too_many_arguments,
    clippy::unnecessary_wraps,
    unused_unsafe,
)]
// #![deny(unsafe_op_in_unsafe_fn)]

mod saga;
mod core;
mod datastructures;
mod gameworld;
mod patterns;
mod ecs;

use core::graphics::{Graphics, StartRenderResult};
use std::time::Instant;

use anyhow::Result;
use saga::PerspectiveCameraBuilder;
use saga::input::Input;
use saga::mesh::MeshBuilder;
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{EventLoop, ControlFlow};
use winit::window::WindowBuilder;
use vulkanalia::prelude::v1_0::*;

use bevy_app::prelude::*;

type Mat4 = cgmath::Matrix4<f32>;

fn main() -> Result<()> {
    pretty_env_logger::init();

    let app = App::new().run();

    // Window
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Saga Engine")
        .with_inner_size(LogicalSize::new(1024, 768))
        .build(&event_loop)?;

    // App
    let mut graphics = Graphics::create(&window)?;
    let mut destroying : bool = false;
    let mut minimized : bool = false;

    let start_time = Instant::now();
    let mut input_manager = Input::default();

    let size = window.inner_size();
    let mut camera = {
        let mut camera_builder = PerspectiveCameraBuilder::default();
        camera_builder
            .set_width(size.width)
            .set_height(size.height);
        camera_builder.build(&graphics)
    }?;

    let path_to_obj = std::env::current_dir().unwrap().join("assets").join("meshes").join("Imphat.obj");
    let path_to_texture = std::env::current_dir().unwrap().join("assets").join("meshes").join("imphat_diffuse.png");

    let mesh_builder = MeshBuilder::new(&path_to_obj, &path_to_texture);

    let mesh = mesh_builder.build(&graphics, &camera)?;

    unsafe { 
        graphics.record_command_buffers(|graphics: &Graphics, command_buffer: vk::CommandBuffer, index: usize| {
            unsafe { 
                mesh.bind(&graphics, command_buffer); 
                camera.bind(&graphics, command_buffer, index);
                mesh.draw(&graphics, command_buffer);
            }
        })?; 
    }

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
                input_manager.handle_key_event(key_ev);
            },
            Event::WindowEvent {
                event: WindowEvent::MouseInput { state, button, .. },
                ..
            } => {
                // input_manager.handle_mouse_button(state, button);
            }
            Event::WindowEvent {
                event: WindowEvent::CursorMoved { position, .. },
                ..
            } => {
                // input.handle_mouse_move(position);
            },
            _ => ()
        }

        let mut should_recreate_swapchain = false;

        (|| { match event {
            // Render a frame if our Vulkan app is not being destroyed.
            Event::MainEventsCleared if !destroying => {
                if minimized {
                    return;
                }

                let image_index = unsafe { 
                    match graphics.start_render(&window) {
                        StartRenderResult::Normal(Ok(image_index)) => image_index,
                        StartRenderResult::Normal(Err(e)) => panic!("{}", e),
                        StartRenderResult::ShouldRecreateSwapchain => {
                            should_recreate_swapchain = true;
                            return;
                        },
                    }
                };

                unsafe { camera.update_uniform_buffer_series(&graphics, image_index).unwrap(); }

                unsafe { 
                    match graphics.end_render(&window, image_index) {
                        Ok(true) => {},
                        Err(e) => panic!("{}", e),
                        Ok(false) => {
                            should_recreate_swapchain = true;
                        }
                    }
                }
            },
            Event::WindowEvent { event: WindowEvent::Resized(size), .. } => {
                if size.width == 0 || size.height == 0 {
                    minimized = true;
                } else {
                    minimized = false;
                    graphics.trigger_resize();
                }
            },
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                // Destroy our Vulkan app.
                destroying = true;
                *control_flow = ControlFlow::Exit;
                unsafe { 
                    graphics.device_wait_idle()
                        .expect("Failed to wait for device idle for destroying graphics"); 
                }

                unsafe { 
                    graphics.free_command_buffers();
                    mesh.unload(&graphics).unwrap(); 
                    camera.unload(&graphics).unwrap(); 
                    graphics.destroy();
                }
            },
            _ => {}
        }
        })();
        
        if should_recreate_swapchain {
            unsafe { 
                graphics.device_wait_idle().unwrap();
                graphics.free_command_buffers();
                camera.before_swapchain_recreate(&graphics).unwrap(); 
                graphics.recreate_swapchain(&window).unwrap();
                camera.after_swapchain_recreate(&graphics).unwrap(); 
                mesh.after_swapchain_recreate(&graphics, &camera).unwrap();
                graphics.record_command_buffers(|graphics: &Graphics, command_buffer: vk::CommandBuffer, index: usize| {
                    unsafe { 
                        mesh.bind(&graphics, command_buffer); 
                        camera.bind(&graphics, command_buffer, index);
                        mesh.draw(&graphics, command_buffer);
                    }
                }).unwrap();
                graphics.continue_after_swapchain_construction();
            }
        }

    })
}

