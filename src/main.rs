#![allow(
    dead_code,
    unused_variables,
    clippy::too_many_arguments,
    clippy::unnecessary_wraps
    )]

mod vulkan;

use crate::vulkan::App;

use anyhow::Result;
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
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            // Render a frame if our Vulkan app is not being destroyed.
            Event::MainEventsCleared if !destroying && !minimized =>
                unsafe { app.render(&window) }.unwrap(),
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
            }
            _ => {}
        }
    });
}

