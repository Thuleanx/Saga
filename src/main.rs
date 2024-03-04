#![allow(
    dead_code,
    unused_variables,
    clippy::too_many_arguments,
    clippy::unnecessary_wraps,
    unused_unsafe
)]
// #![deny(unsafe_op_in_unsafe_fn)]

mod core;
mod datastructures;
mod doomclone;
mod gameworld;
mod patterns;
mod saga;

use anyhow::Result;

type Mat4 = cgmath::Matrix4<f32>;

fn main() -> Result<()> {
    pretty_env_logger::init();
    doomclone::run_app();
    Ok(())
}
