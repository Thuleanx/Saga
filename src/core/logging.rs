pub use log::*;

pub fn init() -> () {
    pretty_env_logger::init();
}
