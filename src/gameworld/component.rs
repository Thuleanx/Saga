use std::time::Instant;

pub trait Component {
    fn start(&mut self) -> ();
    fn update(&mut self, delta_time: Instant) -> ();
    fn end(&mut self) -> ();
}

impl dyn Component {
    fn start(&mut self) -> () {}
    fn update(&mut self, delta_time: Instant) -> () {}
    fn end(&mut self) -> () {}
}
