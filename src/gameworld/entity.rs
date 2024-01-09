use std::{fmt::Display, time::Instant};

use anyhow::{Result, anyhow};
use crate::datastructures::TypeMap;

use super::component::Component;

pub struct Entity {
    name: String,
    components: TypeMap,
}

impl Default for Entity {
    fn default() -> Self {
        Self { name: "default entity".to_string(), components: TypeMap::default() }
    }
}

impl Display for Entity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

/// Modification operations
impl Entity {
    pub fn add_component<T>(&mut self, component: T) -> Result<()> where T: Component + 'static {
        if self.components.contains::<T>() {
            return Err(anyhow!("Entity {0} already contains component of type {1}", self, std::any::type_name::<T>()));
        }
        self.components.put::<T>(component);
        Ok(())
    }
}


/// Logic operations
impl Entity {
    pub fn start(&mut self) -> () {
        self.components
            .values_mut()
            .for_each(|any_component| {
                if let Some(component) = any_component.downcast_mut::<Box<dyn Component>>() {
                    component.start();
                }
            })
    }

    pub fn update(&mut self, instant: Instant) -> () {
        self.components
            .values_mut()
            .for_each(|any_component| {
                if let Some(component) = any_component.downcast_mut::<Box<dyn Component>>() {
                    component.update(instant);
                }
            })
    }

    pub fn end(&mut self) -> () {
        self.components
            .values_mut()
            .for_each(|any_component| {
                if let Some(component) = any_component.downcast_mut::<Box<dyn Component>>() {
                    component.end();
                }
            })
    }
}
