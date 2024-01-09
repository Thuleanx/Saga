use std::{
    any::{Any, TypeId},
    collections::{HashMap, hash_map::ValuesMut},
};

#[derive(Default)]
pub struct TypeMap {
    bindings: HashMap<TypeId, Box<dyn Any>>,
}

impl TypeMap {
    pub fn put<T: Any>(&mut self, value: T) {
        self.bindings.insert(value.type_id(), Box::new(value));
    }

    pub fn get<T: Any>(&self) -> Option<&T> {
        self.bindings
            .get(&TypeId::of::<T>())
            .and_then(|boxed| boxed.downcast_ref())
    }

    pub fn get_mut<T: Any>(&mut self) -> Option<&mut T> {
        self.bindings
            .get_mut(&TypeId::of::<T>())
            .and_then(|boxed| boxed.downcast_mut())
    }

    pub fn contains<T: Any>(&self) -> bool {
        self.bindings.contains_key(&TypeId::of::<T>())
    }

    pub fn values_mut(&mut self) -> ValuesMut<'_, TypeId, Box<dyn Any>> {
        self.bindings.values_mut()
    }
}
