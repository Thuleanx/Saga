use std::{any::TypeId, collections::HashMap};

use type_map::TypeMap;

struct Entity(u64);

struct World {
    max_registered_entity_uid: u32,
}

struct Archetype {
    entities: Vec<Entity>,
    component_containers: TypeMap, // typeid -> ComponentContainer<T>
}

struct ComponentContainer<T> {
    container: Vec<T>,
}

struct Health(u64);
struct Position(cgmath::Vector3<f32>);
