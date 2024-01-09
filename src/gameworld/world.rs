use anyhow::{Result, anyhow};
use thiserror::Error;
use super::entity::Entity;

#[derive(Debug, Error)]
#[error("Reason: {0}")]
pub struct WorldModificationError(pub &'static str);

#[derive(Default)]
pub struct World {
    entities: Vec<Entity>,
}

impl World {
    fn add_entity(&mut self, entity: Option<Entity>) -> Result<&mut Entity> {
        if let Some(wrapped_entity) = entity {
            self.entities.push(wrapped_entity);
        } else {
            self.entities.push(Entity::default());
        }

        if let Some(entity) = self.entities.last_mut() {
            Ok(entity)
        } else {
            Err(anyhow!(WorldModificationError("Cannot add an aditional entity")))
        }
    }

    fn start(&mut self) -> Result<()> {
        for entity in self.entities.iter_mut() {
            entity.start();
        }

        Ok(())
    }
}
