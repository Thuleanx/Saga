use crate::core::graphics::descriptor::{self};
use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

pub struct DescriptorAllocator {
    pools: Vec<descriptor::Pool>,
    template: Vec<descriptor::pool::PoolDescription>,
    pool_index: usize,
    minimum_size: u32,
    maximum_size: u32,
}

impl DescriptorAllocator {
    pub fn new(
        device: &Device,
        pool_descriptions: &[descriptor::pool::PoolDescription],
        minimum_size: u32,
        maximum_size: u32,
    ) -> Self {
        Self {
            pools: vec![],
            template: pool_descriptions.iter().cloned().collect(),
            pool_index: 0,
            minimum_size,
            maximum_size,
        }
    }

    pub unsafe fn allocate(
        &mut self,
        device: &Device,
        descriptor_set_layout: vk::DescriptorSetLayout,
        number_of_set_layouts: usize,
    ) -> Result<Vec<vk::DescriptorSet>> {
        // We perform 2 allocations. If the first one fails
        // then we have ran out of memory on our current pool
        // and must move on to the next one, hence the second allocation

        if let Ok(descriptor_sets) =
            self.allocate_once(device, descriptor_set_layout, number_of_set_layouts)
        {
            Ok(descriptor_sets)
        } else {
            self.pool_index += 1;
            if self.pool_index >= self.pools.len() {
                let pool = self.create_next_pool(device)?;
                self.pools.push(pool);
                self.pool_index = self.pools.len() - 1;
            }

            self.allocate_once(device, descriptor_set_layout, number_of_set_layouts)
        }
    }

    unsafe fn allocate_once(
        &mut self,
        device: &Device,
        descriptor_set_layout: vk::DescriptorSetLayout,
        number_of_set_layouts: usize,
    ) -> Result<Vec<vk::DescriptorSet>> {
        if let Some(pool) = self.pools.get_mut(self.pool_index) {
            pool.create(device, descriptor_set_layout, number_of_set_layouts)
        } else {
            Err(anyhow::anyhow!("Pool is empty, cannot allocate"))
        }
    }

    unsafe fn create_next_pool(&self, device: &Device) -> Result<descriptor::Pool> {
        let next_pool_size: u32 = match self.pools.last() {
            None => self.minimum_size,
            Some(pool) => std::cmp::min(pool.get_size() as u32 * 2, self.maximum_size),
        };

        let descriptions: Vec<descriptor::pool::PoolDescription> = self
            .template
            .iter()
            .map(|pool_description| descriptor::pool::PoolDescription {
                type_: pool_description.type_,
                descriptor_count: pool_description.descriptor_count * next_pool_size,
            })
            .collect();

        unsafe { descriptor::pool::create(device, &descriptions, next_pool_size) }
    }

    pub unsafe fn free(&mut self, device: &Device) -> Result<()> {
        let number_of_pools_to_free = self.pool_index
            + (if self.pool_index < self.pools.len() {
                1
            } else {
                0
            });

        for i in 0..number_of_pools_to_free {
            unsafe {
                self.pools[i].free(device)?;
            }
        }
        self.pool_index = 0;

        Ok(())
    }

    pub fn destroy(&mut self, device: &Device) {
        for pool in self.pools.iter() {
            unsafe {
                descriptor::pool::destroy(device, pool);
            }
        }
        self.pools.clear();
        self.pool_index = 0;
    }
}
