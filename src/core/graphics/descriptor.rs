pub use pool::Pool;

pub mod layout {
    use anyhow::Result;
    use vulkanalia::prelude::v1_0::*;

    pub struct DescriptorInfo {
        pub binding: u32,
        pub descriptor_type: vk::DescriptorType,
        pub descriptor_count: u32,
        pub stage_flags: vk::ShaderStageFlags,
    }

    pub unsafe fn create(
        device: &Device,
        descriptors: &[DescriptorInfo],
    ) -> Result<vk::DescriptorSetLayout> {
        let ubo_bindings: Vec<vk::DescriptorSetLayoutBindingBuilder<'_>> = descriptors
            .iter()
            .map(|spec| {
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(spec.binding)
                    .descriptor_type(spec.descriptor_type)
                    .descriptor_count(spec.descriptor_count)
                    .stage_flags(spec.stage_flags)
            })
            .collect();

        let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&ubo_bindings);

        let descriptor_set_layout = device.create_descriptor_set_layout(&info, None)?;


        Ok(descriptor_set_layout)
    }

    pub unsafe fn destroy(device: &Device, descriptor_set_layout: vk::DescriptorSetLayout) -> () {
        device.destroy_descriptor_set_layout(descriptor_set_layout, None);
    }
}

pub mod pool {
    use anyhow::Result;
    use vulkanalia::prelude::v1_0::*;

    #[derive(Clone, Copy, Debug, Default)]
    pub struct Pool {
        pool: vk::DescriptorPool,
        size: usize,
        currently_allocated: usize,
    }

    impl Pool {
        pub fn get_size(&self) -> usize {
            self.size
        }
        pub fn get_num_descriptors_left(&self) -> usize {
            self.size - self.currently_allocated
        }

        pub unsafe fn create(
            &mut self,
            device: &Device,
            descriptor_set_layout: vk::DescriptorSetLayout,
            number_of_set_layouts: usize,
        ) -> Result<Vec<vk::DescriptorSet>> {
            if number_of_set_layouts + self.currently_allocated > self.size {
                return Err(anyhow::anyhow!("Cannot create more descriptors than that of the pool"));
            }

            let layouts: Vec<vk::DescriptorSetLayout> =
                vec![descriptor_set_layout; number_of_set_layouts];
            let info = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(self.pool)
                .set_layouts(&layouts);

            let descriptor_sets: Vec<vk::DescriptorSet> = device.allocate_descriptor_sets(&info)?;
            self.currently_allocated += number_of_set_layouts;

            Ok(descriptor_sets)
        }

        pub unsafe fn free(&mut self, device: &Device) -> Result<()> {
            device.reset_descriptor_pool(self.pool, vk::DescriptorPoolResetFlags::empty())?;
            self.currently_allocated = 0;
            Ok(())
        }
    }

    #[derive(Clone)]
    pub struct PoolDescription {
        pub type_: vk::DescriptorType,
        pub descriptor_count: u32,
    }

    pub unsafe fn create(
        device: &Device,
        pool_descriptions: &[PoolDescription],
        max_descriptor_sets: u32,
    ) -> Result<Pool> {
        let pool_sizes: Vec<vk::DescriptorPoolSizeBuilder> = pool_descriptions
            .iter()
            .map(|description| {
                vk::DescriptorPoolSize::builder()
                    .type_(description.type_)
                    .descriptor_count(description.descriptor_count)
            })
            .collect();

        let pool_create_info: vk::DescriptorPoolCreateInfoBuilder<'_> =
            vk::DescriptorPoolCreateInfo::builder()
                .pool_sizes(&pool_sizes)
                .max_sets(max_descriptor_sets)
                .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);

        let descriptor_pool = device.create_descriptor_pool(&pool_create_info, None)?;

        Ok(Pool {
            pool: descriptor_pool,
            size: max_descriptor_sets as usize,
            currently_allocated: 0,
        })
    }

    pub unsafe fn destroy(device: &Device, pool: &Pool) {
        device.destroy_descriptor_pool(pool.pool, None);
    }
}

pub mod set {
}
