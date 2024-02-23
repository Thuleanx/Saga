pub use pool::Pool;

pub mod layout {
    use anyhow::Result;
    use vulkanalia::prelude::v1_0::*;

    pub struct UBODescription {
        pub binding: u32,
        pub descriptor_type: vk::DescriptorType,
        pub descriptor_count: u32,
        pub stage_flags: vk::ShaderStageFlags,
    }

    pub unsafe fn create(
        device: &Device,
        ubo_specs: &[UBODescription],
    ) -> Result<vk::DescriptorSetLayout> {
        let ubo_bindings : Vec<vk::DescriptorSetLayoutBindingBuilder<'_>>
            = ubo_specs
            .iter()
            .map(|spec| {
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(spec.binding)
                    .descriptor_type(spec.descriptor_type)
                    .descriptor_count(spec.descriptor_count)
                    .stage_flags(spec.stage_flags)
            }).collect();

        let info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(&ubo_bindings);

        let descriptor_set_layout = device.create_descriptor_set_layout(&info, None)?;

        Ok(descriptor_set_layout)
    }

    pub unsafe fn destroy(
        device: &Device, 
        descriptor_set_layout: vk::DescriptorSetLayout
    ) -> () {
        device.destroy_descriptor_set_layout(descriptor_set_layout, None);
    }
}

pub mod pool {
    use anyhow::Result;
    use vulkanalia::prelude::v1_0::*;

    #[derive(Clone, Copy, Debug, Default)]
    pub struct Pool {
        pool: vk::DescriptorPool,
        type_: vk::DescriptorType,
        size: u32,
    }

    impl Pool {
        pub fn get_type(&self) -> vk::DescriptorType { self.type_ }
        pub fn get_pool(&self) -> vk::DescriptorPool { self.pool }
    }

    pub unsafe fn create(
        device: &Device,
        type_: vk::DescriptorType,
        descriptor_count: u32,
    ) -> Result<Pool> {

        let ubo_size: vk::DescriptorPoolSizeBuilder 
            = vk::DescriptorPoolSize::builder()
            .type_(type_)
            .descriptor_count(descriptor_count);

        let pool_size: &[vk::DescriptorPoolSizeBuilder; 1] 
            = &[ubo_size];
        let pool_create_info: vk::DescriptorPoolCreateInfoBuilder<'_> 
            = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(pool_size)
            .max_sets(descriptor_count)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);

        let descriptor_pool = device.create_descriptor_pool(&pool_create_info, None)?;

        Ok(Pool {
            pool: descriptor_pool,
            type_,
            size: descriptor_count,
        })
    }

    pub unsafe fn destroy(device: &Device, pool: &Pool) {
        device.destroy_descriptor_pool(pool.pool, None);
    }

}

pub mod set {
    use std::mem::size_of;

    use anyhow::Result;
    use vulkanalia::prelude::v1_0::*;
    use super::pool::Pool;

    pub unsafe fn create<T>(
        device: &Device, 
        descriptor_pool: &Pool,
        descriptor_set_layout: vk::DescriptorSetLayout, 
        buffers: &[vk::Buffer],
    ) -> Result<Vec<vk::DescriptorSet>> {
        _create(
            device, 
            size_of::<T>() as u64, 
            descriptor_pool, 
            descriptor_set_layout, 
            buffers
        )
    }

    unsafe fn _create(
        device: &Device, 
        size_of_buffer_object: u64,
        descriptor_pool: &Pool,
        descriptor_set_layout: vk::DescriptorSetLayout, 
        buffers: &[vk::Buffer],
    ) -> Result<Vec<vk::DescriptorSet>> {

        let number_of_buffers : usize = buffers.len();

        let layouts: Vec<vk::DescriptorSetLayout> = 
            vec![descriptor_set_layout; number_of_buffers];
        let info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool.get_pool())
            .set_layouts(&layouts);

        let descriptor_sets: Vec<vk::DescriptorSet> = device.allocate_descriptor_sets(&info)?;

        for i in 0..number_of_buffers {
            let info = vk::DescriptorBufferInfo::builder()
                .buffer(buffers[i])
                .offset(0)
                .range(size_of_buffer_object);

            let buffer_info = &[info];
            let ubo_write = vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_sets[i])
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(descriptor_pool.get_type())
                .buffer_info(buffer_info);

            device.update_descriptor_sets(&[ubo_write], &[] as &[vk::CopyDescriptorSet]);
        }


        Ok(descriptor_sets)
    }

    pub unsafe fn free(device: &Device, pool: Pool, descriptor_sets: &[vk::DescriptorSet]) -> Result<()> {
        Ok(device.free_descriptor_sets(pool.get_pool(), descriptor_sets)?)
    }
}
