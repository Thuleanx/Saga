pub mod layout {
    use anyhow::Result;
    use vulkanalia::prelude::v1_0::*;

    pub unsafe fn create_descriptor_set_layout(
        device: &Device,
        binding: u32,
        descriptor_type: vk::DescriptorType,
        descriptor_count: u32,
        stage_flags: vk::ShaderStageFlags,

    ) -> Result<vk::DescriptorSetLayout> {

        let ubo_binding = vk::DescriptorSetLayoutBinding::builder()
            .binding(binding)
            .descriptor_type(descriptor_type)
            .descriptor_count(descriptor_count)
            .stage_flags(stage_flags);

        let ubo_bindings = &[ubo_binding];
        let info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(ubo_bindings);

        let descriptor_set_layout = device.create_descriptor_set_layout(&info, None)?;

        Ok(descriptor_set_layout)
    }

    pub unsafe fn destroy_descriptor_set_layout(
        device: &Device, 
        descriptor_set_layout: vk::DescriptorSetLayout
    ) -> () {
        device.destroy_descriptor_set_layout(descriptor_set_layout, None);
    }
}

pub mod pool {
    use anyhow::Result;
    use vulkanalia::prelude::v1_0::*;

    pub unsafe fn create_descriptor_pool(
        device: &Device,
        type_: vk::DescriptorType,
        descriptor_count: u32,
    ) -> Result<vk::DescriptorPool> {

        let ubo_size: vk::DescriptorPoolSizeBuilder 
            = vk::DescriptorPoolSize::builder()
            .type_(type_)
            .descriptor_count(descriptor_count);

        let pool_size: &[vk::DescriptorPoolSizeBuilder; 1] 
            = &[ubo_size];
        let pool_create_info: vk::DescriptorPoolCreateInfoBuilder<'_> 
            = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(pool_size)
            .max_sets(descriptor_count);

        let descriptor_pool = device.create_descriptor_pool(&pool_create_info, None)?;

        Ok(descriptor_pool)
    }

    pub unsafe fn destroy_descriptor_pool(device: &Device, descriptor_pool: vk::DescriptorPool) {
        device.destroy_descriptor_pool(descriptor_pool, None);
    }
}

pub mod set {
    use anyhow::{Result, anyhow};
    use vulkanalia::prelude::v1_0::*;

    pub unsafe fn create_descriptor_sets(
        device: &Device, 
        size_of_buffer_object: u64,
        descriptor_type: vk::DescriptorType,
        descriptor_set_layout: vk::DescriptorSetLayout, 
        descriptor_pool: vk::DescriptorPool, 
        descriptor_set: &[vk::DescriptorSet],
        buffers: &[vk::Buffer],
    ) -> Result<Vec<vk::DescriptorSet>> {

        let buffer_size : usize = buffers.len();
        let descriptor_set_size : usize = descriptor_set.len();
        if buffer_size != descriptor_set_size {
            return Err(anyhow!(
                "Given buffer size {0} does not fit with descriptor set size {1}", 
                buffer_size, descriptor_set_size)
            );
        }

        let number_of_buffers : usize = descriptor_set.len();

        let layouts: Vec<vk::DescriptorSetLayout> = 
            vec![descriptor_set_layout; number_of_buffers];
        let info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
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
                .descriptor_type(descriptor_type)
                .buffer_info(buffer_info);

            device.update_descriptor_sets(&[ubo_write], &[] as &[vk::CopyDescriptorSet]);
        }


        Ok(descriptor_sets)
    }
}
