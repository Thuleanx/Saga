use anyhow::Result;
use super::uniform_buffer_object::UniformBufferObject;
use vulkanalia::prelude::v1_0::*;
use std::mem::size_of;

pub unsafe fn create_descriptor_pool(device: &Device, swapchain_length : u32) -> Result<vk::DescriptorPool> {
    let ubo_size: vk::DescriptorPoolSizeBuilder 
        = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(swapchain_length);

    let pool_size: &[vk::DescriptorPoolSizeBuilder; 1] 
        = &[ubo_size];
    let pool_create_info: vk::DescriptorPoolCreateInfoBuilder<'_> 
        = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(pool_size)
        .max_sets(swapchain_length);

    let descriptor_pool = device.create_descriptor_pool(&pool_create_info, None)?;

    Ok(descriptor_pool)
}

pub unsafe fn create_descriptor_sets(
    device: &Device, 
    descriptor_set_layout: vk::DescriptorSetLayout, 
    descriptor_pool: vk::DescriptorPool, 
    swapchain_length: usize,
    uniform_buffers: &Vec<vk::Buffer>,
    descriptor_set: &Vec<vk::DescriptorSet>,
) -> Result<Vec<vk::DescriptorSet>> {

    let layouts: Vec<vk::DescriptorSetLayout> = vec![descriptor_set_layout; swapchain_length];
    let info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&layouts);

    let descriptor_sets: Vec<vk::DescriptorSet> = device.allocate_descriptor_sets(&info)?;

    for i in 0..swapchain_length {
        let info = vk::DescriptorBufferInfo::builder()
            .buffer(uniform_buffers[i])
            .offset(0)
            .range(size_of::<UniformBufferObject>() as u64);


        let buffer_info = &[info];
        let ubo_write = vk::WriteDescriptorSet::builder()
            .dst_set(descriptor_sets[i])
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(buffer_info);

        device.update_descriptor_sets(&[ubo_write], &[] as &[vk::CopyDescriptorSet]);
    }


    Ok(descriptor_sets)
}

pub unsafe fn destroy_descriptor_pool(device: &Device, descriptor_pool: vk::DescriptorPool) {
    device.destroy_descriptor_pool(descriptor_pool, None);
}
