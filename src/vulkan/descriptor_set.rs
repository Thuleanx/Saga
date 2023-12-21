use anyhow::Result;
use super::{
    appdata::AppData, 
    App, uniform_buffer_object::UniformBufferObject
};
use vulkanalia::prelude::v1_0::*;
use std::mem::size_of;

pub unsafe fn create_descriptor_pool(device: &Device, data: &mut AppData) -> Result<()> {
    let ubo_size: vk::DescriptorPoolSizeBuilder 
        = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(data.swapchain_images.len() as u32);

    let pool_size: &[vk::DescriptorPoolSizeBuilder; 1] 
        = &[ubo_size];
    let pool_create_info: vk::DescriptorPoolCreateInfoBuilder<'_> 
        = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(pool_size)
        .max_sets(data.swapchain_images.len() as u32);

    data.descriptor_pool = device.create_descriptor_pool(&pool_create_info, None)?;

    Ok(())
}

pub unsafe fn create_descriptor_sets(device: &Device, data: &mut AppData) -> Result<()> {
    let layouts: Vec<vk::DescriptorSetLayout> = vec![data.descriptor_set_layout; data.swapchain_images.len()];
    let info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(data.descriptor_pool)
        .set_layouts(&layouts);

    data.descriptor_sets = device.allocate_descriptor_sets(&info)?;

    for i in 0..data.swapchain_images.len() {
        let info = vk::DescriptorBufferInfo::builder()
            .buffer(data.uniform_buffers[i])
            .offset(0)
            .range(size_of::<UniformBufferObject>() as u64);


        let buffer_info = &[info];
        let ubo_write = vk::WriteDescriptorSet::builder()
            .dst_set(data.descriptor_sets[i])
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(buffer_info);

        device.update_descriptor_sets(&[ubo_write], &[] as &[vk::CopyDescriptorSet]);
    }


    Ok(())
}

pub unsafe fn destroy_descriptor_pool(app: &App) {
    app.device.destroy_descriptor_pool(app.data.descriptor_pool, None);
}
