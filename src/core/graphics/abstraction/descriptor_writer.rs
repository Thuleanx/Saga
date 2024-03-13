use crate::core::graphics::{ImageSampler, LoadedImage};
use vulkanalia::prelude::v1_0::*;

struct ImageWriteInformation {
    info: [vk::DescriptorImageInfoBuilder; 1],
    binding: u32,
    descriptor_set: vk::DescriptorSet,
}

struct UniformWriteInformation {
    info: [vk::DescriptorBufferInfoBuilder; 1],
    binding: u32,
    descriptor_set: vk::DescriptorSet,
}

#[derive(Default)]
pub struct DescriptorWriter {
    image_writes: Vec<ImageWriteInformation>,
    uniform_writes: Vec<UniformWriteInformation>,
}

impl DescriptorWriter {
    pub fn queue_write_image(
        &mut self,
        device: &Device,
        sampler: &ImageSampler,
        image: &LoadedImage,
        descriptor_sets: &[vk::DescriptorSet],
        binding: u32,
    ) {
        let info = vk::DescriptorImageInfo::builder()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(image.get_image_view())
            .sampler(sampler.get_sampler());

        descriptor_sets.iter().for_each(|descriptor_set| {
            self.image_writes.push(ImageWriteInformation {
                info: [info],
                binding,
                descriptor_set: descriptor_set.clone(),
            })
        })
    }

    pub fn queue_write_buffers<UniformBufferObject>(
        &mut self,
        device: &Device,
        uniform_buffer: vk::Buffer,
        descriptor_sets: &[vk::DescriptorSet],
        binding: u32,
    ) {
        let size_of_buffer_object = std::mem::size_of::<UniformBufferObject>() as u64;
        let info = vk::DescriptorBufferInfo::builder()
            .buffer(uniform_buffer)
            .offset(0)
            .range(size_of_buffer_object);

        descriptor_sets.iter().for_each(|descriptor_set| {
            self.uniform_writes.push(UniformWriteInformation {
                info: [info],
                binding,
                descriptor_set: descriptor_set.clone(),
            })
        })
    }

    pub fn write(&mut self, device: &Device) {
        let writes: Vec<vk::WriteDescriptorSetBuilder<'_>> = self
            .image_writes
            .iter()
            .map(|info| {
                vk::WriteDescriptorSet::builder()
                    .dst_set(info.descriptor_set)
                    .dst_binding(info.binding)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&info.info)
            })
            .chain(self.uniform_writes.iter().map(|info| {
                vk::WriteDescriptorSet::builder()
                    .dst_set(info.descriptor_set)
                    .dst_binding(info.binding)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&info.info)
            }))
            .collect();

        unsafe {
            device.update_descriptor_sets(&writes, &[] as &[vk::CopyDescriptorSet]);
        }

        self.clear();
    }

    pub fn clear(&mut self) {
        self.image_writes.clear();
        self.uniform_writes.clear();
    }
}
