use anyhow::Result;
use vulkanalia::{
    vk::{self, DeviceV1_0, HasBuilder},
    Device,
};

use super::LoadedImage;

pub struct ImageSampler {
    sampler: vk::Sampler,
}

impl ImageSampler {
    pub unsafe fn create(device: &Device) -> Result<Self> {
        let sampler = unsafe { create_image_sampler(device)? };

        Ok(Self { sampler })
    }

    pub unsafe fn destroy(&self, device: &Device) {
        device.destroy_sampler(self.sampler, None);
    }
}

pub unsafe fn create_image_sampler(device: &Device) -> Result<vk::Sampler> {
    let info = vk::SamplerCreateInfo::builder()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::REPEAT)
        .address_mode_w(vk::SamplerAddressMode::REPEAT)
        .address_mode_v(vk::SamplerAddressMode::REPEAT)
        .anisotropy_enable(true)
        .max_anisotropy(16.0)
        .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
        .unnormalized_coordinates(false)
        .compare_enable(false)
        .compare_op(vk::CompareOp::ALWAYS)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .mip_lod_bias(0.0)
        .min_lod(0.0)
        .max_lod(0.0);

    let texture_sampler = device.create_sampler(&info, None)?;

    Ok(texture_sampler)
}

pub unsafe fn bind_sampler_to_descriptor_sets(device: &Device, sampler: &ImageSampler, image: &LoadedImage, descriptor_sets: &[vk::DescriptorSet], binding: u32) {
    descriptor_sets
        .iter()
        .for_each(|descriptor_set| {
            let info = vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(image.get_image_view())
                .sampler(sampler.sampler);

            let image_info = &[info];
            let sampler_write = vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set.clone())
                .dst_binding(binding)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(image_info);

            device.update_descriptor_sets(
                &[sampler_write],
                &[] as &[vk::CopyDescriptorSet],
            );
        });
        
}
