use anyhow::{anyhow, Result};
use std::{fs::File, path::Path};
use vulkanalia::{
    vk::{self, DeviceV1_0, HasBuilder},
    Device, Instance,
};

struct ImageSampler {
    sampler: vk::Sampler,
}

impl ImageSampler {
    unsafe fn create(device: &Device) -> Result<Self> {
        let sampler = unsafe { create_image_sampler(device)? };

        Ok(Self { sampler })
    }

    unsafe fn destroy(&self, device: &Device) {
        device.destroy_sampler(self.sampler, None);
    }
}

unsafe fn create_image_sampler(device: &Device) -> Result<vk::Sampler> {
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

    let texture_sampler = device.create_sampler(&info, None);

    Ok(texture_sampler)
}
