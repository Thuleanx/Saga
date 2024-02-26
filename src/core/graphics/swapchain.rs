use anyhow::Result;
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::KhrSurfaceExtension;
use vulkanalia::vk::KhrSwapchainExtension;
use winit::window::Window;

use super::queue_families::QueueFamilyIndices;
use super::wrappers::create_image_view;

pub struct Swapchain {
    chain: vk::SwapchainKHR,
    format: vk::Format,
    extent: vk::Extent2D,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    destroyed: bool, // marked true when swapchain is destroyed.
}

impl Swapchain {
    pub unsafe fn new(window: &Window, instance: &Instance, device: &Device, 
              surface: vk::SurfaceKHR, physical_device: vk::PhysicalDevice
    ) -> Result<Self> {
        let (swapchain, swapchain_images, swapchain_format, swapchain_extent)
            = create_swapchain(window, &instance, &device, surface, physical_device)?;
        let swapchain_image_views = create_swapchain_image_views(
            device, &swapchain_images, swapchain_format)?;
        Ok(Self { 
            chain: swapchain, 
            format: swapchain_format, 
            extent: swapchain_extent, 
            images: swapchain_images, 
            image_views: swapchain_image_views,
            destroyed: false,
        })
    }

    pub fn get_chain(&self) -> vk::SwapchainKHR { self.chain }
    pub fn get_format(&self) -> vk::Format { self.format }
    pub fn get_extent(&self) -> vk::Extent2D { self.extent }
    pub fn get_image_views(&self) -> &[vk::ImageView] { &self.image_views }
    pub fn get_images(&self) -> &[vk::Image] { &self.images }
    pub fn get_length(&self) -> usize { self.images.len() }

    pub unsafe fn destroy(&mut self, device: &Device) {
        self.destroyed = true;
        destroy_swapchain_and_image_views(
            device, self.chain, &self.image_views);
    }

}

#[derive(Clone, Debug)]
struct SwapchainSupport {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupport {
    unsafe fn get(
        instance: &Instance,
        window_surface: vk::SurfaceKHR,
        physical_device: vk::PhysicalDevice,
    )-> Result<Self> {
        Ok(Self {
            capabilities: instance
                .get_physical_device_surface_capabilities_khr(physical_device, window_surface)?,
            formats: instance
                .get_physical_device_surface_formats_khr(physical_device, window_surface)?,
            present_modes: instance
                .get_physical_device_surface_present_modes_khr(physical_device, window_surface)?
        })
    }
}

fn get_swapchain_surface_format(
    formats: &[vk::SurfaceFormatKHR]
) -> vk::SurfaceFormatKHR {
    formats.iter()
        .cloned()
        .find(|f| {
            f.format == vk::Format::B8G8R8A8_SRGB
                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .unwrap_or_else(|| formats[0])
}

fn get_swapchain_present_mode(
    present_modes: &[vk::PresentModeKHR],
) -> vk::PresentModeKHR {
    present_modes
        .iter()
        .cloned()
        .find(|m| *m == vk::PresentModeKHR::MAILBOX)
        .unwrap_or(vk::PresentModeKHR::FIFO)
}

fn get_swapchain_extent(
    window: &Window,
    capabilities: vk::SurfaceCapabilitiesKHR,
) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::MAX {
        capabilities.current_extent
    } else {
        let size = window.inner_size();
        let clamp = |min: u32, max:u32, v:u32| min.max(max.min(v));
        vk::Extent2D::builder()
            .width(clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
                size.width
            ))
            .height(clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
                size.height
            )).build()
    }
}

pub unsafe fn create_swapchain(
    window: &Window,
    instance: &Instance,
    device: &Device,
    window_surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
) -> Result<(vk::SwapchainKHR, Vec<vk::Image>, vk::Format, vk::Extent2D)> {

    let indices = QueueFamilyIndices::get(instance, window_surface, physical_device)?;
    let support = SwapchainSupport::get(instance, window_surface, physical_device)?;

    let surface_format = get_swapchain_surface_format(&support.formats);
    let present_mode = get_swapchain_present_mode(&support.present_modes);
    let extent = get_swapchain_extent(window, support.capabilities);

    let image_count = (support.capabilities.min_image_count + 1).max(
        support.capabilities.max_image_count.max(support.capabilities.min_image_count));

    let mut queue_family_indices = vec![];
    let image_sharing_mode = if indices.graphics != indices.present {
        queue_family_indices.push(indices.graphics);
        queue_family_indices.push(indices.present);
        vk::SharingMode::CONCURRENT
    } else {
        vk::SharingMode::EXCLUSIVE
    };

    let info = vk::SwapchainCreateInfoKHR::builder()
        .surface(window_surface)
        .min_image_count(image_count)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(image_sharing_mode)
        .queue_family_indices(&queue_family_indices)
        .pre_transform(support.capabilities.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true)
        .old_swapchain(vk::SwapchainKHR::null());

    let swapchain = device.create_swapchain_khr(&info, None)?;
    let swapchain_images = device.get_swapchain_images_khr(swapchain)?;
    let swapchain_format : vk::Format = surface_format.format;
    let swapchain_extent = extent;

    Ok((swapchain, swapchain_images, swapchain_format, swapchain_extent))
}

pub unsafe fn create_swapchain_image_views(
    device: &Device,
    swapchain_images: &Vec<vk::Image>,
    swapchain_format: vk::Format,
) -> Result<Vec<vk::ImageView>> {
    let swapchain_image_views = swapchain_images
        .iter()
        .map(|i| create_image_view(device, *i, swapchain_format) )
        .collect::<Result<Vec<_>, _>>()?;
    Ok(swapchain_image_views)
}

pub unsafe fn destroy_swapchain_and_image_views(device: &Device, 
        swapchain: vk::SwapchainKHR, swapchain_image_views: &[vk::ImageView]) {
    swapchain_image_views
        .iter()
        .for_each(|v| device.destroy_image_view(*v, None));
    device.destroy_swapchain_khr(swapchain, None);
}

pub unsafe fn create_texture_sampler(device: &Device) -> Result<()> {
    Ok(())
}

