use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use super::wrappers::DepthBuffer;

pub unsafe fn create_framebuffers(
    device: &Device,
    swapchain_image_views: &[vk::ImageView],
    render_pass: vk::RenderPass,
    depth_buffer: &DepthBuffer,
    swapchain_extent: vk::Extent2D,
) -> Result<Vec<vk::Framebuffer>> {
    let framebuffers = swapchain_image_views
        .iter()
        .map(|i| {
            let attachments = &[*i, depth_buffer.get_image_view()];
            let create_info = vk::FramebufferCreateInfo::builder()
                .render_pass(render_pass)
                .attachments(attachments)
                .width(swapchain_extent.width)
                .height(swapchain_extent.height)
                .layers(1);

            device.create_framebuffer(&create_info, None)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(framebuffers)
}

pub unsafe fn destroy_framebuffers(device: &Device, framebuffers: &[vk::Framebuffer]) {
    framebuffers
        .iter()
        .for_each(|f| device.destroy_framebuffer(*f, None));
}
