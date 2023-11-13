use anyhow::Result;
use vulkanalia::prelude::v1_0::*;

use super::app::App;
use super::appdata::AppData;

pub unsafe fn create_framebuffers(
    device: &Device,
    data: &mut AppData) 
-> Result<()> {
    data.framebuffers = data
        .swapchain_image_views
        .iter()
        .map(|i| {
            let attachments = &[*i];
            let create_info = vk::FramebufferCreateInfo::builder()
                .render_pass(data.render_pass)
                .attachments(attachments)
                .width(data.swapchain_extent.width)
                .height(data.swapchain_extent.height)
                .layers(1);

            device.create_framebuffer(&create_info, None)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(())
}

pub unsafe fn destroy_framebuffers(app: &App) {
    app.data.framebuffers
        .iter()
        .for_each(|f| app.device.destroy_framebuffer(*f, None));
}
