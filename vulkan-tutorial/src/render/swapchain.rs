use anyhow::Result;

use winit::window::Window;

use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::KhrSurfaceExtensionInstanceCommands;
use vulkanalia::vk::KhrSwapchainExtensionDeviceCommands;

use crate::resources::create_image_view;
use crate::gpu::QueueFamilyIndices;

//========================================
// Swapchain Resources
//========================================

/// Generate a swapchain and all its related objects.
/// 
/// ## Returns
/// 
/// `Result<(vk::SwapchainKHR, vk::Format, vk::Extent2D, Vec<vk::Image>)>`:
/// - The swapchain
/// - The swapchain format
/// - The swapchain extent
/// - The swapchain images
pub fn create_swapchain(
    window: &Window,
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    queue_family_indices: &mut QueueFamilyIndices,
) -> Result<(vk::SwapchainKHR, vk::Format, vk::Extent2D, Vec<vk::Image>)> {
    // Image
    let graphics = queue_family_indices.get(vk::QueueFlags::GRAPHICS)?;
    let present = queue_family_indices.present;
    let support = SwapchainSupport::get(instance, surface, physical_device)?;

    let surface_format = get_swapchain_surface_format(&support.formats);
    let present_mode = get_swapchain_present_mode(&support.present_modes);
    let extent = get_swapchain_extent(window, support.capabilities);

    let swapchain_format = surface_format.format;
    let swapchain_extent = extent;

    let mut image_count = support.capabilities.min_image_count + 1;
    if support.capabilities.max_image_count != 0 && image_count > support.capabilities.max_image_count {
        image_count = support.capabilities.max_image_count;
    }

    let mut queue_family_indices = vec![];
    let image_sharing_mode = if graphics != present {
        queue_family_indices.push(graphics);
        queue_family_indices.push(present);
        vk::SharingMode::CONCURRENT
    } else {
        vk::SharingMode::EXCLUSIVE
    };

    // Create

    let info = vk::SwapchainCreateInfoKHR::builder()
        .surface(surface)
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

    let swapchain = unsafe { device.create_swapchain_khr(&info, None)? };

    // Images
    let swapchain_images = unsafe { device.get_swapchain_images_khr(swapchain)? };

    Ok((swapchain, swapchain_format, swapchain_extent, swapchain_images))
}

pub fn create_swapchain_image_views(
    device: &Device,
    swapchain_images: &[vk::Image],
    swapchain_format: vk::Format,
) -> Result<Vec<vk::ImageView>> {
    let swapchain_image_views = swapchain_images
        .iter()
        .map(|i| create_image_view(device, *i, swapchain_format, vk::ImageAspectFlags::COLOR, 1))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(swapchain_image_views)
}

fn get_swapchain_surface_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
    formats
        .iter()
        .cloned()
        .find(|f| f.format == vk::Format::B8G8R8A8_SRGB && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
        .unwrap_or_else(|| formats[0])
}

fn get_swapchain_present_mode(present_modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
    present_modes
        .iter()
        .cloned()
        .find(|m| *m == vk::PresentModeKHR::MAILBOX)
        .unwrap_or(vk::PresentModeKHR::FIFO)
}

fn get_swapchain_extent(window: &Window, capabilities: vk::SurfaceCapabilitiesKHR) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::MAX {
        capabilities.current_extent
    } else {
        vk::Extent2D::builder()
            .width(window.inner_size().width.clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
            ))
            .height(window.inner_size().height.clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ))
            .build()
    }
}

//========================================
// Swapchain Support
//========================================
#[derive(Debug)]
pub struct SwapchainSupport {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupport {
    pub fn get(instance: &Instance, surface: vk::SurfaceKHR, physical_device: vk::PhysicalDevice) -> Result<Self> {
        Ok(Self {
            capabilities: unsafe { instance.get_physical_device_surface_capabilities_khr(physical_device, surface)? },
            formats: unsafe { instance.get_physical_device_surface_formats_khr(physical_device, surface)? },
            present_modes: unsafe { instance.get_physical_device_surface_present_modes_khr(physical_device, surface)? },
        })
    }
}

//========================================
// Swapchain
//========================================

#[derive(Debug)]
pub struct Swapchain {
	pub vk_format:		vk::Format,
	pub vk_extent:		vk::Extent2D,
	pub vk_swapchain:	vk::SwapchainKHR,
	pub vk_images:		Vec<vk::Image>,
	pub vk_image_views:	Vec<vk::ImageView>,
}

impl Swapchain {
	pub fn new(
		window:					&Window,
		instance:				&Instance,
		device:					&Device,
		physical_device:		vk::PhysicalDevice,
		surface:				vk::SurfaceKHR,
		queue_family_indices:	&mut QueueFamilyIndices,
	) -> Result<Self> {
		let (vk_swapchain, vk_format, vk_extent, vk_images) = create_swapchain(
            window,
            instance,
            device,
            physical_device,
            surface,
            queue_family_indices
        )?;

        let vk_image_views = create_swapchain_image_views(
            &device,
            &vk_images,
            vk_format
        )?;

		Ok(Self {
			vk_format,
			vk_extent,
			vk_swapchain,
			vk_images,
			vk_image_views
		})
	}

	#[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn destroy(&self, device: &Device) {
        self.vk_image_views.iter().for_each(|v| device.destroy_image_view(*v, None));
        device.destroy_swapchain_khr(self.vk_swapchain, None);
    }
}