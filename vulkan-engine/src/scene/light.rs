use anyhow::Result;
use std::ptr::copy_nonoverlapping as memcpy;

use vulkanalia::prelude::v1_0::*;

use crate::math::Vec4;
use crate::resources::create_buffer;

/// Represent a ponctual light source.
/// 
/// ## Fields
/// 
/// - `position` ( [Vec4] ) - xyz = position, w = radius.
/// - `color` ( [Vec4] ) - rgb = color, w = intensity.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct Light {
	pub position: Vec4,
	pub color: Vec4
}

#[derive(Debug, Clone)]
pub struct LightBuffer {
	pub buffer:		vk::Buffer,
	pub memory:		vk::DeviceMemory,
	pub lights:		Vec<Light>,
	pub max_lights:	usize,
}

impl LightBuffer {
	pub fn create(
		instance: &Instance,
		device: &Device,
		physical_device: vk::PhysicalDevice,
		max_lights: usize
	) -> Result<Self> {
		let size = (max_lights * size_of::<Light>()) as vk::DeviceSize;

		let (buffer, memory) = create_buffer(
			instance,
			device,
			physical_device,
			size,
			vk::BufferUsageFlags::STORAGE_BUFFER,
			vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
		)?;

		Ok(Self {
			buffer,
			memory,
			lights: Vec::new(),
			max_lights
		})
	}

	pub fn update(&self, device: &Device) -> Result<()> {
		let size = (self.lights.len() * size_of::<Light>()) as vk::DeviceSize;

		unsafe {
			let ptr = device.map_memory(self.memory, 0, size, vk::MemoryMapFlags::empty())?;
			memcpy(self.lights.as_ptr(), ptr.cast(), self.lights.len());
			device.unmap_memory(self.memory);
		}

		Ok(())
	}

	#[allow(unsafe_op_in_unsafe_fn)]
	pub unsafe fn destroy(&self, device: &Device) {
		device.destroy_buffer(self.buffer, None);
		device.free_memory(self.memory, None);
	}
}