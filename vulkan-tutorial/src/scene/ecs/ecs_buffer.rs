use anyhow::Result;
use std::ptr::copy_nonoverlapping as memcpy;

use bevy_ecs::prelude::*;
use vulkanalia::prelude::v1_0::*;

use crate::constants::MAX_TOTAL_JOINTS;
use crate::math::Mat4;
use crate::resources::create_buffer;

#[derive(Resource)]
pub struct SkinningBuffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
}

impl SkinningBuffer {
    pub fn create(
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        let size = (MAX_TOTAL_JOINTS * size_of::<Mat4>()) as vk::DeviceSize;

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
            memory
        })
    }

    pub fn write_slice(
        &self,
        device: &Device,
        offset: u32,
        joint_matrices: &[Mat4]
    ) -> Result<()> {
        let byte_offset = (offset as usize * size_of::<Mat4>()) as vk::DeviceSize;
        let size = (joint_matrices.len() * size_of::<Mat4>()) as vk::DeviceSize;

        unsafe {
            let ptr = device.map_memory(self.memory, byte_offset, size, vk::MemoryMapFlags::empty())? as *mut u8;
            memcpy(joint_matrices.as_ptr() as *const u8, ptr, joint_matrices.len());
            device.unmap_memory(self.memory);
        }

        Ok(())
    }

    #[allow(unsafe_op_in_unsafe_fn)]
	pub unsafe fn destroy(&self, device: &Device) {
		device.destroy_buffer(self.buffer, None);
		device.free_memory(self.memory, None);
	}

    #[inline]
    pub fn get_range() -> vk::DeviceSize { (MAX_TOTAL_JOINTS * size_of::<Mat4>()) as vk::DeviceSize }
}