use anyhow::Result;
use std::ptr::copy_nonoverlapping as memcpy;

use bevy_ecs::prelude::*;
use vulkanalia::prelude::v1_0::*;

use crate::constants::{MAX_TOTAL_JOINTS, FRAME_STRIDE};
use crate::math::Mat4;
use crate::resources::create_buffer;

/// A skinning buffer structre
/// 
/// ## Fields
/// 
/// - `buffer` ( [vk::Buffer] ) - The container (buffer).
/// - `memory` ( [vk::DeviceMemory] ) - The buffer memory (IS HOST COHERENT).
/// - `mapped` ( `*mut Mat4` ) - Pointer to the skin matrices map in memory. We need that because
/// "mapping the same [vk::DeviceMemory] block multiple times is illegal - only one mapping at a time is allowed.
/// This includes mapping disjoint regions. Mapping is not reference-counted internally by Vulkan.
/// It is also not thread-safe."
/// see *https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/memory_mapping.html*
#[derive(Resource)]
pub struct SkinningBuffer {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    
    mapped: *mut Mat4,
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

        let mapped = unsafe {
            let ptr = device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty())?;
            std::ptr::write_bytes(ptr.cast::<u8>(), 0, size as usize);
            ptr.cast::<Mat4>()
        };

        Ok(Self {
            buffer,
            memory,
            mapped
        })
    }

    pub fn write_slice(&self, frame: usize, offset: u32, matrices: &[Mat4]) {
        let base = frame * FRAME_STRIDE + offset as usize;
        unsafe {
            memcpy(matrices.as_ptr(), self.mapped.add(base), matrices.len());
        }
    }

    #[allow(unsafe_op_in_unsafe_fn)]
	pub unsafe fn destroy(&self, device: &Device) {
        device.unmap_memory(self.memory);
		device.destroy_buffer(self.buffer, None);
		device.free_memory(self.memory, None);
	}

    #[inline]
    pub fn get_range() -> vk::DeviceSize { (MAX_TOTAL_JOINTS * size_of::<Mat4>()) as vk::DeviceSize }
}

// Send and Sync are justified: the mapped pointer will be valid for all SkinningBuffer lifetime (map in new, unmap in destroy)
// NO DANGLING. And memory is "HOST_COHERENT" => no manual flush / invalidate to do
// Only `update_skeletons` writes to it, and bevy_ecs guarantees a single system instance at a time.
// TODO: Revisit if skinning writes ever become multi-threaded.
unsafe impl Send for SkinningBuffer {}
unsafe impl Sync for SkinningBuffer {}