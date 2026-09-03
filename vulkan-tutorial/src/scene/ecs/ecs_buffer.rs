use anyhow::Result;
use std::ptr::copy_nonoverlapping as memcpy;

use bevy_ecs::prelude::*;
use vulkanalia::prelude::v1_0::*;

use crate::constants::{FRAME_STRIDE, MAX_INSTANCES, MAX_INSTANCES_DATA, MAX_TOTAL_JOINTS};
use crate::math::Mat4;
use crate::resources::create_buffer;
use crate::render::InstanceData;

/// A host-visible storage buffer with a persistent mapping.
///
/// The memory is mapped once at creation and stays mapped until [destroy], since
/// mapping the same [vk::DeviceMemory] block more than once is illegal — only one
/// mapping at a time is allowed, including for disjoint regions. Mapping is neither
/// reference-counted nor thread-safe in Vulkan.
///
/// See <https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/memory_mapping.html>
///
/// The memory is `HOST_COHERENT`, so writes need no manual flush, and it is zeroed
/// at creation so unwritten regions read as valid zero data rather than garbage.
///
/// ## Fields
///
/// - `vk_buffer` ( [vk::Buffer] ) - The buffer handle bound to a descriptor.
/// - `vk_memory` ( [vk::DeviceMemory] ) - Backing allocation, host-visible and host-coherent.
/// - `mapped` ( `*mut u8` ) - Base pointer of the persistent mapping. Typed access goes through [as_ptr].
/// - `range` ( [vk::DeviceSize] ) - Total byte size of the buffer, used both to allocate it and to set the descriptor range.
pub struct StorageBuffer {
    pub vk_buffer: vk::Buffer,
    pub vk_memory: vk::DeviceMemory,
    mapped: *mut u8,
    range: vk::DeviceSize,
}

impl StorageBuffer {
    fn new(
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
        size: vk::DeviceSize,
    ) -> Result<Self> {
        let (vk_buffer, vk_memory) = create_buffer(
            instance,
            device,
            physical_device,
            size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        )?;

        let mapped = unsafe {
            let ptr = device.map_memory(vk_memory, 0, size, vk::MemoryMapFlags::empty())?;
            std::ptr::write_bytes(ptr.cast::<u8>(), 0, size as usize);
            ptr.cast::<u8>()
        };

        Ok(Self {
            vk_buffer,
            vk_memory,
            mapped,
            range: size,
        })
    }

    #[allow(unsafe_op_in_unsafe_fn)]
	unsafe fn destroy(&self, device: &Device) {
        device.unmap_memory(self.vk_memory);
		device.destroy_buffer(self.vk_buffer, None);
		device.free_memory(self.vk_memory, None);
	}

    #[inline]
    fn as_ptr<T>(&self) -> *mut T {
        self.mapped.cast::<T>()
    }

    /// Total byte size of the buffer, used both to allocate it and to set the
    /// descriptor range.
    #[inline]
    pub fn get_range(&self) -> vk::DeviceSize { self.range }
}

/// Joint matrices of every skinned instance, for every frame in flight.
///
/// The buffer is laid out as `MAX_FRAMES_IN_FLIGHT` consecutive regions of
/// [FRAME_STRIDE] matrices.
///
/// Duplicating per frame in flight prevents the CPU from overwriting matrices a
/// pending frame is still reading.
///
/// The vertex shader reads it as set 2, indexing with the per-instance offset
/// carried in [InstanceData].
#[derive(Resource)]
pub struct SkinningBuffer(StorageBuffer);

impl SkinningBuffer {
    pub fn new(
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        let size = (MAX_TOTAL_JOINTS * size_of::<Mat4>()) as vk::DeviceSize;
        let storage = StorageBuffer::new(instance, device, physical_device, size)?;

        Ok(Self(storage))
    }

    /// Writes `matrices` into this instance's slot for the given frame in flight.
    ///
    /// ## Arguments
    ///
    /// - `frame` ( `usize` ) - Frame in flight index, selecting which region to write into.
    /// - `offset` ( `u32` ) - Slot offset within the region, as handed out by the skinning allocator.
    /// - `matrices` ( &\[[Mat4]] ) - Joint matrices of one skeleton.
    #[inline]
    pub fn write_slice(&self, frame: usize, offset: u32, matrices: &[Mat4]) {
        let base = frame * FRAME_STRIDE + offset as usize;
        unsafe {
            memcpy(matrices.as_ptr(), self.0.as_ptr::<Mat4>().add(base), matrices.len());
        }
    }

    #[inline]
    #[allow(unsafe_op_in_unsafe_fn)]
	pub unsafe fn destroy(&self, device: &Device) { self.0.destroy(device); }

    #[inline]
    pub fn storage(&self) -> &StorageBuffer { &self.0 }
}

/// Per-instance world matrix and skinning offset, for every frame in flight.
///
/// The buffer is laid out as `MAX_FRAMES_IN_FLIGHT` consecutive regions of
/// [MAX_INSTANCES] entries. Instances are written in the order the renderer sorts
/// them, so instances of the same model form contiguous ranges addressable through
/// the `firstInstance` argument of `vkCmdDrawIndexed`.
///
/// The vertex shader reads it as set 3, indexing with `gl_InstanceIndex`.
#[derive(Resource)]
pub struct InstanceBuffer(StorageBuffer);

impl InstanceBuffer {
    pub fn new(
        instance: &Instance,
        device: &Device,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        let size = (MAX_INSTANCES_DATA * size_of::<InstanceData>()) as vk::DeviceSize;
        let storage = StorageBuffer::new(instance, device, physical_device, size)?;

        Ok(Self(storage))
    }

    pub fn write(&self, frame: usize, data: &[InstanceData]) {
        let base = frame * MAX_INSTANCES as usize;
        unsafe {
            memcpy(data.as_ptr(), self.0.as_ptr::<InstanceData>().add(base), data.len());
        }
    }

    #[inline]
    #[allow(unsafe_op_in_unsafe_fn)]
	pub unsafe fn destroy(&self, device: &Device) { self.0.destroy(device); }

    #[inline]
    pub fn storage(&self) -> &StorageBuffer { &self.0 }
}


unsafe impl Send for StorageBuffer {}
unsafe impl Sync for StorageBuffer {}

// Send and Sync are justified: the mapped pointer will be valid for all SkinningBuffer lifetime (map in new, unmap in destroy)
// NO DANGLING. And memory is "HOST_COHERENT" => no manual flush / invalidate to do
// Only `update_skeletons` writes to it, and bevy_ecs guarantees a single system instance at a time.
// TODO: Revisit if skinning writes ever become multi-threaded.
unsafe impl Send for SkinningBuffer {}
unsafe impl Sync for SkinningBuffer {}

// The mapped pointer stays valid for the entire lifetime of the buffer
// (mapped in `create`, unmapped in `destroy`) — no dangling. Memory is HOST_COHERENT,
// so no manual flush/invalidate is required.
// Only `CommandData::record_scene` writes to it, from the single render thread.
// TODO: Revisit if command buffer recording ever becomes multi-threaded.
unsafe impl Send for InstanceBuffer {}
unsafe impl Sync for InstanceBuffer {}

