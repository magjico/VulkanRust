use anyhow::Result;
use vulkanalia::vk::DeviceMemory;

use std::collections::HashMap;
use std::ptr::copy_nonoverlapping as memcpy;
use std::mem::size_of_val;

use vulkanalia::prelude::v1_0::*;

use crate::gpu::get_memory_type_index;
use crate::render::UniformBufferObject;
use crate::scene::{Vertex, ModelsStorage};
use crate::type_safety::{MeshOffset, ModelId, NodeId};

use super::{begin_setup_command_buffer, flush_setup_command_buffer};

//================================================
// Buffers (general)
//================================================

/// Create and return a buffer and its memory object.
/// 
/// ## Arguments
/// 
/// - `instance` (&[`Instance`]) - Vulkan instance.
/// - `device` (&[`Device`]) - Vulkan device.
/// - `physical_device` ([`vk::PhysicalDevice`]) - The physical device.
/// - `size` ([`vk::DeviceSize`]) - Buffer size to allocate.
/// - `usage` ([`vk::BufferUsageFlags`]) - Buffer usage for optimization.
/// - `properties` ([`vk::MemoryPropertyFlags`]) - Buffer properties.
pub fn create_buffer(
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    // Buffer initialization
    let buffer_info = vk::BufferCreateInfo::builder()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = unsafe { device.create_buffer(&buffer_info, None)? };

    // Memory allocation
    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

    let memory_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(requirements.size)
        .memory_type_index(get_memory_type_index(instance, physical_device, properties, requirements)?);

    let buffer_memory = unsafe { device.allocate_memory(&memory_info, None)? };

    unsafe { device.bind_buffer_memory(buffer, buffer_memory, 0)? };

    Ok((buffer, buffer_memory))
}

/// Copy element from one source buffer to a destination buffer
/// 
/// ## Arguments
/// 
/// - `device` (&[`Device`]) - Vulkan device.
/// - `source` ([`vk::Buffer`]) - src buffer to copy.
/// - `destination` ([`vk::Buffer`]) - destination buffer to receive the copy.
/// - `setup_command_buffer` ([`vk::CommandBuffer`]) - command buffer to manage command.
/// - `sizes` (`&[vk::DeviceSize]`) - byte size of each region to copy.
/// - `dst_offsets` (`&[vk::DeviceSize]`) - where in the destination buffer should each region be copied.
/// - `graphics_queue` ([`vk::Queue`]) - the graphic queue to execute (gpu) command needed for the use of the setup command buffer.
pub fn copy_buffers(
    device: &Device,
    source: vk::Buffer,
    destination: vk::Buffer,
    setup_command_buffer: vk::CommandBuffer,
    sizes: &[vk::DeviceSize],
    dst_offsets: &[vk::DeviceSize],
    graphics_queue: vk::Queue,
) -> Result<()> {
    begin_setup_command_buffer(device, setup_command_buffer)?;

    // Commands
    let mut src_offset = 0;
    let mut regions = Vec::<vk::BufferCopy>::with_capacity(sizes.len());

    for (i, &size) in sizes.iter().enumerate() {
        regions.push(
            vk::BufferCopy::builder()
                .src_offset(src_offset)
                .dst_offset(dst_offsets[i])
                .size(size)
                .build()
        );

        src_offset += size;
    }
    unsafe { device.cmd_copy_buffer(setup_command_buffer, source, destination, &regions) };

    flush_setup_command_buffer(device, setup_command_buffer, graphics_queue)?;

    Ok(())
}

/// Destroy a list of buffers and free their memories
/// 
/// ## Arguments
/// 
/// - `device` (`&Device`) - Vulkan Device.
/// - `buffers` (`&[vk`) - the buffers to destroys.
/// - `buffers_memory` (`&[vk`) - the memories to free.
pub fn destroy_buffers(
    device: &Device,
    buffers: &[vk::Buffer],
    buffers_memory: &[vk::DeviceMemory]
) {
    unsafe {
        buffers_memory.iter().for_each(|mem| device.free_memory(*mem, None));
        buffers.iter().for_each(|buf| device.destroy_buffer(*buf, None));
    };
}

//===============================================
// FrameBuffer
//===============================================

/// Generate a framebuffer for the msaa sampling inside the swapchain.
/// useless in dynamic rendering.
pub fn create_framebuffers(
    device: &Device,
    render_pass: vk::RenderPass,
    swapchain_image_views: &[vk::ImageView],
    color_view: vk::ImageView,
    depth_view: vk::ImageView,
    width: u32,
    height: u32,
) -> Result<Vec<vk::Framebuffer>> {
    let framebuffers = swapchain_image_views
        .iter()
        .map(|i| {
            let attachments = &[color_view, depth_view, *i];
            let create_info = vk::FramebufferCreateInfo::builder()
                .render_pass(render_pass)
                .attachments(attachments)
                .width(width)
                .height(height)
                .layers(1);

            unsafe { device.create_framebuffer(&create_info, None) }
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(framebuffers)
}

//===============================================
// Interleaved Buffer (Vertex and Index Buffers)
//===============================================

/// Create an interleaved-buffer: a buffer that contain both the information of the index buffer and the vertex buffer
/// First the vertex buffer then the index one.
/// 
/// ## Arguments
/// 
/// - `instance` ( &[Instance] ) - Vulkan instance.
/// - `device` ( &[Device] ) - Vulkan device.
/// - `physical_device` ( [vk::PhysicalDevice] ) - a physical device.
/// - `vertices` ( &[[Vertex]] ) - array of vertices.
/// - `indices` ( &\[u32] ) - array of vertex indices.
/// - `setup_command_buffer` ( [vk::CommandBuffer] ) - setup command buffer use for intern operation.
/// - `graphics_queue` ( [vk::Queue] ) - necessary to work with the setup command buffer.
/// 
/// ## Returns
/// 
/// - `Result<(vk::Buffer, vk::DeviceMemory, u64)>` - The interleaved_buffer, the device memory associated,
///   and finally an index that says where the index part start in the buffer.
/// ```
pub fn create_interleaved_buffer(
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    vertices: &[Vertex],
    indices: &[u32],
    setup_command_buffer: vk::CommandBuffer,
    graphics_queue: vk::Queue,
) -> Result<(vk::Buffer, vk::DeviceMemory, u64)> {
    let vertex_size = size_of_val(vertices) as u64;
    let index_size = size_of_val(indices) as u64;
    let size = vertex_size + index_size;

    let (staging_buffer, staging_buffer_memory) = create_buffer(
        instance,
        device,
        physical_device,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;

    // Copy (staging)
    unsafe {
        let memory =  device.map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty())? as *mut u8;
        memcpy(vertices.as_ptr() as *const u8, memory, vertex_size as usize);
        memcpy(indices.as_ptr() as *const u8, memory.add(vertex_size as usize), index_size as usize);
        device.unmap_memory(staging_buffer_memory);
    }

    // bytes size alignment
    let alignment = size_of::<u32>() as u64;
    let aligned_vertex_size = (vertex_size + alignment - 1) & !(alignment - 1);
    let size = aligned_vertex_size + index_size;

    // Create (interleaved buffer)
    let (interleaved_buffer, interleaved_buffer_memory) = create_buffer(
        instance,
        device,
        physical_device,
        size,
        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL
    )?;

    // Copy
    copy_buffers(
        device,
        staging_buffer,
        interleaved_buffer,
        setup_command_buffer,
        &[vertex_size, index_size],
        &[0, aligned_vertex_size],
        graphics_queue,
    )?;

    // Cleanup
    destroy_buffers(device, &[staging_buffer], &[staging_buffer_memory]);

    Ok((interleaved_buffer, interleaved_buffer_memory, aligned_vertex_size))
}

//===============================================
// Uniform Buffers
//===============================================

/// Create the selected number of uniform buffers, and their associated memory.
/// Then return them. 
/// 
/// ## Arguments
/// 
/// - `instance` ( &[Instance] ) - Vulkan instance.
/// - `device` ( &[Device] ) - Vulkan device.
/// - `physical_device` ( [vk::PhysicalDevice] ) - a physical device.
/// - `count` ( usize ) - number of buffers to create.
/// 
/// ## Returns
/// 
/// - `Result<(Vec<vk::Buffer>, Vec<vk::DeviceMemory>)>` - the buffers and their memories.
pub fn create_uniform_buffers(
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    count: usize,
) -> Result<(Vec<vk::Buffer>, Vec<vk::DeviceMemory>)> {
    let mut uniform_buffers = Vec::new();
    let mut uniform_buffers_memory = Vec::new();

    for _ in 0..count {
        let (uniform_buffer, uniform_buffer_memory) = create_buffer(
            instance,
            device,
            physical_device,
            size_of::<UniformBufferObject>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        )?;

        uniform_buffers.push(uniform_buffer);
        uniform_buffers_memory.push(uniform_buffer_memory);
    }
    

    Ok((uniform_buffers, uniform_buffers_memory))

}

/// Re-create a list of uniform buffers and their memories
/// 
/// ## Arguments
/// 
/// - `instance` ( &[Instance] ) - Vulkan instance.
/// - `device` ( &[Device] ) - Vulkan device.
/// - `physical_device` ( [vk::PhysicalDevice] ) - a physical device.
/// - `uniform_buffers` ( &mut Vec<[vk::Buffer]> ) - the uniform buffers to recreates.
/// - `uniform_buffers_memory` ( &mut Vec<[vk::DeviceMemory]> ) - the uniform buffers memories associated.
/// - `count` (  usize ) - the number of buffers to recreate.
pub fn recreate_uniform_buffers(
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    uniform_buffers: &mut Vec<vk::Buffer>,
    uniform_buffers_memory: &mut Vec<vk::DeviceMemory>,
    count: usize,
) -> Result<()> {
    destroy_buffers(device, uniform_buffers, uniform_buffers_memory);

    // TODO: Maybe this should recreate the exact same number of buffer
    // so we can get rid of the count args for .len() 
    (*uniform_buffers, *uniform_buffers_memory) = create_uniform_buffers(
        instance,
        device,
        physical_device,
        count
    )?;

    Ok(())
}

//===============================================
// Structure for app: GeometryBuffer & UniformBuffers
//===============================================

/// Vertex and index data of every loaded model, packed into a single shared buffer.
///
/// Vertices and indices of all models are concatenated at load time so the renderer
/// binds them once per frame and draws each mesh through offsets instead. Those
/// offsets are recorded in `mesh_offsets`, keyed by the model and node they belong to.
///
/// This buffer is `DEVICE_LOCAL` and immutable once built; it survives swapchain
/// recreation.
///
/// ## Fields
///
/// - `buffer` ( [vk::Buffer] ) - Holds vertices first, then indices.
/// - `memory` ( [vk::DeviceMemory] ) - Backing allocation.
/// - `interleaved_offset` ( `u64` ) - Byte offset where the index data starts.
/// - `mesh_offsets` ( HashMap<([ModelId], [NodeId]), [MeshOffset]> ) - Where each mesh sits in the buffer. The key pairs a model with a node because node indices restart at zero for every model.
#[derive(Debug)]
pub struct GeometryBuffer {
	pub vk_buffer:			vk::Buffer,
	pub vk_buffer_memory:	vk::DeviceMemory,
	pub interleaved_offset:	u64,
	pub mesh_offsets:		HashMap<(ModelId, NodeId), MeshOffset>,
}


impl GeometryBuffer {
	pub fn new(
		instance:				&Instance,
		device:					&Device,
		physical_device:		vk::PhysicalDevice,
		models:					&ModelsStorage,
		setup_command_buffer:	vk::CommandBuffer,
		graphics_queue:			vk::Queue
	) -> Result<Self> {
		let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut mesh_offsets = HashMap::new();

		for (model_id, model) in models.iter().enumerate() {
            let model_id = ModelId(model_id);

            for (node_idx, node) in model.graph.iter().enumerate() {
                if let Some(mesh) = &node.value.mesh {
                    mesh_offsets.insert(
                        (model_id, NodeId(node_idx)),
                        MeshOffset {vertex_offset: vertices.len() as u32, first_index: indices.len() as u32 }
                    );

                    vertices.extend_from_slice(&mesh.vertices);
                    indices.extend_from_slice(&mesh.indices);
                }
            }
        }

        let (vk_buffer, vk_buffer_memory, interleaved_offset) = create_interleaved_buffer(
            instance,
            device,
            physical_device,
            &vertices,
            &indices,
            setup_command_buffer,
            graphics_queue,
        )?;

		Ok(Self {
			vk_buffer,
			vk_buffer_memory,
			interleaved_offset,
			mesh_offsets
		})
	}

    /// ## Safety
    /// 
    /// The caller must ensure the device is idle and that no pending command buffer
    /// still references these resources.
	#[rustfmt::skip]
    #[allow(unsafe_op_in_unsafe_fn)]
	pub unsafe fn destroy(&self, device: &Device) {
		destroy_buffers(device, &[self.vk_buffer], &[self.vk_buffer_memory]);
	}

	#[inline]
	pub fn get_mesh_offset(&self, model_id: ModelId, node_id: NodeId) -> MeshOffset {
		*self.mesh_offsets.get(&(model_id, node_id))
			.unwrap_or_else(|| panic!("No mesh offset for ({:?}, {:?})", model_id, node_id))
	}
}

/// Per-swapchain-image uniform buffers holding the frame's camera and PBR parameters.
///
/// One buffer per image is required because the CPU writes the next frame's values
/// while a pending frame may still be reading the previous ones. They are destroyed
/// and recreated on swapchain recreation, since the image count may change.
///
/// ## Fields
///
/// - `vk_buffers` ( Vec<[vk::Buffer]> ) - One buffer per swapchain image, in index order.
/// - `vk_buffers_memories` ( Vec<[vk::DeviceMemory]> ) - Backing allocations, in matching order.
#[derive(Debug)]
pub struct UniformBuffers {
	pub vk_buffers:				Vec<vk::Buffer>,
	pub vk_buffers_memories:	Vec<DeviceMemory>
}

impl UniformBuffers {
	pub fn new(
		instance:			&Instance,
		device:				&Device,
		physical_device:	vk::PhysicalDevice,
		images_count:		usize,
	) -> Result<Self> {
		let (vk_buffers, vk_buffers_memories) = create_uniform_buffers(
            instance,
            device,
            physical_device,
            images_count,
        )?;

		Ok(Self {
			vk_buffers,
			vk_buffers_memories
		})
	}

    /// ## Safety
    /// 
    /// The caller must ensure the device is idle and that no pending command buffer
    /// still references these resources.
	#[rustfmt::skip]
    #[allow(unsafe_op_in_unsafe_fn)]
	pub unsafe fn destroy(&self, device: &Device) {
		destroy_buffers(device, &self.vk_buffers, &self.vk_buffers_memories);
	}

    /// Uploads the frame's uniform data into the buffer of the given swapchain image.
    ///
    /// ## Safety
    ///
    /// The caller must ensure the GPU is not currently reading the buffer for
    /// `image_index`.
    ///
    /// `image_index` must be within bounds of the allocated buffers.
    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn write(
        &self,
        device:			&Device,
        image_index:	usize,
        ubo:			&UniformBufferObject,
    ) -> Result<()> {
        let memory = device.map_memory(
            self.vk_buffers_memories[image_index],
            0,
            size_of::<UniformBufferObject>() as u64,
            vk::MemoryMapFlags::empty()
        )?;

        memcpy(ubo, memory.cast(), 1);
        device.unmap_memory(self.vk_buffers_memories[image_index]);

        Ok(())
    }
}