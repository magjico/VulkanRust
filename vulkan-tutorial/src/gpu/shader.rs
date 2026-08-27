use anyhow::Result;

use vulkanalia::bytecode::Bytecode;
use vulkanalia::prelude::v1_0::*;

//===========================================
// Descriptor Set
//===========================================

/// create **UBO** descriptor set layout.
/// 
/// **change every frame**
pub fn create_global_descriptor_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout> {
    // Base-color binding
    let ubo_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT);

    // Light binding
    let light_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(1)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT);

    let bindings = &[ubo_binding, light_binding];
    let info = vk::DescriptorSetLayoutCreateInfo::builder()
        .bindings(bindings);

    let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&info, None)? };

    Ok(descriptor_set_layout)
}

/// create a unique descriptor set layout for **all skins**.
pub fn create_skinning_descriptor_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout> {
    // Animation joint binding
    let binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::VERTEX);

    let bindings = &[binding];
    let info = vk::DescriptorSetLayoutCreateInfo::builder()
        .bindings(bindings);

    let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&info, None)? };

    Ok(descriptor_set_layout)
}

/// create material set layout in this order:
/// base color, metallic-roughness, normal map, occlusion map, emissive map.
/// 
/// **change by mesh**
pub fn create_material_descriptor_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout> {
    let bindings: Vec<vk::DescriptorSetLayoutBinding>  = (0..5u32)
        .map(|i| *vk::DescriptorSetLayoutBinding::builder()
            .binding(i)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        ).collect();
    
    let info = vk::DescriptorSetLayoutCreateInfo::builder()
        .bindings(&bindings);

    let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&info, None)? };
    Ok(descriptor_set_layout)
}

/// create a unique descriptor set layout for all model instances
pub fn create_instance_descriptor_set_layout(device: &Device) -> Result<vk::DescriptorSetLayout> {
    let binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::VERTEX);

    let bindings = &[binding];
    let info = vk::DescriptorSetLayoutCreateInfo::builder()
        .bindings(bindings);

    let descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&info, None)? };
    Ok(descriptor_set_layout)
}

//===========================================
// Shader pipeline and module
//===========================================

fn create_shader_module(device: &Device, bytecode: &[u8]) -> Result<vk::ShaderModule> {
    let bytecode = Bytecode::new(bytecode).unwrap();
    let info = vk::ShaderModuleCreateInfo::builder()
        .code(bytecode.code())
        .code_size(bytecode.code_size());
    Ok( unsafe { device.create_shader_module(&info, None)? })
}

pub struct ShaderStagesBuilder {
    modules: Vec<vk::ShaderModule>,
    stages: Vec<vk::ShaderStageFlags>,
}

impl ShaderStagesBuilder {
    pub fn new(
        device: &Device,
        shaders: &[(&[u8], vk::ShaderStageFlags)],
    ) -> Result<Self> {
        let mut modules = Vec::with_capacity(shaders.len());
        let mut stages = Vec::with_capacity(shaders.len());

        for (bytecode, stage) in shaders {
            modules.push(create_shader_module(device, bytecode)?);
            stages.push(*stage);
        }

        Ok(Self {
            modules,
            stages
        })
    }

    pub fn build(&self) -> Vec<vk::PipelineShaderStageCreateInfoBuilder<'_>> {
        self.modules.iter()
            .zip(&self.stages)
            .map(|(module, stage)| {
                vk::PipelineShaderStageCreateInfo::builder()
                    .stage(*stage)
                    .module(*module)
                    .name(b"main\0")
            })
            .collect()
    }

    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn destroy(&self, device: &Device) {
        for module in &self.modules {
            device.destroy_shader_module(*module, None);
        }
    }


}