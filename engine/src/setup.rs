//! Setup multiple **specific** app objects
use anyhow::anyhow;
use cgmath::Rotation3;
use cgmath::{One, Deg};
use anyhow::Result;

use vulkanalia::prelude::v1_0::*;
use bevy_ecs::prelude::*;
use bevy_ecs::schedule::IntoScheduleConfigs;

use crate::assets::load_model_with_offset;
use crate::ops::SlotAllocator;
use crate::math::*;
use crate::render::*;
use crate::scene::*;
use crate::constants::*;
use crate::type_safety::*;

//===============================================
// Bevy ECS World
//===============================================

pub fn init_ecs_context(
    device: &Device,
	instance: &Instance,
	physical_device: vk::PhysicalDevice,
) -> Result<ECSContext> {
    let mut ecs_context = ECSContext::init();
    let skinning_buff = SkinningBuffer::new(instance, device, physical_device)?;
    let instance_buff = InstanceBuffer::new(instance, device, physical_device)?;

    // world
    ecs_context.world.insert_resource(VulkanDevice(device.clone()));
    ecs_context.world.insert_resource(skinning_buff);
    ecs_context.world.insert_resource(instance_buff);
	ecs_context.world.insert_resource(SSBOSkiningAllocator(SlotAllocator::new(MAX_INSTANCES, MAX_JOINT_PER_INSTANCE)));
	ecs_context.world.insert_resource(Time::default());
    ecs_context.world.insert_resource(CurrentFrame::default());

    // schedule
    ecs_context.schedule.add_systems(
        (
            propagate_transforms_from_root,
            propagate_transforms_to_children,
            update_skeletons_wrapped
        ).chain()
    );

    Ok(ecs_context)
}

//===============================================
// models
//===============================================

/// load gltf models
pub fn load_gltf_models(
	device: &Device,
	instance: &Instance,
	physical_device: vk::PhysicalDevice,
	setup_command_buffer: vk::CommandBuffer,
	graphics_queue: vk::Queue,
	models: &mut ModelsStorage,
    textures: &mut TexturesStorage,
    model_registry: &mut ModelRegistry,
) -> Result<()> {
    for (model_path, model_key) in MODEL_INFO.iter() {
        load_model_with_offset(
            device,
            instance,
            physical_device,
            model_path,
            model_key,
            setup_command_buffer,
            graphics_queue,
            models,
            textures,
            model_registry
        )?;
    }

	Ok(())
}


/// spawn 4 models with the bevy ecs systems
pub fn spawn_from_cesium_man_instances(
	world: &mut World,
    model_registry: &ModelRegistry
) -> Result<Vec<Entity>> {
    let cesium_assets = model_registry.entries.get(CESIUM_MAN_KEY)
        .ok_or_else(|| anyhow!("key <{}> not found in registry", CESIUM_MAN_KEY))?;

    let positions = [
		Vec3::new(-2.0, -2.0, 0.0),
		Vec3::new(-2.0, 2.0, 0.0),
		Vec3::new(2.0, -2.0, 0.0),
		Vec3::new(2.0, 2.0, 0.0),
	];

	let entities = positions.iter().enumerate()
		.map(|(i, &pos)| {
			let transform =  Transform::new(
				pos,
				Quat::from_angle_y(Deg(90.0)),
				Vec3::new(1.0, 1.0, 1.0)
			);

            let anim_time =  0.5 * i as f32;

			ModelSpawnBuilder::new(cesium_assets.model_id)
				.with_animation(AnimationSpec::Animated { skin_id: SkinId(0), anim_id: AnimationId(0) })
                .animation_start_at(anim_time)
				.spawn_at(world, transform)
		})
		.collect::<Result<Vec<_>>>()?;

	Ok(entities)
}

pub fn spawn_from_brain_stem_instance(
    world: &mut World,
    model_registry: &ModelRegistry
) -> Result<Vec<Entity>> {
    let assets = model_registry.entries.get(BRAIN_STEM_KEY)
        .ok_or_else(|| anyhow!("key <{}> not found in registry", BRAIN_STEM_KEY))?;

    let positions = [
        Vec3::new(0.0, 0.0, 0.0)
    ];

    let entities = positions.iter()
        .map(|&pos| {
            let transform = Transform::new(
                pos,
                Quat::one(),
                Vec3::new(1.0, 1.0, 1.0)
            );

            ModelSpawnBuilder::new(assets.model_id)
                .with_animation(AnimationSpec::Animated { skin_id: SkinId(0), anim_id: AnimationId(0) })
                .spawn_at(world, transform)
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(entities)
}


//===============================================
// defaults
//===============================================

pub fn create_default_texture(
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice,
    setup_command_buffer: vk::CommandBuffer,
    graphics_queue: vk::Queue,
) -> Result<TextureData> {
    // White Pixel 1x1 RGBA
    let pixels = [255u8, 255, 255, 255];

    let extent = vk::Extent3D {
        width: 1,
        height: 1,
        depth: 1,
    };

    let mip_levels = 1;

    let (image, image_memory) = create_texture_image(
        instance,
        device,
        physical_device,
        setup_command_buffer,
        graphics_queue,
        extent,
        mip_levels,
        &pixels,
        vk::Format::R8G8B8A8_SRGB,
    )?;

    let image_view = create_texture_image_view(device, image, mip_levels)?;
    let sampler = create_texture_sampler(device, mip_levels as f32)?;

    Ok(TextureData {
        image,
        image_memory,
        image_view,
        sampler,
        mip_levels,
    })
}

pub fn create_default_lighting(
    instance: &Instance,
    device: &Device,
    physical_device: vk::PhysicalDevice
) -> Result<LightBuffer> {
    let mut light_buffer = LightBuffer::create(
        instance,
        device,
        physical_device,
        4
    )?;

    light_buffer.lights = vec![
        Light {
            position: Vec4::new(-10.0, 10.0, 10.0, 1.0),
            color:    Vec4::new(1.0, 0.0, 0.0, 15.0),
        },
        Light {
            position: Vec4::new(10.0, 10.0, 10.0, 1.0),
            color:    Vec4::new(1.0, 0.0, 0.0, 15.0),
        },
        Light {
            position: Vec4::new(-10.0, -10.0, 10.0, 1.0),
            color:    Vec4::new(1.0, 0.0, 0.0, 15.0),
        },
        Light {
            position: Vec4::new(10.0, -10.0, 10.0, 1.0),
            color:    Vec4::new(1.0, 0.0, 0.0, 15.0),
        },
    ];

    light_buffer.update(device)?;

    Ok(light_buffer)
}