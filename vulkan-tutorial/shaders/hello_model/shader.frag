#version 450

// Take a look at create_material_descriptor_set_layout in shader.rs
layout(set = 1, binding = 0) uniform sampler2D base_color_texture;
layout(set = 1, binding = 1) uniform sampler2D metallic_roughness_texture;
layout(set = 1, binding = 2) uniform sampler2D normal_texture;
layout(set = 1, binding = 3) uniform sampler2D occlusion_texture;
layout(set = 1, binding = 4) uniform sampler2D emissive_texture;

layout(push_constant) uniform PushConstants {
    layout(offset = 64) float opacity;
} pcs;

// Take a look at the out vector inside shader.vert
layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 fragNormal; 

layout(location = 0) out vec4 outColor;

void main() {
    vec4 base_color = texture(base_color_texture, fragTexCoord);
    outColor = vec4(base_color.rgb * fragColor, base_color.a * pcs.opacity);
}