//! Draw the environment cubemap as a background, behind everything else.
//!
//! check: https://www.saschawillems.de/blog/2016/08/13/vulkan-tutorial-on-rendering-a-fullscreen-quad-without-buffers/
//! check: http://marcinignac.com/blog/pragmatic-pbr-hdr/ section 303
#version 450

// check render/uniform.rs/UniformBufferObject
layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;

    vec4 camPos;
    float exposure;
    float gamma;
    float prefilteredCubMipLevels;
    float scaleIBLAmbient;
} ubo;

layout(location = 0) out vec3 viewDir;

void main()
{
	vec2 UV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
	vec4 clip = vec4(UV * 2.0 - 1.0, 1.0, 1.0);

	// transform from the normalized device coordinates back to the view space
	vec4 unprojected = inverse(ubo.proj) * clip;
	
	// transfrom from the view space back to the world space
    // and use it as a sampling vector
	viewDir = transpose(mat3(ubo.view)) * (unprojected.xyz / unprojected.w);
	gl_Position = clip;
}