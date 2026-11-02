//! Sample the environment cubemap along each pixel's view direction.
//!
//! check: https://webgl2fundamentals.org/webgl/lessons/webgl-skybox.html
#version 450

layout(set = 0, binding = 0) uniform UniformBufferObject {
	mat4 view;
	mat4 proj;

	vec4 camPos;
	float exposure;
	float gamma;
	float prefilteredCubMipLevels;
	float scaleIBLAmbient;
} ubo;

layout(set = 1, binding = 0) uniform textureCube envMap;
layout(set = 1, binding = 1) uniform sampler envSampler;

layout(location = 0) in vec3 viewDir;
layout(location = 0) out vec4 outColor;

/*
	* ACES tonemapping, same as the one in shader.frag
*/
vec3 ACESFilm(vec3 x) {
	return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

void main()
{
	vec3 color = texture(samplerCube(envMap, envSampler), normalize(viewDir)).rgb;
	outColor = vec4(ACESFilm(color * ubo.exposure), 1.0);
}