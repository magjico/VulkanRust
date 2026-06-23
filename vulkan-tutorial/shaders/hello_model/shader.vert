#version 450

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
} ubo;

layout(set = 2, binding = 0) readonly buffer JointMats {
    mat4 joints[];
};

layout(push_constant) uniform PushConstants {
    mat4 model;
    layout(offset = 64) int has_skin;
} pcs;

// Take a look at Vertex in geometry.rs to understand the "in" vector.
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(location = 3) in vec2 inTexCoord;
layout(location = 4) in uvec4 inJointIndices;
layout(location = 5) in vec4 inJointWeight;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;
layout(location = 2) out vec3 fragNormal;

void main() {
    mat4 finalModel = pcs.model;

    if (pcs.has_skin != -1) {
        mat4 skinMat =
            inJointWeight.x * joints[inJointIndices.x] +
            inJointWeight.y * joints[inJointIndices.y] +
            inJointWeight.z * joints[inJointIndices.z] +
            inJointWeight.w * joints[inJointIndices.w];

        finalModel *= skinMat;
    }

    gl_Position = ubo.proj * ubo.view * finalModel * vec4(inPosition, 1.0);
    fragColor = inColor;
    fragTexCoord = inTexCoord;
    fragNormal = mat3(transpose(inverse(finalModel))) * inNormal;
}