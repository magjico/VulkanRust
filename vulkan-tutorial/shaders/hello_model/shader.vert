#version 450

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
} ubo;

layout(set = 0, binding = 1) readonly buffer JointMats {
    mat4 joints[];
};

layout(push_constant) uniform PushConstants {
    mat4 model;
} pcs;

// Take a look at Vertex in geometry.rs to understand the "in" vector.
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(location = 3) in vec2 inTexCoord;
layout(location = 4) in uvec4 inJoint;
layout(location = 5) in vec4 inWeight;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;
layout(location = 2) out vec3 fragNormal;

void main() {
    mat4 skinMat =
        inWeight.x * joints[inJoint.x] +
        inWeight.y * joints[inJoint.y] +
        inWeight.z * joints[inJoint.z] +
        inWeight.w * joints[inJoint.w];

    mat4 finalModel = pcs.model * skinMat; 

    gl_Position = ubo.proj * ubo.view * finalModel * vec4(inPosition, 1.0);
    fragColor = inColor;
    fragTexCoord = inTexCoord;
    fragNormal = mat3(transpose(inverse(finalModel))) * inNormal;
}