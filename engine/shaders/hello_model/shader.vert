#version 450

struct InstanceData {
    mat4 model;
    uint ssboOffset;
};

// check uniform.rs inside render to understand UBO.
layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;

    // unused for now (or in use but in the frag)
    vec4 camPos;
    float exposure;
    float gamma;
    float prefilteredCubMipLevels;
    float scaleIBLAmbient;
} ubo;

layout(set = 2, binding = 0) readonly buffer JointMats {
    mat4 joints[];
};

layout(set = 3, binding = 0) readonly buffer Instances {
    InstanceData instances[];
};

layout(push_constant) uniform PushConstants {
    mat4 nodeMatrix;
} pcs;

const uint NO_SKIN = 0xFFFFFFFFu;

// Take a look at Vertex in geometry.rs to understand the "in" vectors.
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(location = 3) in vec2 inUV0;
layout(location = 4) in vec2 inUV1;
layout(location = 5) in vec4 inTangent;
layout(location = 6) in uvec4 inJointIndices;
layout(location = 7) in vec4 inJointWeight;

// Take a look at the fragment shader to understand the "frag" vectors.
layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec2 fragUV0;
layout(location = 3) out vec2 fragUV1;
layout(location = 4) out vec4 fragTangent;

void main() {
    InstanceData inst = instances[gl_InstanceIndex];

    mat4 finalModel = inst.model * pcs.nodeMatrix;

    if (inst.ssboOffset != NO_SKIN) {
        mat4 skinMat =
            inJointWeight.x * joints[inst.ssboOffset + inJointIndices.x] +
            inJointWeight.y * joints[inst.ssboOffset + inJointIndices.y] +
            inJointWeight.z * joints[inst.ssboOffset + inJointIndices.z] +
            inJointWeight.w * joints[inst.ssboOffset + inJointIndices.w];

        finalModel *= skinMat;
    }

    vec4 worldPos = finalModel * vec4(inPosition, 1.0);
    mat3 normalMat = transpose(inverse(mat3(finalModel)));
    
    fragWorldPos = worldPos.xyz;
    fragNormal = normalize(normalMat * inNormal);
    fragUV0 = inUV0;
    fragUV1 = inUV1;
    fragTangent = vec4(normalize(normalMat * inTangent.xyz), inTangent.w);

    gl_Position = ubo.proj * ubo.view * worldPos;
    // gl_Position = vec4(inPosition * 0.5, 1.0);
}