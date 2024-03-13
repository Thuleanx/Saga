#version 450

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
} global;

layout(set = 1, binding = 0) uniform MeshUniformBufferObject {
    mat4 model;
} instance;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inUV;

layout(location = 1) out vec2 fragUV;

void main() {
    vec3 worldPosition = vec3(instance.model * vec4(inPosition, 1.0));

    gl_Position = global.proj * global.view * instance.model * vec4(inPosition, 1.0);

    fragUV = inUV;
}
