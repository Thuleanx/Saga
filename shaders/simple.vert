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

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragUV;

void main() {
    vec3 worldPosition = vec3(instance.model * vec4(inPosition, 1.0));
    vec3 cameraPosition = vec3(global.view[0][3],global.view[1][3],global.view[2][3]);

    gl_Position = global.proj * global.view * instance.model * vec4(inPosition, 1.0);

    fragColor = vec3(length(cameraPosition - worldPosition) / 5);
    fragUV = inUV;
}
