#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) in vec3 inPosition;
//layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 fragColor;

void main() {
    vec3 worldPosition = vec3(ubo.model * vec4(inPosition, 1.0));
    vec3 cameraPosition = vec3(ubo.view[0][3],ubo.view[1][3],ubo.view[2][3]);

    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);

    fragColor = vec3(length(cameraPosition - worldPosition) / 5);
}
