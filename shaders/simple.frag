#version 450
#
layout(set = 1, binding = 1) uniform sampler2D textureSampler;
layout(set = 1, binding = 2) uniform MeshUniformFrag {
    vec4 tint;
} instance;

layout(location = 1) in vec2 uv;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = instance.tint * texture(textureSampler, uv);
    if (outColor.a == 0) {
        discard;
    }
}
