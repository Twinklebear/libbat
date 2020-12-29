#version 450 core

uniform mat4 proj_view;
uniform vec3 fcolor;
uniform vec2 range;
layout(binding = 0) uniform sampler1D colormap;

layout(location = 0) in vec3 pos;
layout(location = 1) in double attrib;

flat out vec3 fcol;

void main(void)
{
    float x = (float(attrib) - range.x) / (range.y - range.x);
    vec4 c = texture(colormap, x);
    // If it's culled by the transfer function move it outside
    // the NDC to discard
    if (c.a < 0.02) {
        gl_Position = vec4(0, 0, -2, 1);
    } else {
        gl_Position = proj_view * vec4(pos, 1.f);
    }
    fcol = c.rgb;
}

