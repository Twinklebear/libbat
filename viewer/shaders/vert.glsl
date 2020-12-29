#version 450 core

uniform mat4 proj_view;
uniform vec3 fcolor;

layout(location = 0) in vec3 pos;

flat out vec3 fcol;

void main(void)
{
    gl_Position = proj_view * vec4(pos, 1.f);
    fcol = fcolor;
}

