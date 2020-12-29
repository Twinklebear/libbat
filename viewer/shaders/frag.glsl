#version 450 core

flat in vec3 fcol;

out vec4 color;


float linear_to_srgb(float x)
{
    if (x <= 0.0031308f) {
        return 12.92f * x;
    }
    return 1.055f * pow(x, 1.f / 2.4f) - 0.055f;
}

void main(void)
{
    color = vec4(linear_to_srgb(fcol.x),
                 linear_to_srgb(fcol.y),
                 linear_to_srgb(fcol.z), 1.f);
}

