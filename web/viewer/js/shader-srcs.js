var linearTosrgb =
`
#line 22
float linear_to_srgb(float x) {
	if (x <= 0.0031308) {
		return 12.92 * x;
	}
	return 1.055 * pow(x, 1.0/2.4) - 0.055;
}
`

var vertShader =
`#version 300 es
#line 15
layout(location = 0) in vec3 pos;
layout(location = 1) in float v_attrib;

uniform mat4 proj_view;
uniform float radius_scale;

flat out float attrib;

#line 23
void main(void) {
	gl_Position = proj_view * vec4(pos, 1.0);
    gl_PointSize = radius_scale;
    attrib = v_attrib;
}`

var fragShader =
`#version 300 es
#line 33
precision highp int;
precision highp float;\n
#define M_PI 3.1415926535897932384626433832795\n

layout(location = 0) out highp vec4 color;

uniform highp sampler2D colormap;
uniform highp vec2 value_range;

flat in float attrib;

void main(void) {
    float t = (attrib - value_range.x) / (value_range.y - value_range.x);
    color = vec4(texture(colormap, vec2(t, 0.5)).rgb, 1.0);
}`;

