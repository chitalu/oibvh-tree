#version 430 core

uniform mat4 modelViewProjection;

layout(location = 0) in vec3 vertexPosition;

void main(void) {
  gl_Position = (modelViewProjection * vec4(vertexPosition, 1.0f));
}
