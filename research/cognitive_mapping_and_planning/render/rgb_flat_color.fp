precision highp float;
varying vec4 vColor;
varying vec2 vTextureCoord;

uniform sampler2D uTexture;

void main(void) {
  vec4 color = vColor;
  color = texture2D(uTexture, vTextureCoord);
  gl_FragColor = color;
}
