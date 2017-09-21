// This shader computes per-pixel depth (-z coordinate in the camera space, or
// orthogonal distance to the camera plane). The result is multiplied by the
// `kFixedPointFraction` constant and is encoded to RGB channels as an integer
// (R being the least significant byte).

#ifdef GL_ES
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
#endif

const float kFixedPointFraction = 1000.0;

varying float vDepth;

void main(void) {
  float d = vDepth;

  // Encode the depth to RGB.
  d *= (kFixedPointFraction / 255.0);
  gl_FragColor.r = mod(d, 1.0);
  d = (d - gl_FragColor.r) / 255.0;
  gl_FragColor.g = mod(d, 1.0);
  d = (d - gl_FragColor.g) / 255.0;
  gl_FragColor.b = mod(d, 1.0);

  gl_FragColor.a = 1.0;
}
