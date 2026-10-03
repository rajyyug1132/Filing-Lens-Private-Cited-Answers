export function packVectorsNode(vs: Float32Array[]): { buffer: Buffer; dim: number } {
  const dim = vs[0]?.length ?? 384;
  const flat = new Float32Array(vs.length * dim);
  vs.forEach((v, i) => flat.set(v, i * dim));
  return { buffer: Buffer.from(flat.buffer), dim };
}
