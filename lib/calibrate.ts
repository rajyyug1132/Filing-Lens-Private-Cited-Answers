// Logistic regression + temperature scaling + calibration metrics.
// Training runs in the bench (Node); inference runs in the browser from the
// exported JSON. No dependencies, so both sides use the same arithmetic.

export interface DecisionModel {
  featureNames: string[];
  k: number; // retrieval depth the gate was trained with; the app must use the same
  tokenBudget: number; // max prompt tokens (system + question + chunks)
  tokPerChar: number; // conservative tokens-per-character estimate: the max over bench prompts + 0.005
  mean: number[];
  std: number[];
  weights: number[];
  bias: number;
  temperature: number;
  threshold: number;
  trainedOn: string;
}

export const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

export function standardize(X: number[][]): { mean: number[]; std: number[] } {
  const d = X[0].length;
  const mean = Array(d).fill(0);
  const std = Array(d).fill(0);
  for (const x of X) x.forEach((v, j) => (mean[j] += v / X.length));
  for (const x of X) x.forEach((v, j) => (std[j] += (v - mean[j]) ** 2 / X.length));
  return { mean, std: std.map((s) => Math.sqrt(s) || 1) };
}

export function logit(m: Pick<DecisionModel, 'mean' | 'std' | 'weights' | 'bias'>, x: number[]): number {
  let z = m.bias;
  for (let j = 0; j < x.length; j++) z += m.weights[j] * ((x[j] - m.mean[j]) / m.std[j]);
  return z;
}

// x may carry more features than the model uses (feature sets are prefixes).
export function predict(m: DecisionModel, x: number[]): number {
  return sigmoid(logit(m, x.slice(0, m.weights.length)) / m.temperature);
}

// Class weights that make the effective prior 50/50 (training sets can be 1:2).
export function balancedWeights(y: number[]): number[] {
  const pos = y.filter((v) => v === 1).length;
  const neg = y.length - pos;
  return y.map((v) => (v === 1 ? y.length / (2 * pos) : y.length / (2 * neg)));
}

// Full-batch gradient descent with L2 and optional sample weights.
export function fitLogistic(X: number[][], y: number[], l2 = 1e-2, iters = 3000, lr = 0.1, sw: number[] = y.map(() => 1)) {
  const { mean, std } = standardize(X);
  const Z = X.map((x) => x.map((v, j) => (v - mean[j]) / std[j]));
  const d = Z[0].length;
  const w = Array(d).fill(0);
  let b = 0;
  for (let it = 0; it < iters; it++) {
    const gw = Array(d).fill(0);
    let gb = 0;
    for (let i = 0; i < Z.length; i++) {
      let z = b;
      for (let j = 0; j < d; j++) z += w[j] * Z[i][j];
      const e = (sigmoid(z) - y[i]) * sw[i];
      for (let j = 0; j < d; j++) gw[j] += e * Z[i][j];
      gb += e;
    }
    const W = sw.reduce((a, v) => a + v, 0);
    for (let j = 0; j < d; j++) w[j] -= lr * (gw[j] / W + l2 * w[j]);
    b -= lr * (gb / W);
  }
  return { mean, std, weights: w, bias: b };
}

export function nll(p: number[], y: number[], sw: number[] = y.map(() => 1)): number {
  const eps = 1e-12;
  const W = sw.reduce((a, v) => a + v, 0);
  return -y.reduce((s, yi, i) => s + sw[i] * (yi * Math.log(p[i] + eps) + (1 - yi) * Math.log(1 - p[i] + eps)), 0) / W;
}

// Temperature scaling (Guo et al., 2017): one scalar T fitted on held-out
// logits by minimising NLL. Grid search is enough for one parameter.
export function fitTemperature(logits: number[], y: number[], sw: number[] = y.map(() => 1)): number {
  let best = 1;
  let bestLoss = Infinity;
  for (let T = 0.2; T <= 6.0001; T += 0.01) {
    const loss = nll(logits.map((z) => sigmoid(z / T)), y, sw);
    if (loss < bestLoss) {
      bestLoss = loss;
      best = T;
    }
  }
  return Number(best.toFixed(2));
}

export interface Bin {
  lo: number;
  hi: number;
  count: number;
  meanConf: number;
  fracPos: number;
}

// ECE over p(supported), 10 equal-width bins: sum_b |B|/N * |acc(B) - conf(B)|
export function reliability(p: number[], y: number[], nBins = 10): { ece: number; bins: Bin[] } {
  const bins: Bin[] = Array.from({ length: nBins }, (_, i) => ({
    lo: i / nBins,
    hi: (i + 1) / nBins,
    count: 0,
    meanConf: 0,
    fracPos: 0,
  }));
  p.forEach((pi, i) => {
    const b = bins[Math.min(nBins - 1, Math.floor(pi * nBins))];
    b.count++;
    b.meanConf += pi;
    b.fracPos += y[i];
  });
  let ece = 0;
  for (const b of bins) {
    if (!b.count) continue;
    b.meanConf /= b.count;
    b.fracPos /= b.count;
    ece += (b.count / p.length) * Math.abs(b.fracPos - b.meanConf);
  }
  return { ece, bins };
}

export function auroc(p: number[], y: number[]): number {
  const pos = p.filter((_, i) => y[i] === 1);
  const neg = p.filter((_, i) => y[i] === 0);
  if (!pos.length || !neg.length) return NaN;
  let s = 0;
  for (const a of pos) for (const b of neg) s += a > b ? 1 : a === b ? 0.5 : 0;
  return s / (pos.length * neg.length);
}
