import assert from 'node:assert/strict';
import { test } from 'node:test';
import { auroc, fitLogistic, fitTemperature, logit, reliability, sigmoid } from '../../lib/calibrate';

test('ECE is 0 for perfectly calibrated bins and |1-p| for constant confidence', () => {
  assert.equal(reliability([1, 1, 1, 1], [1, 1, 1, 1]).ece, 0);
  assert.ok(Math.abs(reliability([1, 1, 1, 1], [1, 0, 1, 0]).ece - 0.5) < 1e-9);
});

test('temperature > 1 when logits are overconfident', () => {
  // true p = sigmoid(z/3); give the fitter z, expect T ≈ 3
  const z: number[] = [], y: number[] = [];
  let s = 7;
  const rand = () => ((s = (s * 16807) % 2147483647) / 2147483647);
  for (let i = 0; i < 4000; i++) {
    const zi = (rand() - 0.5) * 16;
    z.push(zi);
    y.push(rand() < sigmoid(zi / 3) ? 1 : 0);
  }
  const T = fitTemperature(z, y);
  assert.ok(T > 2.4 && T < 3.6, `T=${T}`);
});

test('logistic regression separates a linearly separable set', () => {
  const X = [[0], [0.1], [0.2], [0.8], [0.9], [1]];
  const y = [0, 0, 0, 1, 1, 1];
  const m = fitLogistic(X, y);
  assert.equal(auroc(X.map((x) => logit(m, x)), y), 1);
});
