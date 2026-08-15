import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { Tensor } from '../src/core/Tensor';
import {
  mseLoss,
  mseLossGrad,
  maeLoss,
  maeLossGrad,
  bceLoss,
  bceLossGrad,
  pixelArtLoss,
  pixelArtLossGrad,
} from '../src/training/Loss';

const close = (a: number, b: number, eps = 1e-4) =>
  assert.ok(Math.abs(a - b) <= eps, `expected ${b}, got ${a}`);

function numericGrad(lossFn: (p: Float32Array) => number, pred: Tensor): Float32Array {
  // eps=1e-3: large enough to avoid float32 rounding dominating the
  // finite-difference estimate, small enough that truncation error is negligible.
  const eps = 1e-3;
  const grad = new Float32Array(pred.data.length);
  for (let i = 0; i < pred.data.length; i++) {
    const orig = pred.data[i];
    pred.data[i] = orig + eps;
    const plus = lossFn(pred.data);
    pred.data[i] = orig - eps;
    const minus = lossFn(pred.data);
    pred.data[i] = orig;
    grad[i] = (plus - minus) / (2 * eps);
  }
  return grad;
}

function compareGrads(analytic: Tensor, numeric: Float32Array, tol = 1e-3): void {
  assert.equal(analytic.data.length, numeric.length);
  for (let i = 0; i < numeric.length; i++) {
    const denom = Math.max(1, Math.abs(numeric[i]), Math.abs(analytic.data[i]));
    close(Math.abs(analytic.data[i] - numeric[i]) / denom, 0, tol);
  }
}

describe('mseLoss', () => {
  it('computes the loss', () => {
    const p = new Tensor([0, 0], [2]);
    const t = new Tensor([1, 2], [2]);
    close(mseLoss(p, t), (1 + 4) / 2);
  });

  it('gradient is 2(p - t) / N', () => {
    const p = new Tensor([0.1, 0.5, 0.3], [3]);
    const t = new Tensor([0.2, 0.4, 0.6], [3]);
    const g = mseLossGrad(p, t);
    close(g.data[0], 2 * (0.1 - 0.2) / 3);
    close(g.data[1], 2 * (0.5 - 0.4) / 3);
  });

  it('throws on shape mismatch', () => {
    assert.throws(() => mseLoss(new Tensor([1], [1]), new Tensor([1, 2], [2])));
  });
});

describe('maeLoss', () => {
  it('computes the loss', () => {
    const p = new Tensor([0, 3], [2]);
    const t = new Tensor([2, 5], [2]);
    close(maeLoss(p, t), (2 + 2) / 2);
  });

  it('gradient matches numeric subgradient away from zero', () => {
    const p = new Tensor([0.1, 0.5, 0.9], [3]);
    const t = new Tensor([0.3, 0.2, 0.4], [3]);
    const g = maeLossGrad(p, t);
    const numeric = numericGrad(v => maeLoss(new Tensor(v, [3]), t), p);
    compareGrads(g, numeric);
    // sign checks: p>t -> 1/N ; p<t -> -1/N
    close(g.data[0], -1 / 3);
    close(g.data[1], 1 / 3);
  });
});

describe('bceLoss', () => {
  it('computes the loss', () => {
    const p = new Tensor([0.5, 0.5], [2]);
    const t = new Tensor([1, 0], [2]);
    const expected = -1 / 2 * (Math.log(0.5) + Math.log(0.5));
    close(bceLoss(p, t), expected);
  });

  it('gradient matches numeric gradient', () => {
    const p = new Tensor([0.3, 0.6, 0.8], [3]);
    const t = new Tensor([1, 0, 1], [3]);
    const g = bceLossGrad(p, t);
    const numeric = numericGrad(v => bceLoss(new Tensor(v, [3]), t), p);
    compareGrads(g, numeric);
  });
});

describe('pixelArtLoss', () => {
  const pred = new Tensor(
    [0.1, 0.5, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6],
    [1, 1, 2, 4]
  );
  const target = new Tensor(
    [0.05, 0.2, 0.1, 0.35, 0.15, 0.45, 0.25, 0.5],
    [1, 1, 2, 4]
  );

  it('combines MSE with edge preservation', () => {
    const mse = mseLoss(pred, target);
    const total = pixelArtLoss(pred, target);
    assert.ok(total >= mse);
  });

  it('gradient matches numeric gradient', () => {
    const g = pixelArtLossGrad(pred, target);
    const numeric = numericGrad(
      v => pixelArtLoss(new Tensor(v, [1, 1, 2, 4]), target),
      pred
    );
    compareGrads(g, numeric, 1e-2);
  });
});