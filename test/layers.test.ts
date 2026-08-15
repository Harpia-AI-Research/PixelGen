import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { Tensor } from '../src/core/Tensor';
import { Conv2D } from '../src/layers/Conv2D';
import { Dense } from '../src/layers/Dense';
import { MaxPool2D, AvgPool2D, Upsample2D } from '../src/layers/Pooling';
import { mseLoss, mseLossGrad } from '../src/training/Loss';

const close = (actual: number, expected: number, eps = 1e-3) =>
  assert.ok(
    Math.abs(actual - expected) <= eps,
    `expected ${expected}, got ${actual}`
  );

/**
 * Numeric gradient of a scalar loss w.r.t. a single parameter tensor,
 * computed via central finite differences.
 */
function numericParamGrad(
  lossFn: () => number,
  param: Tensor
): Float32Array {
  const eps = 1e-4;
  const grad = new Float32Array(param.data.length);
  for (let i = 0; i < param.data.length; i++) {
    const orig = param.data[i];
    param.data[i] = orig + eps;
    const plus = lossFn();
    param.data[i] = orig - eps;
    const minus = lossFn();
    param.data[i] = orig;
    grad[i] = (plus - minus) / (2 * eps);
  }
  return grad;
}

/**
 * Verify analytic gradients match numeric gradients for a layer under test.
 * Each run creates a fresh layer + input, runs forward to a scalar loss,
 * then compares backward gradients against finite differences.
 */
function checkLayerGradients(
  params: Tensor[],
  numericLossFns: (() => number)[],
  tol = 5e-2
): void {
  assert.equal(params.length, numericLossFns.length);
  for (let i = 0; i < params.length; i++) {
    const param = params[i];
    const numeric = numericParamGrad(numericLossFns[i], param);
    const analytic = param.grad;
    assert.ok(analytic, 'expected analytic gradient to be set');
    assert.equal(numeric.length, analytic.data.length, 'gradient length mismatch');
    for (let j = 0; j < numeric.length; j++) {
      const denom = Math.max(1, Math.abs(numeric[j]), Math.abs(analytic.data[j]));
      close(Math.abs(numeric[j] - analytic.data[j]) / denom, 0, tol);
    }
  }
}

describe('Conv2D', () => {
  it('computes the correct output shape', () => {
    const conv = new Conv2D(1, 2, 2, 1, 0); // no padding
    const input = new Tensor(new Float32Array(3 * 3), [1, 1, 3, 3]);
    const out = conv.forward(input);
    assert.deepEqual(out.shape, [1, 2, 2, 2]);
  });

  it('computes bias gradient as sum of output gradients', () => {
    const conv = new Conv2D(1, 1, 1, 1, 0);
    const input = new Tensor([1, 2, 3, 4], [1, 1, 2, 2], true);
    const out = conv.forward(input);
    out.grad = Tensor.ones([1, 1, 2, 2], false);
    out.backward();
    // bias gradient = sum of output grads = 4
    close(conv.bias.grad!.data[0], 4);
  });

  it('matches numerical gradients for filters and bias', () => {
    const conv = new Conv2D(1, 2, 2, 1, 0);
    const input = new Tensor(
      [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
      [1, 1, 3, 3],
      true
    );
    const target = new Tensor(new Float32Array(8), [1, 2, 2, 2], false);

    const params = conv.parameters();
    const lossFns = params.map(() => () => {
      const out = conv.forward(input);
      return mseLoss(out, target);
    });

    const out = conv.forward(input);
    const loss = mseLoss(out, target);
    out.grad = mseLossGrad(out, target);
    out.backward();

    assert.ok(loss > 0);
    checkLayerGradients(params, lossFns);
  });
});

describe('Dense', () => {
  it('computes the correct output shape', () => {
    const dense = new Dense(3, 2);
    const input = new Tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [2, 3]);
    const out = dense.forward(input);
    assert.deepEqual(out.shape, [2, 2]);
  });

  it('matches numerical gradients', () => {
    const dense = new Dense(3, 2);
    const input = new Tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [2, 3], true);
    const target = new Tensor(new Float32Array(4), [2, 2], false);

    const params = dense.parameters();
    const lossFns = params.map(() => () => {
      const out = dense.forward(input);
      return mseLoss(out, target);
    });

    const out = dense.forward(input);
    const loss = mseLoss(out, target);
    out.grad = require('../src/training/Loss').mseLossGrad(out, target);
    out.backward();

    assert.ok(loss > 0);
    checkLayerGradients(params, lossFns);
  });
});

describe('MaxPool2D', () => {
  it('downsamples and routes gradient to argmax location', () => {
    const pool = new MaxPool2D(2, 2);
    const input = new Tensor(
      [1, 3, 5, 7], // 2x2, max is 7 at index 3
      [1, 1, 2, 2],
      true
    );
    const out = pool.forward(input);
    assert.deepEqual(out.shape, [1, 1, 1, 1]);
    close(out.data[0], 7);

    out.grad = new Tensor([3], [1, 1, 1, 1], false);
    out.backward();

    assert.equal(input.grad!.data[0], 0);
    assert.equal(input.grad!.data[3], 3);
  });
});

describe('AvgPool2D', () => {
  it('averages and distributes gradient evenly', () => {
    const pool = new AvgPool2D(2, 2);
    const input = new Tensor([1, 3, 5, 7], [1, 1, 2, 2], true);
    const out = pool.forward(input);
    close(out.data[0], 4);

    out.grad = new Tensor([4], [1, 1, 1, 1], false);
    out.backward();
    for (const v of input.grad!.data) {
      close(v, 1);
    }
  });
});

describe('Upsample2D', () => {
  it('upsamples with nearest neighbor and routes gradients back', () => {
    const up = new Upsample2D(2);
    const input = new Tensor([1, 2, 3, 4], [1, 1, 2, 2], true);
    const out = up.forward(input);
    assert.deepEqual(out.shape, [1, 1, 4, 4]);

    out.grad = Tensor.ones([1, 1, 4, 4], false);
    out.backward();
    // Each source pixel receives gradient from its 2x2 block -> sum of 4 ones
    for (const v of input.grad!.data) {
      close(v, 4);
    }
  });
});