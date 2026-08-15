import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { Tensor } from '../src/core/Tensor';
import { SGD, Adam } from '../src/training/Optimizer';

const close = (a: number, b: number, eps = 1e-4) =>
  assert.ok(Math.abs(a - b) <= eps, `expected ${b}, got ${a}`);

function tensorWithGrad(values: number[], grads: number[]): Tensor {
  const t = new Tensor(values, [values.length], true);
  t.grad = new Tensor(grads, [values.length], false);
  return t;
}

describe('SGD', () => {
  it('updates parameters in the negative gradient direction', () => {
    const param = tensorWithGrad([1, 1], [0.1, -0.2]);
    const sgd = new SGD(0.1, 0);
    sgd.step([param]);
    close(param.data[0], 1 - 0.1 * 0.1);
    close(param.data[1], 1 - 0.1 * (-0.2));
  });

  it('applies momentum', () => {
    const param = tensorWithGrad([1, 1], [0.1, -0.2]);
    const sgd = new SGD(0.1, 0.9);
    sgd.step([param]);
    // First momentum step: v = -lr*grad, param += v
    close(param.data[0], 1 - 0.1 * 0.1);
    close(param.data[1], 1 + 0.1 * 0.2);

    // Second step: v = 0.9*v - lr*grad
    param.grad = new Tensor([0.5, 0.5], [2], false);
    sgd.step([param]);
    const v0 = 0.9 * (-0.01) - 0.1 * 0.5;
    close(param.data[0], (1 - 0.01) + v0);
  });

  it('skips parameters without gradients', () => {
    const param = new Tensor([1], [1], true);
    const sgd = new SGD(0.1, 0);
    sgd.step([param]);
    close(param.data[0], 1); // unchanged, no grad
  });
});

describe('Adam', () => {
  it('updates parameters using bias-corrected moments', () => {
    const param = tensorWithGrad([1, 1], [0.1, -0.2]);
    const adam = new Adam(0.1, 0.9, 0.999, 1e-8);
    adam.step([param]);

    // t=1: mHat = grad, vHat = grad^2 -> update ~ lr * sign(grad)
    close(param.data[0], 1 - 0.1, 1e-3);
    close(param.data[1], 1 + 0.1, 1e-3);
  });

  it('accumulates moments across steps', () => {
    const param = tensorWithGrad([1], [0.1]);
    const adam = new Adam(0.01, 0.9, 0.999, 1e-8);
    const first = param.data[0];
    adam.step([param]);
    const second = param.data[0];
    adam.step([param]);
    assert.ok(second < first, 'parameter should keep decreasing');
  });

  it('handles zero gradients without dividing by zero', () => {
    const param = tensorWithGrad([1], [0]);
    const adam = new Adam(0.1, 0.9, 0.999, 1e-8);
    adam.step([param]);
    close(param.data[0], 1, 1e-6); // nearly unchanged, no NaN
    assert.ok(!Number.isNaN(param.data[0]));
  });
});