import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { Tensor } from '../src/core/Tensor';
import { softmax, relu } from '../src/core/activations';

const close = (actual: number, expected: number, eps = 1e-4) =>
  assert.ok(
    Math.abs(actual - expected) <= eps,
    `expected ${expected}, got ${actual}`
  );

describe('Tensor basics', () => {
  it('creates from flat array and shape', () => {
    const t = new Tensor([1, 2, 3, 4], [2, 2]);
    assert.deepEqual(t.shape, [2, 2]);
    assert.equal(t.data.length, 4);
  });

  it('creates from nested array and infers shape', () => {
    const t = new Tensor([[1, 2], [3, 4]]);
    assert.deepEqual(t.shape, [2, 2]);
  });

  it('validates shape/data length mismatch', () => {
    assert.throws(() => new Tensor([1, 2, 3], [2, 2]));
  });

  it('supports get/set via multi-indices', () => {
    const t = new Tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    assert.equal(t.get(1, 2), 6);
    t.set(99, 0, 0);
    assert.equal(t.get(0, 0), 99);
  });

  it('reshapes', () => {
    const t = new Tensor([1, 2, 3, 4, 5, 6], [2, 3]);
    const r = t.reshape([3, 2]);
    assert.deepEqual(r.shape, [3, 2]);
    assert.throws(() => t.reshape([4, 4]));
  });

  it('transposes 2D tensors', () => {
    const t = new Tensor([[1, 2, 3], [4, 5, 6]]);
    const tT = t.transpose();
    assert.deepEqual(tT.shape, [3, 2]);
    assert.deepEqual(Array.from(tT.data), [1, 4, 2, 5, 3, 6]);
    assert.throws(() => new Tensor([1, 2, 3], [3]).transpose());
  });

  it('provides zeros/ones/random factories', () => {
    assert.deepEqual(Array.from(Tensor.zeros([2, 2]).data), [0, 0, 0, 0]);
    assert.deepEqual(Array.from(Tensor.ones([3]).data), [1, 1, 1]);
    const r = Tensor.random([5]);
    assert.ok(r.data.every(v => v >= -1 && v <= 1));
  });
});

describe('Tensor autograd', () => {
  it('propagates gradients through add and multiply', () => {
    const x = new Tensor([1, 2, 3], [3], true);
    const y = x.add(new Tensor([1, 1, 1], [3], false)).multiply(2);
    y.backward();
    // dy/dx = 2 everywhere
    assert.deepEqual(Array.from(x.grad!.data), [2, 2, 2]);
  });

  it('propagates gradients through matmul', () => {
    const a = new Tensor([[1, 2, 3], [4, 5, 6]], undefined, true);
    const b = new Tensor([[1, 2], [3, 4], [5, 6]], undefined, true);
    const y = a.matmul(b);
    y.backward();

    // A.grad[i][k] = sum_j grad[j] * B[k][j]  -> B[k][0]+B[k][1]
    assert.deepEqual(Array.from(a.grad!.data), [3, 7, 11, 3, 7, 11]);
    // B.grad[k][j] = sum_i A[i][k]             -> A[0][k]+A[1][k]
    assert.deepEqual(Array.from(b.grad!.data), [5, 5, 7, 7, 9, 9]);
  });

  it('handles scalar multiply gradients', () => {
    const x = new Tensor([2, 4], [2], true);
    const y = x.multiply(3);
    y.backward();
    assert.deepEqual(Array.from(x.grad!.data), [3, 3]);
  });

  it('zeroes gradients', () => {
    const x = new Tensor([2], [1], true);
    x.multiply(3).backward();
    assert.equal(x.grad!.data[0], 3);
    x.zeroGrad();
    assert.equal(x.grad!.data[0], 0);
  });

  it('throws when backward is called on a non-grad tensor', () => {
    const x = new Tensor([2], [1], false);
    assert.throws(() => x.backward());
  });

  it('supports cloned tensors with independent data', () => {
    const t = new Tensor([1, 2], [2]);
    const c = t.clone();
    c.data[0] = 99;
    assert.equal(t.data[0], 1);
  });

  it('matches analytic gradient numerically (x^2 sum)', () => {
    const x = new Tensor([0.5, 1.5, 2.5], [3], true);
    const y = x.multiply(x);
    const loss = y.add(new Tensor([0, 0, 0], [3], false));
    loss.backward();

    // grad of sum(x^2) = 2x
    for (let i = 0; i < 3; i++) {
      close(x.grad!.data[i], 2 * x.data[i]);
    }
  });

  it('adds a broadcast bias across the batch dimension', () => {
    const a = new Tensor([[1, 2], [3, 4]], undefined, true);
    const bias = new Tensor([10, 20], [2], true);
    const y = a.add(bias);
    assert.deepEqual(Array.from(y.data), [11, 22, 13, 24]);
    y.backward();
    // bias grads accumulate across the batch
    assert.deepEqual(Array.from(bias.grad!.data), [2, 2]);
    // a grads are ones (dL/dA = 1)
    assert.deepEqual(Array.from(a.grad!.data), [1, 1, 1, 1]);
  });

  it('rejects non-broadcastable shapes in add', () => {
    const a = new Tensor([1, 2], [2]);
    assert.throws(() => a.add(new Tensor([1, 2, 3], [3])));
    assert.throws(() => a.add(new Tensor([[1], [2]], [2, 1])));
  });
});

describe('Activations', () => {
  it('relu zeroes negatives and keeps positives', () => {
    const x = new Tensor([-1, 0.5, 0], [3], true);
    const r = relu(x);
    assert.deepEqual(Array.from(r.data), [0, 0.5, 0]);
    r.backward();
    assert.deepEqual(Array.from(x.grad!.data), [0, 1, 0]);
  });

  it('softmax normalizes each row across the last dim', () => {
    const x = new Tensor([0, 0, 0, 1, 2, 3], [2, 3], true);
    const s = softmax(x);
    const row0 = Array.from(s.data.slice(0, 3));
    const row1 = Array.from(s.data.slice(3, 6));
    for (const v of row0) {
      close(v, 1 / 3);
    }
    close(row0.reduce((a, b) => a + b, 0), 1);
    close(row1.reduce((a, b) => a + b, 0), 1);
    assert.ok(row1[2] > row1[1] && row1[1] > row1[0]);
  });

  it('softmax backward matches numeric gradient', () => {
    const x = new Tensor([0.5, -0.2, 1.1, 2, -1, 0.3], [2, 3], true);
    const w = new Tensor([1, 2, -1, 0.5, 1, 1.5], [2, 3], false);

    const y = softmax(x);
    y.grad = w;
    y.backward();

    // eps=1e-3 avoids float32 rounding dominating the finite differences
    const eps = 1e-3;
    const numeric = new Float32Array(x.data.length);
    for (let i = 0; i < x.data.length; i++) {
      const orig = x.data[i];
      x.data[i] = orig + eps;
      const ap = softmax(x);
      let lp = 0;
      for (let j = 0; j < ap.data.length; j++) lp += ap.data[j] * w.data[j];
      x.data[i] = orig - eps;
      const am = softmax(x);
      let lm = 0;
      for (let j = 0; j < am.data.length; j++) lm += am.data[j] * w.data[j];
      x.data[i] = orig;
      numeric[i] = (lp - lm) / (2 * eps);
    }

    for (let i = 0; i < numeric.length; i++) {
      const denom = Math.max(1, Math.abs(numeric[i]), Math.abs(x.grad!.data[i]));
      close(Math.abs(x.grad!.data[i] - numeric[i]) / denom, 0, 1e-3);
    }
  });
});