import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { PixelGenModel } from '../src/model/PixelGenModel';
import { Trainer, TrainingConfig } from '../src/training/Trainer';
import { createTestDataset, ImageDataset } from '../src/data/ImageLoader';
import { Tensor } from '../src/core/Tensor';

function makeDataset(size = 4, side = 8): ImageDataset {
  return createTestDataset(size, 3, side, side);
}

function makeConfig(partial?: Partial<TrainingConfig>): TrainingConfig {
  return {
    epochs: 2,
    batchSize: 2,
    learningRate: 0.01,
    optimizer: 'adam',
    lossFunction: 'mse',
    verbose: false,
    ...partial,
  };
}

function makeInput(batch = 1, side = 8): Tensor {
  const input = new Tensor(new Float32Array(batch * 3 * side * side), [
    batch,
    3,
    side,
    side,
  ]);
  for (let i = 0; i < input.data.length; i++) {
    input.data[i] = (Math.sin(i) * 0.5 + 0.5) / 255;
  }
  return input;
}

describe('PixelGenModel', () => {
  it('forward output matches input shape', () => {
    const model = new PixelGenModel(3, 3);
    const out = model.forward(makeInput(2));
    assert.deepEqual(out.shape, [2, 3, 8, 8]);
  });

  it('backward populates gradients for all parameters', () => {
    const model = new PixelGenModel(3, 3);
    const out = model.forward(makeInput(1));
    out.grad = Tensor.ones(out.shape, false);
    out.backward();

    const params = model.parameters();
    assert.equal(params.length, 12);
    for (const p of params) {
      assert.ok(p.grad !== undefined, 'expected a gradient on every parameter');
      assert.ok(
        p.grad.data.every(v => Number.isFinite(v)),
        'gradients must be finite'
      );
      assert.ok(
        p.grad.data.some(v => v !== 0),
        'gradients must be non-zero somewhere'
      );
    }
  });

  it('zeroGrad clears acquired gradients', () => {
    const model = new PixelGenModel(3, 3);
    const out = model.forward(makeInput(1));
    out.grad = Tensor.ones(out.shape, false);
    out.backward();
    model.zeroGrad();
    for (const p of model.parameters()) {
      assert.equal(p.grad!.data[0], 0);
    }
  });
});

describe('Trainer', () => {
  it('runs with the SGD optimizer', () => {
    const model = new PixelGenModel(3, 3);
    const trainer = new Trainer(model, makeConfig({ optimizer: 'sgd' }));
    const stats = trainer.train(makeDataset());
    assert.equal(stats.length, 2);
    assert.ok(Number.isFinite(stats[0].loss));
  });

  it('runs with the Adam optimizer', () => {
    const model = new PixelGenModel(3, 3);
    const trainer = new Trainer(model, makeConfig({ optimizer: 'adam' }));
    const stats = trainer.train(makeDataset());
    assert.equal(stats.length, 2);
    assert.ok(Number.isFinite(stats[0].loss));
  });

  it('supports every loss function', () => {
    for (const lossFunction of ['mse', 'mae', 'bce', 'pixelart'] as const) {
      const model = new PixelGenModel(3, 3);
      const trainer = new Trainer(
        model,
        makeConfig({ lossFunction, optimizer: 'adam', epochs: 1 })
      );
      const stats = trainer.train(makeDataset(4));
      assert.equal(stats.length, 1);
      assert.ok(Number.isFinite(stats[0].loss), `${lossFunction} loss must be finite`);
    }
  });

  it('calls epoch callbacks', () => {
    const model = new PixelGenModel(3, 3);
    const trainer = new Trainer(model, makeConfig({ epochs: 2 }));
    let calls = 0;
    trainer.train(makeDataset(), {
      onEpoch: () => {
        calls++;
      },
    });
    assert.equal(calls, 2);
  });

  it('evaluates loss on a dataset', () => {
    const model = new PixelGenModel(3, 3);
    const trainer = new Trainer(model, makeConfig({ epochs: 0 }));
    const loss = trainer.evaluate(makeDataset());
    assert.ok(Number.isFinite(loss));
  });

  it('records training stats', () => {
    const model = new PixelGenModel(3, 3);
    const trainer = new Trainer(model, makeConfig({ epochs: 3 }));
    trainer.train(makeDataset());
    const stats = trainer.getStats();
    assert.equal(stats.length, 3);
    for (const s of stats) {
      assert.ok(s.epoch >= 1);
      assert.ok(s.time >= 0);
    }
  });
});