import { Tensor } from '../core/Tensor';
import { Backend } from '../core/backend';
import { WasmBackend } from '../core/backend/WasmBackend';

export class Dense {
  public weights: Tensor;
  public bias: Tensor;
  private inFeatures: number;
  private outFeatures: number;
  private backend?: Backend;

  constructor(inFeatures: number, outFeatures: number, backend?: Backend) {
    this.inFeatures = inFeatures;
    this.outFeatures = outFeatures;
    this.backend = backend;

    this.weights = Tensor.xavier([inFeatures, outFeatures], true);
    this.bias = Tensor.zeros([outFeatures], true);
  }

  async forward(input: Tensor): Promise<Tensor> {
    if (input.shape.length !== 2) {
      throw new Error('Dense layer expects 2D input [batch, features]');
    }

    const [batch, inFeatures] = input.shape;

    if (inFeatures !== this.inFeatures) {
      throw new Error(`Input features mismatch: expected ${this.inFeatures}, got ${inFeatures}`);
    }

    const useBackend = this.backend || Tensor.backend;

    let output: Tensor;

    if (useBackend) {
      const result = await useBackend.matmul(input, this.weights);
      output = result.output;
    } else {
      output = this.computeForward(input, batch);
    }

    output._prev.add(input);
    output._prev.add(this.weights);
    output._prev.add(this.bias);

    output._backward = async () => {
      if (!output.grad) return;

      if (!this.weights.grad) this.weights.grad = Tensor.zerosLike(this.weights);
      if (!this.bias.grad) this.bias.grad = Tensor.zerosLike(this.bias);
      if (input.requiresGrad && !input.grad) input.grad = Tensor.zerosLike(input);

      const runBackend = async (backend: Backend) => {
        const backwardResult = await backend.matmul(input.transpose(), output.grad!);
        this.weights.grad = backwardResult.output;

        const biasGrad = new Float32Array(this.outFeatures);
        for (let i = 0; i < output.grad!.data.length; i++) {
          const b = Math.floor(i / this.outFeatures);
          const f = i % this.outFeatures;
          biasGrad[f] += output.grad!.data[i];
        }
        this.bias.grad = new Tensor(biasGrad, [this.outFeatures], false);

        if (input.requiresGrad) {
          const inputGradResult = await backend.matmul(output.grad!, this.weights.transpose());
          input.grad = inputGradResult.output;
        }
      };

      if (useBackend) {
        if (useBackend instanceof WasmBackend) {
          await runBackend(useBackend);
        } else {
          await useBackend.then(runBackend);
        }
      } else {
        this.computeBackward(input, output, batch);
      }
    };

    return output;
  }

  private computeForward(input: Tensor, batch: number): Tensor {
    const resultData = new Float32Array(batch * this.outFeatures);

    for (let i = 0; i < batch; i++) {
      for (let j = 0; j < this.outFeatures; j++) {
        let sum = 0;
        for (let k = 0; k < this.inFeatures; k++) {
          sum += input.data[i * this.inFeatures + k] * this.weights.data[k * this.outFeatures + j];
        }
        resultData[i * this.outFeatures + j] = sum + this.bias.data[j];
      }
    }

    return new Tensor(resultData, [batch, this.outFeatures], true);
  }

  private computeBackward(input: Tensor, output: Tensor, batch: number): void {
    for (let i = 0; i < batch; i++) {
      for (let j = 0; j < this.outFeatures; j++) {
        const g = output.grad!.data[i * this.outFeatures + j];

        this.bias.grad!.data[j] += g;

        for (let k = 0; k < this.inFeatures; k++) {
          const wIdx = k * this.outFeatures + j;
          this.weights.grad!.data[wIdx] += g * input.data[i * this.inFeatures + k];

          if (input.requiresGrad) {
            input.grad!.data[i * this.inFeatures + k] += g * this.weights.data[wIdx];
          }
        }
      }
    }
  }

  parameters(): Tensor[] {
    return [this.weights, this.bias];
  }

  zeroGrad(): void {
    this.weights.zeroGrad();
    this.bias.zeroGrad();
  }
}
