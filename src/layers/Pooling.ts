import { Tensor } from '../core/Tensor';
import { Backend } from '../core/backend';

export class MaxPool2D {
  private poolSize: number;
  private stride: number;
  private backend?: Backend;

  constructor(poolSize: number = 2, stride?: number, backend?: Backend) {
    this.poolSize = poolSize;
    this.stride = stride || poolSize;
    this.backend = backend;
  }

  async forward(input: Tensor): Promise<Tensor> {
    if (input.shape.length !== 4) {
      throw new Error(`MaxPool2D expects 4D input, got ${input.shape}`);
    }

    const useBackend = this.backend || Tensor.backend;

    let result: { output: Tensor; argmax?: Int32Array };

    if (useBackend) {
      result = await useBackend.poolForward(input, this.poolSize, this.stride);
    } else {
      result = this.computeForward(input);
    }

    const outputTensor = result.output;
    const argmax = result.argmax;

    if (input.requiresGrad) {
      outputTensor._prev.add(input);

      outputTensor._backward = () => {
        if (!outputTensor.grad) return;
        if (!input.grad) input.grad = Tensor.zerosLike(input);

        if (argmax) {
          for (let i = 0; i < outputTensor.grad.data.length; i++) {
            input.grad.data[argmax[i]] += outputTensor.grad.data[i];
          }
        }
      };
    }

    return outputTensor;
  }

  private computeForward(input: Tensor): { output: Tensor; argmax: Int32Array } {
    const [batch, channels, height, width] = input.shape;
    const outHeight = Math.floor((height - this.poolSize) / this.stride) + 1;
    const outWidth = Math.floor((width - this.poolSize) / this.stride) + 1;

    const outputSize = batch * channels * outHeight * outWidth;
    const output = new Float32Array(outputSize);
    const argmax = new Int32Array(outputSize);

    for (let b = 0; b < batch; b++) {
      for (let c = 0; c < channels; c++) {
        for (let oh = 0; oh < outHeight; oh++) {
          for (let ow = 0; ow < outWidth; ow++) {
            let max = -Infinity;
            let maxIdx = 0;

            for (let ph = 0; ph < this.poolSize; ph++) {
              for (let pw = 0; pw < this.poolSize; pw++) {
                const ih = oh * this.stride + ph;
                const iw = ow * this.stride + pw;

                const inputIdx = b * channels * height * width + c * height * width + ih * width + iw;

                if (input.data[inputIdx] > max) {
                  max = input.data[inputIdx];
                  maxIdx = inputIdx;
                }
              }
            }

            const outputIdx = b * channels * outHeight * outWidth + c * outHeight * outWidth + oh * outWidth + ow;
            output[outputIdx] = max;
            argmax[outputIdx] = maxIdx;
          }
        }
      }
    }

    return { output: new Tensor(output, [batch, channels, outHeight, outWidth], input.requiresGrad), argmax };
  }

  parameters(): Tensor[] {
    return [];
  }

  zeroGrad(): void {
    // No parameters to zero
  }
}

export class AvgPool2D {
  private poolSize: number;
  private stride: number;
  private backend?: Backend;

  constructor(poolSize: number = 2, stride?: number, backend?: Backend) {
    this.poolSize = poolSize;
    this.stride = stride || poolSize;
    this.backend = backend;
  }

  async forward(input: Tensor): Promise<Tensor> {
    if (input.shape.length !== 4) {
      throw new Error(`AvgPool2D expects 4D input, got ${input.shape}`);
    }

    const useBackend = this.backend || Tensor.backend;
    const output: Tensor = useBackend
      ? (await useBackend.avgPoolForward(input, this.poolSize, this.stride)).output
      : this.computeForward(input);

    if (input.requiresGrad) {
      output._prev.add(input);

      output._backward = () => {
        if (!output.grad) return;
        if (!input.grad) input.grad = Tensor.zerosLike(input);

        const scale = 1 / (this.poolSize * this.poolSize);
        const [batch, channels, height, width] = input.shape;
        const outHeight = Math.floor((height - this.poolSize) / this.stride) + 1;
        const outWidth = Math.floor((width - this.poolSize) / this.stride) + 1;

        for (let b = 0; b < batch; b++) {
          for (let c = 0; c < channels; c++) {
            for (let oh = 0; oh < outHeight; oh++) {
              for (let ow = 0; ow < outWidth; ow++) {
                const outputIdx = b * channels * outHeight * outWidth + c * outHeight * outWidth + oh * outWidth + ow;
                const g = output.grad.data[outputIdx];

                for (let ph = 0; ph < this.poolSize; ph++) {
                  for (let pw = 0; pw < this.poolSize; pw++) {
                    const ih = oh * this.stride + ph;
                    const iw = ow * this.stride + pw;

                    const inputIdx = b * channels * height * width + c * height * width + ih * width + iw;
                    input.grad.data[inputIdx] += g * scale;
                  }
                }
              }
            }
          }
        }
      };
    }

    return output;
  }

  private computeForward(input: Tensor): Tensor {
    const [batch, channels, height, width] = input.shape;
    const outHeight = Math.floor((height - this.poolSize) / this.stride) + 1;
    const outWidth = Math.floor((width - this.poolSize) / this.stride) + 1;

    const outputSize = batch * channels * outHeight * outWidth;
    const output = new Float32Array(outputSize);
    const poolArea = this.poolSize * this.poolSize;

    for (let b = 0; b < batch; b++) {
      for (let c = 0; c < channels; c++) {
        for (let oh = 0; oh < outHeight; oh++) {
          for (let ow = 0; ow < outWidth; ow++) {
            let sum = 0;

            for (let ph = 0; ph < this.poolSize; ph++) {
              for (let pw = 0; pw < this.poolSize; pw++) {
                const ih = oh * this.stride + ph;
                const iw = ow * this.stride + pw;

                const inputIdx = b * channels * height * width + c * height * width + ih * width + iw;
                sum += input.data[inputIdx];
              }
            }

            const outputIdx = b * channels * outHeight * outWidth + c * outHeight * outWidth + oh * outWidth + ow;
            output[outputIdx] = sum / poolArea;
          }
        }
      }
    }

    return new Tensor(output, [batch, channels, outHeight, outWidth], input.requiresGrad);
  }

  parameters(): Tensor[] {
    return [];
  }

  zeroGrad(): void {
    // No parameters
  }
}

export class Upsample2D {
  private scale: number;
  private backend?: Backend;

  constructor(scale: number = 2, backend?: Backend) {
    this.scale = scale;
    this.backend = backend;
  }

  async forward(input: Tensor): Promise<Tensor> {
    if (input.shape.length !== 4) {
      throw new Error(`Upsample2D expects 4D input, got ${input.shape}`);
    }

    const useBackend = this.backend || Tensor.backend;
    const output: Tensor = useBackend
      ? (await useBackend.upsampleForward(input, this.scale)).output
      : this.computeForward(input);

    if (input.requiresGrad) {
      output._prev.add(input);

      output._backward = () => {
        if (!output.grad) return;
        if (!input.grad) input.grad = Tensor.zerosLike(input);

        const [batch, channels, height, width] = input.shape;
        const outHeight = height * this.scale;
        const outWidth = width * this.scale;

        for (let b = 0; b < batch; b++) {
          for (let c = 0; c < channels; c++) {
            for (let oh = 0; oh < outHeight; oh++) {
              for (let ow = 0; ow < outWidth; ow++) {
                const ih = Math.floor(oh / this.scale);
                const iw = Math.floor(ow / this.scale);

                const inputIdx = b * channels * height * width + c * height * width + ih * width + iw;
                const outputIdx = b * channels * outHeight * outWidth + c * outHeight * outWidth + oh * outWidth + ow;

                input.grad.data[inputIdx] += output.grad.data[outputIdx];
              }
            }
          }
        }
      };
    }

    return output;
  }

  private computeForward(input: Tensor): Tensor {
    const [batch, channels, height, width] = input.shape;
    const outHeight = height * this.scale;
    const outWidth = width * this.scale;

    const outputSize = batch * channels * outHeight * outWidth;
    const output = new Float32Array(outputSize);

    for (let b = 0; b < batch; b++) {
      for (let c = 0; c < channels; c++) {
        for (let oh = 0; oh < outHeight; oh++) {
          for (let ow = 0; ow < outWidth; ow++) {
            const ih = Math.floor(oh / this.scale);
            const iw = Math.floor(ow / this.scale);

            const inputIdx = b * channels * height * width + c * height * width + ih * width + iw;
            const outputIdx = b * channels * outHeight * outWidth + c * outHeight * outWidth + oh * outWidth + ow;

            output[outputIdx] = input.data[inputIdx];
          }
        }
      }
    }

    return new Tensor(output, [batch, channels, outHeight, outWidth], input.requiresGrad);
  }

  parameters(): Tensor[] {
    return [];
  }

  zeroGrad(): void {
    // No parameters
  }
}
