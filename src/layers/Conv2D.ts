import { Tensor } from '../core/Tensor';
import { Backend } from '../core/backend';
import { WasmBackend } from '../core/backend/WasmBackend';

export class Conv2D {
  public filters: Tensor;
  public bias: Tensor;
  private inChannels: number;
  private outChannels: number;
  private kernelSize: number;
  private stride: number;
  private padding: number;
  private backend?: Backend;

  constructor(
    inChannels: number,
    outChannels: number,
    kernelSize: number = 3,
    stride: number = 1,
    padding: number = 1,
    backend?: Backend
  ) {
    this.inChannels = inChannels;
    this.outChannels = outChannels;
    this.kernelSize = kernelSize;
    this.stride = stride;
    this.padding = padding;
    this.backend = backend;

    const filterSize = outChannels * inChannels * kernelSize * kernelSize;
    this.filters = Tensor.xavier(
      [outChannels, inChannels, kernelSize, kernelSize],
      true
    );
    this.bias = Tensor.zeros([outChannels], true);
  }

  async forward(input: Tensor): Promise<Tensor> {
    if (input.shape.length !== 4) {
      throw new Error(`Conv2D expects 4D input [batch, channels, height, width], got ${input.shape}`);
    }

    const [batch, channels, height, width] = input.shape;

    if (channels !== this.inChannels) {
      throw new Error(
        `Input channels mismatch: expected ${this.inChannels}, got ${channels}`
      );
    }

    const outHeight = Math.floor((height + 2 * this.padding - this.kernelSize) / this.stride) + 1;
    const outWidth = Math.floor((width + 2 * this.padding - this.kernelSize) / this.stride) + 1;

    const useBackend = this.backend || Tensor.backend;

    let result: { output: Tensor; paddedInput: Tensor };

    if (useBackend) {
      result = await useBackend.conv2dForward(
        input,
        this.filters,
        this.bias,
        this.outChannels,
        this.kernelSize,
        this.stride,
        this.padding
      );
    } else {
      const paddedInput = this.padding > 0 ? this.padInput(input) : input;
      const output = this.computeConvolution(input, paddedInput, batch, channels, height, width, outHeight, outWidth);
      result = { output, paddedInput };
    }

    const outputTensor = result.output;

    outputTensor._prev.add(input);
    outputTensor._prev.add(this.filters);
    outputTensor._prev.add(this.bias);

    const paddedInput = result.paddedInput;

    outputTensor._backward = async () => {
      if (!outputTensor.grad) return;

      if (!this.filters.grad) this.filters.grad = Tensor.zerosLike(this.filters);
      if (!this.bias.grad) this.bias.grad = Tensor.zerosLike(this.bias);
      if (input.requiresGrad && !input.grad) input.grad = Tensor.zerosLike(input);

      const runBackend = async (backend: Backend) => {
        const backwardResult = await backend.conv2dBackward(
          input,
          paddedInput,
          this.filters,
          outputTensor.grad!,
          this.outChannels,
          this.kernelSize,
          this.stride,
          this.padding,
          batch,
          channels,
          height,
          width,
          outHeight,
          outWidth
        );

        this.filters.grad = backwardResult.filterGrad;
        this.bias.grad = backwardResult.biasGrad;

        if (input.requiresGrad) {
          input.grad = backwardResult.inputGrad;
        }
      };

      if (useBackend) {
        if (useBackend instanceof WasmBackend) {
          await runBackend(useBackend);
        } else {
          await useBackend.then(runBackend);
        }
      } else {
        const backwardResult = this.computeBackward(
          input,
          paddedInput,
          outputTensor.grad,
          batch,
          channels,
          height,
          width,
          outHeight,
          outWidth
        );

        this.filters.grad = backwardResult.filterGrad;
        this.bias.grad = backwardResult.biasGrad;

        if (input.requiresGrad) {
          input.grad = backwardResult.inputGrad;
        }
      }
    };

    return outputTensor;
  }

  private computeConvolution(
    input: Tensor,
    paddedInput: Tensor,
    batch: number,
    channels: number,
    height: number,
    width: number,
    outHeight: number,
    outWidth: number
  ): Tensor {
    const [, , paddedHeight, paddedWidth] = paddedInput.shape;
    const outputSize = batch * this.outChannels * outHeight * outWidth;
    const output = new Float32Array(outputSize);

    for (let b = 0; b < batch; b++) {
      for (let oc = 0; oc < this.outChannels; oc++) {
        for (let oh = 0; oh < outHeight; oh++) {
          for (let ow = 0; ow < outWidth; ow++) {
            let sum = 0;

            for (let ic = 0; ic < channels; ic++) {
              for (let kh = 0; kh < this.kernelSize; kh++) {
                for (let kw = 0; kw < this.kernelSize; kw++) {
                  const ih = oh * this.stride + kh;
                  const iw = ow * this.stride + kw;

                  const inputIdx =
                    b * channels * paddedHeight * paddedWidth +
                    ic * paddedHeight * paddedWidth +
                    ih * paddedWidth +
                    iw;

                  const filterIdx =
                    oc * channels * this.kernelSize * this.kernelSize +
                    ic * this.kernelSize * this.kernelSize +
                    kh * this.kernelSize +
                    kw;

                  sum += paddedInput.data[inputIdx] * this.filters.data[filterIdx];
                }
              }
            }

            sum += this.bias.data[oc];

            const outputIdx =
              b * this.outChannels * outHeight * outWidth +
              oc * outHeight * outWidth +
              oh * outWidth +
              ow;

            output[outputIdx] = sum;
          }
        }
      }
    }

    return new Tensor(output, [batch, this.outChannels, outHeight, outWidth], true);
  }

  private computeBackward(
    input: Tensor,
    paddedInput: Tensor,
    outputGrad: Tensor,
    batch: number,
    channels: number,
    height: number,
    width: number,
    outHeight: number,
    outWidth: number
  ): { inputGrad: Tensor; filterGrad: Tensor; biasGrad: Tensor } {
    const [, , paddedHeight, paddedWidth] = paddedInput.shape;

    const filterGrad = new Float32Array(this.filters.data.length);
    const biasGrad = new Float32Array(this.outChannels);
    const gradPadded = new Float32Array(paddedInput.data.length);

    for (let b = 0; b < batch; b++) {
      for (let oc = 0; oc < this.outChannels; oc++) {
        for (let oh = 0; oh < outHeight; oh++) {
          for (let ow = 0; ow < outWidth; ow++) {
            const outputIdx =
              b * this.outChannels * outHeight * outWidth +
              oc * outHeight * outWidth +
              oh * outWidth +
              ow;

            const g = outputGrad.data[outputIdx];

            biasGrad[oc] += g;

            for (let ic = 0; ic < channels; ic++) {
              for (let kh = 0; kh < this.kernelSize; kh++) {
                for (let kw = 0; kw < this.kernelSize; kw++) {
                  const ih = oh * this.stride + kh;
                  const iw = ow * this.stride + kw;

                  const filterIdx =
                    oc * channels * this.kernelSize * this.kernelSize +
                    ic * this.kernelSize * this.kernelSize +
                    kh * this.kernelSize +
                    kw;

                  const inputIdx =
                    b * channels * paddedHeight * paddedWidth +
                    ic * paddedHeight * paddedWidth +
                    ih * paddedWidth +
                    iw;

                  filterGrad[filterIdx] += g * paddedInput.data[inputIdx];
                  gradPadded[inputIdx] += g * this.filters.data[filterIdx];
                }
              }
            }
          }
        }
      }
    }

    const inputGrad = new Float32Array(input.data.length);
    if (this.padding > 0) {
      for (let b = 0; b < batch; b++) {
        for (let c = 0; c < channels; c++) {
          for (let h = 0; h < height; h++) {
            for (let w = 0; w < width; w++) {
              const inputIdx = b * channels * height * width + c * height * width + h * width + w;
              const paddedIdx =
                b * channels * paddedHeight * paddedWidth +
                c * paddedHeight * paddedWidth +
                (h + this.padding) * paddedWidth +
                (w + this.padding);

              inputGrad[inputIdx] = gradPadded[paddedIdx];
            }
          }
        }
      }
    } else {
      for (let i = 0; i < input.data.length; i++) {
        inputGrad[i] = gradPadded[i];
      }
    }

    return {
      inputGrad: new Tensor(inputGrad, [batch, channels, height, width], false),
      filterGrad: new Tensor(filterGrad, this.filters.shape, false),
      biasGrad: new Tensor(biasGrad, [this.outChannels], false),
    };
  }

  private padInput(input: Tensor): Tensor {
    const [batch, channels, height, width] = input.shape;
    const paddedHeight = height + 2 * this.padding;
    const paddedWidth = width + 2 * this.padding;

    const padded = new Float32Array(batch * channels * paddedHeight * paddedWidth);

    for (let b = 0; b < batch; b++) {
      for (let c = 0; c < channels; c++) {
        for (let h = 0; h < height; h++) {
          for (let w = 0; w < width; w++) {
            const inputIdx = b * channels * height * width + c * height * width + h * width + w;
            const paddedIdx =
              b * channels * paddedHeight * paddedWidth +
              c * paddedHeight * paddedWidth +
              (h + this.padding) * paddedWidth +
              (w + this.padding);

            padded[paddedIdx] = input.data[inputIdx];
          }
        }
      }
    }

    return new Tensor(padded, [batch, channels, paddedHeight, paddedWidth], input.requiresGrad);
  }

  parameters(): Tensor[] {
    return [this.filters, this.bias];
  }

  zeroGrad(): void {
    this.filters.zeroGrad();
    this.bias.zeroGrad();
  }
}
