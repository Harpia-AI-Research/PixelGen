import { Tensor } from '../Tensor';
import { Backend, Conv2DForwardResult, Conv2DBackwardResult, PoolForwardResult, UpsampleForwardResult, MatmulResult } from './Backend';

export class CPUBackend implements Backend {
  async conv2dForward(
    input: Tensor,
    filters: Tensor,
    bias: Tensor,
    outChannels: number,
    kernelSize: number,
    stride: number,
    padding: number
  ): Promise<Conv2DForwardResult> {
    return Promise.resolve(this.computeConv2DForward(input, filters, bias, outChannels, kernelSize, stride, padding));
  }

  async conv2dBackward(
    input: Tensor,
    paddedInput: Tensor,
    filters: Tensor,
    outputGrad: Tensor,
    outChannels: number,
    kernelSize: number,
    stride: number,
    padding: number,
    batch: number,
    channels: number,
    height: number,
    width: number,
    outHeight: number,
    outWidth: number
  ): Promise<Conv2DBackwardResult> {
    return Promise.resolve(this.computeConv2DBackward(input, paddedInput, filters, outputGrad, outChannels, kernelSize, stride, padding, batch, channels, height, width, outHeight, outWidth));
  }

  async poolForward(input: Tensor, poolSize: number, stride: number): Promise<PoolForwardResult> {
    return Promise.resolve(this.computePoolForward(input, poolSize, stride));
  }

  async avgPoolForward(input: Tensor, poolSize: number, stride: number): Promise<PoolForwardResult> {
    return Promise.resolve(this.computeAvgPoolForward(input, poolSize, stride));
  }

  async upsampleForward(input: Tensor, scale: number): Promise<UpsampleForwardResult> {
    return Promise.resolve(this.computeUpsampleForward(input, scale));
  }

  async matmul(a: Tensor, b: Tensor): Promise<MatmulResult> {
    return Promise.resolve(this.computeMatmul(a, b));
  }

  private computeConv2DForward(
    input: Tensor,
    filters: Tensor,
    bias: Tensor,
    outChannels: number,
    kernelSize: number,
    stride: number,
    padding: number
  ): Conv2DForwardResult {
    const [batch, channels, height, width] = input.shape;

    const outHeight = Math.floor((height + 2 * padding - kernelSize) / stride) + 1;
    const outWidth = Math.floor((width + 2 * padding - kernelSize) / stride) + 1;

    const paddedInput = padding > 0 ? this.padInput(input, padding) : input;
    const [, , paddedHeight, paddedWidth] = paddedInput.shape;

    const outputSize = batch * outChannels * outHeight * outWidth;
    const output = new Float32Array(outputSize);

    for (let b = 0; b < batch; b++) {
      for (let oc = 0; oc < outChannels; oc++) {
        for (let oh = 0; oh < outHeight; oh++) {
          for (let ow = 0; ow < outWidth; ow++) {
            let sum = 0;

            for (let ic = 0; ic < channels; ic++) {
              for (let kh = 0; kh < kernelSize; kh++) {
                for (let kw = 0; kw < kernelSize; kw++) {
                  const ih = oh * stride + kh;
                  const iw = ow * stride + kw;

                  const inputIdx =
                    b * channels * paddedHeight * paddedWidth +
                    ic * paddedHeight * paddedWidth +
                    ih * paddedWidth +
                    iw;

                  const filterIdx =
                    oc * channels * kernelSize * kernelSize +
                    ic * kernelSize * kernelSize +
                    kh * kernelSize +
                    kw;

                  sum += paddedInput.data[inputIdx] * filters.data[filterIdx];
                }
              }
            }

            sum += bias.data[oc];

            const outputIdx =
              b * outChannels * outHeight * outWidth +
              oc * outHeight * outWidth +
              oh * outWidth +
              ow;

            output[outputIdx] = sum;
          }
        }
      }
    }

    const outputTensor = new Tensor(output, [batch, outChannels, outHeight, outWidth], true);
    return { output: outputTensor, paddedInput };
  }

  private computeConv2DBackward(
    input: Tensor,
    paddedInput: Tensor,
    filters: Tensor,
    outputGrad: Tensor,
    outChannels: number,
    kernelSize: number,
    stride: number,
    padding: number,
    batch: number,
    channels: number,
    height: number,
    width: number,
    outHeight: number,
    outWidth: number
  ): Conv2DBackwardResult {
    const [, , paddedHeight, paddedWidth] = paddedInput.shape;

    const filterGrad = new Float32Array(filters.data.length);
    const biasGrad = new Float32Array(outChannels);
    const gradPadded = new Float32Array(paddedInput.data.length);

    for (let b = 0; b < batch; b++) {
      for (let oc = 0; oc < outChannels; oc++) {
        for (let oh = 0; oh < outHeight; oh++) {
          for (let ow = 0; ow < outWidth; ow++) {
            const outputIdx =
              b * outChannels * outHeight * outWidth +
              oc * outHeight * outWidth +
              oh * outWidth +
              ow;

            const g = outputGrad.data[outputIdx];

            biasGrad[oc] += g;

            for (let ic = 0; ic < channels; ic++) {
              for (let kh = 0; kh < kernelSize; kh++) {
                for (let kw = 0; kw < kernelSize; kw++) {
                  const ih = oh * stride + kh;
                  const iw = ow * stride + kw;

                  const filterIdx =
                    oc * channels * kernelSize * kernelSize +
                    ic * kernelSize * kernelSize +
                    kh * kernelSize +
                    kw;

                  const inputIdx =
                    b * channels * paddedHeight * paddedWidth +
                    ic * paddedHeight * paddedWidth +
                    ih * paddedWidth +
                    iw;

                  filterGrad[filterIdx] += g * paddedInput.data[inputIdx];
                  gradPadded[inputIdx] += g * filters.data[filterIdx];
                }
              }
            }
          }
        }
      }
    }

    const inputGrad = new Float32Array(input.data.length);
    if (padding > 0) {
      for (let b = 0; b < batch; b++) {
        for (let c = 0; c < channels; c++) {
          for (let h = 0; h < height; h++) {
            for (let w = 0; w < width; w++) {
              const inputIdx = b * channels * height * width + c * height * width + h * width + w;
              const paddedIdx =
                b * channels * paddedHeight * paddedWidth +
                c * paddedHeight * paddedWidth +
                (h + padding) * paddedWidth +
                (w + padding);

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
      filterGrad: new Tensor(filterGrad, filters.shape, false),
      biasGrad: new Tensor(biasGrad, [outChannels], false),
    };
  }

  private computePoolForward(input: Tensor, poolSize: number, stride: number): PoolForwardResult {
    const [batch, channels, height, width] = input.shape;
    const outHeight = Math.floor((height - poolSize) / stride) + 1;
    const outWidth = Math.floor((width - poolSize) / stride) + 1;

    const outputSize = batch * channels * outHeight * outWidth;
    const output = new Float32Array(outputSize);
    const argmax = new Int32Array(outputSize);

    for (let b = 0; b < batch; b++) {
      for (let c = 0; c < channels; c++) {
        for (let oh = 0; oh < outHeight; oh++) {
          for (let ow = 0; ow < outWidth; ow++) {
            let max = -Infinity;
            let maxIdx = 0;

            for (let ph = 0; ph < poolSize; ph++) {
              for (let pw = 0; pw < poolSize; pw++) {
                const ih = oh * stride + ph;
                const iw = ow * stride + pw;

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

    return { output: new Tensor(output, [batch, channels, outHeight, outWidth], false), argmax };
  }

  private computeAvgPoolForward(input: Tensor, poolSize: number, stride: number): PoolForwardResult {
    const [batch, channels, height, width] = input.shape;
    const outHeight = Math.floor((height - poolSize) / stride) + 1;
    const outWidth = Math.floor((width - poolSize) / stride) + 1;

    const outputSize = batch * channels * outHeight * outWidth;
    const output = new Float32Array(outputSize);
    const poolArea = poolSize * poolSize;

    for (let b = 0; b < batch; b++) {
      for (let c = 0; c < channels; c++) {
        for (let oh = 0; oh < outHeight; oh++) {
          for (let ow = 0; ow < outWidth; ow++) {
            let sum = 0;

            for (let ph = 0; ph < poolSize; ph++) {
              for (let pw = 0; pw < poolSize; pw++) {
                const ih = oh * stride + ph;
                const iw = ow * stride + pw;

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

    return { output: new Tensor(output, [batch, channels, outHeight, outWidth], false) };
  }

  private computeUpsampleForward(input: Tensor, scale: number): UpsampleForwardResult {
    const [batch, channels, height, width] = input.shape;
    const outHeight = height * scale;
    const outWidth = width * scale;

    const outputSize = batch * channels * outHeight * outWidth;
    const output = new Float32Array(outputSize);

    for (let b = 0; b < batch; b++) {
      for (let c = 0; c < channels; c++) {
        for (let oh = 0; oh < outHeight; oh++) {
          for (let ow = 0; ow < outWidth; ow++) {
            const ih = Math.floor(oh / scale);
            const iw = Math.floor(ow / scale);

            const inputIdx = b * channels * height * width + c * height * width + ih * width + iw;
            const outputIdx = b * channels * outHeight * outWidth + c * outHeight * outWidth + oh * outWidth + ow;

            output[outputIdx] = input.data[inputIdx];
          }
        }
      }
    }

    return { output: new Tensor(output, [batch, channels, outHeight, outWidth], false) };
  }

  private computeMatmul(a: Tensor, b: Tensor): MatmulResult {
    if (a.shape.length !== 2 || b.shape.length !== 2) {
      throw new Error('Matmul requires 2D tensors');
    }

    const [m, n] = a.shape;
    const [n2, p] = b.shape;

    if (n !== n2) {
      throw new Error(`Cannot multiply matrices of shapes ${a.shape} and ${b.shape}`);
    }

    const resultData = new Float32Array(m * p);

    for (let i = 0; i < m; i++) {
      for (let j = 0; j < p; j++) {
        let sum = 0;
        for (let k = 0; k < n; k++) {
          sum += a.data[i * n + k] * b.data[k * p + j];
        }
        resultData[i * p + j] = sum;
      }
    }

    return { output: new Tensor(resultData, [m, p], false) };
  }

  private padInput(input: Tensor, padding: number): Tensor {
    const [batch, channels, height, width] = input.shape;
    const paddedHeight = height + 2 * padding;
    const paddedWidth = width + 2 * padding;

    const padded = new Float32Array(batch * channels * paddedHeight * paddedWidth);

    for (let b = 0; b < batch; b++) {
      for (let c = 0; c < channels; c++) {
        for (let h = 0; h < height; h++) {
          for (let w = 0; w < width; w++) {
            const inputIdx = b * channels * height * width + c * height * width + h * width + w;
            const paddedIdx =
              b * channels * paddedHeight * paddedWidth +
              c * paddedHeight * paddedWidth +
              (h + padding) * paddedWidth +
              (w + padding);

            padded[paddedIdx] = input.data[inputIdx];
          }
        }
      }
    }

    return new Tensor(padded, [batch, channels, paddedHeight, paddedWidth], input.requiresGrad);
  }
}
