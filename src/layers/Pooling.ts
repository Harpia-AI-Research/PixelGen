import { Tensor } from '../core/Tensor';

/**
 * Max Pooling 2D layer
 */
export class MaxPool2D {
  private poolSize: number;
  private stride: number;

  constructor(poolSize: number = 2, stride?: number) {
    this.poolSize = poolSize;
    this.stride = stride || poolSize;
  }

  /**
   * Forward pass
   * Input shape: [batch, channels, height, width]
   */
  forward(input: Tensor): Tensor {
    if (input.shape.length !== 4) {
      throw new Error(`MaxPool2D expects 4D input, got ${input.shape}`);
    }

    const [batch, channels, height, width] = input.shape;
    const outHeight = Math.floor((height - this.poolSize) / this.stride) + 1;
    const outWidth = Math.floor((width - this.poolSize) / this.stride) + 1;

    const outputSize = batch * channels * outHeight * outWidth;
    const output = new Float32Array(outputSize);

    for (let b = 0; b < batch; b++) {
      for (let c = 0; c < channels; c++) {
        for (let oh = 0; oh < outHeight; oh++) {
          for (let ow = 0; ow < outWidth; ow++) {
            let max = -Infinity;

            // Find max in pool window
            for (let ph = 0; ph < this.poolSize; ph++) {
              for (let pw = 0; pw < this.poolSize; pw++) {
                const ih = oh * this.stride + ph;
                const iw = ow * this.stride + pw;

                const inputIdx = b * channels * height * width + c * height * width + ih * width + iw;

                if (input.data[inputIdx] > max) {
                  max = input.data[inputIdx];
                }
              }
            }

            const outputIdx = b * channels * outHeight * outWidth + c * outHeight * outWidth + oh * outWidth + ow;
            output[outputIdx] = max;
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
    // No parameters to zero
  }
}

/**
 * Average Pooling 2D layer
 */
export class AvgPool2D {
  private poolSize: number;
  private stride: number;

  constructor(poolSize: number = 2, stride?: number) {
    this.poolSize = poolSize;
    this.stride = stride || poolSize;
  }

  forward(input: Tensor): Tensor {
    if (input.shape.length !== 4) {
      throw new Error(`AvgPool2D expects 4D input, got ${input.shape}`);
    }

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

            // Sum pool window
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

/**
 * Upsampling layer (nearest neighbor)
 * Used for decoder part of autoencoder
 */
export class Upsample2D {
  private scale: number;

  constructor(scale: number = 2) {
    this.scale = scale;
  }

  forward(input: Tensor): Tensor {
    if (input.shape.length !== 4) {
      throw new Error(`Upsample2D expects 4D input, got ${input.shape}`);
    }

    const [batch, channels, height, width] = input.shape;
    const outHeight = height * this.scale;
    const outWidth = width * this.scale;

    const outputSize = batch * channels * outHeight * outWidth;
    const output = new Float32Array(outputSize);

    for (let b = 0; b < batch; b++) {
      for (let c = 0; c < channels; c++) {
        for (let oh = 0; oh < outHeight; oh++) {
          for (let ow = 0; ow < outWidth; ow++) {
            // Nearest neighbor
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
