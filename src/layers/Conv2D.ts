import { Tensor } from '../core/Tensor';

/**
 * 2D Convolutional layer
 * Simplified implementation for PixelArt (small images)
 */
export class Conv2D {
  public filters: Tensor;
  public bias: Tensor;
  private inChannels: number;
  private outChannels: number;
  private kernelSize: number;
  private stride: number;
  private padding: number;

  constructor(
    inChannels: number,
    outChannels: number,
    kernelSize: number = 3,
    stride: number = 1,
    padding: number = 1
  ) {
    this.inChannels = inChannels;
    this.outChannels = outChannels;
    this.kernelSize = kernelSize;
    this.stride = stride;
    this.padding = padding;

    // Initialize filters [outChannels, inChannels, kernelSize, kernelSize]
    const filterSize = outChannels * inChannels * kernelSize * kernelSize;
    this.filters = Tensor.xavier(
      [outChannels, inChannels, kernelSize, kernelSize],
      true
    );
    this.bias = Tensor.zeros([outChannels], true);
  }

  /**
   * Forward pass
   * Input shape: [batch, channels, height, width]
   * Output shape: [batch, outChannels, outHeight, outWidth]
   */
  forward(input: Tensor): Tensor {
    if (input.shape.length !== 4) {
      throw new Error(`Conv2D expects 4D input [batch, channels, height, width], got ${input.shape}`);
    }

    const [batch, channels, height, width] = input.shape;

    if (channels !== this.inChannels) {
      throw new Error(
        `Input channels mismatch: expected ${this.inChannels}, got ${channels}`
      );
    }

    // Calculate output dimensions
    const outHeight = Math.floor((height + 2 * this.padding - this.kernelSize) / this.stride) + 1;
    const outWidth = Math.floor((width + 2 * this.padding - this.kernelSize) / this.stride) + 1;

    // Pad input if needed
    const paddedInput = this.padding > 0 ? this.padInput(input) : input;
    const [, , paddedHeight, paddedWidth] = paddedInput.shape;

    // Output tensor
    const outputSize = batch * this.outChannels * outHeight * outWidth;
    const output = new Float32Array(outputSize);

    // Perform convolution
    for (let b = 0; b < batch; b++) {
      for (let oc = 0; oc < this.outChannels; oc++) {
        for (let oh = 0; oh < outHeight; oh++) {
          for (let ow = 0; ow < outWidth; ow++) {
            let sum = 0;

            // Convolve with kernel
            for (let ic = 0; ic < this.inChannels; ic++) {
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
                    oc * this.inChannels * this.kernelSize * this.kernelSize +
                    ic * this.kernelSize * this.kernelSize +
                    kh * this.kernelSize +
                    kw;

                  sum += paddedInput.data[inputIdx] * this.filters.data[filterIdx];
                }
              }
            }

            // Add bias
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

    return new Tensor(
      output,
      [batch, this.outChannels, outHeight, outWidth],
      input.requiresGrad
    );
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
