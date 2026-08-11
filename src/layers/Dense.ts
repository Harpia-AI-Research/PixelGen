import { Tensor } from '../core/Tensor';

/**
 * Fully connected (Dense) layer
 */
export class Dense {
  public weights: Tensor;
  public bias: Tensor;
  private inputSize: number;
  private outputSize: number;

  constructor(inputSize: number, outputSize: number) {
    this.inputSize = inputSize;
    this.outputSize = outputSize;

    // Initialize weights with Xavier initialization
    this.weights = Tensor.xavier([inputSize, outputSize], true);
    this.bias = Tensor.zeros([outputSize], true);
  }

  /**
   * Forward pass
   * Input shape: [batchSize, inputSize]
   * Output shape: [batchSize, outputSize]
   */
  forward(input: Tensor): Tensor {
    if (input.shape.length !== 2) {
      throw new Error(`Dense layer expects 2D input, got shape ${input.shape}`);
    }

    if (input.shape[1] !== this.inputSize) {
      throw new Error(
        `Input size mismatch: expected ${this.inputSize}, got ${input.shape[1]}`
      );
    }

    // Matrix multiplication: input @ weights
    const output = input.matmul(this.weights);
    
    // Add bias (broadcast across batch)
    return output.add(this.bias);
  }

  /**
   * Get trainable parameters
   */
  parameters(): Tensor[] {
    return [this.weights, this.bias];
  }

  /**
   * Zero gradients
   */
  zeroGrad(): void {
    this.weights.zeroGrad();
    this.bias.zeroGrad();
  }
}
