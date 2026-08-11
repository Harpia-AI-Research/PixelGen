import { Tensor } from '../core/Tensor';
import { relu } from '../core/activations';
import { Conv2D } from '../layers/Conv2D';
import { MaxPool2D, Upsample2D } from '../layers/Pooling';

/**
 * PixelGen Model - Fixed architecture for v1
 * Simple convolutional autoencoder optimized for PixelArt
 */
export class PixelGenModel {
  // Encoder
  private conv1: Conv2D;
  private pool1: MaxPool2D;
  private conv2: Conv2D;
  private pool2: MaxPool2D;
  private conv3: Conv2D;

  // Decoder
  private upsample1: Upsample2D;
  private conv4: Conv2D;
  private upsample2: Upsample2D;
  private conv5: Conv2D;
  private convOut: Conv2D;

  private inputChannels: number;
  private outputChannels: number;

  constructor(inputChannels: number = 3, outputChannels: number = 3) {
    this.inputChannels = inputChannels;
    this.outputChannels = outputChannels;

    // Encoder: progressively reduce spatial dimensions, increase channels
    this.conv1 = new Conv2D(inputChannels, 32, 3, 1, 1); // 32x32 -> 32x32
    this.pool1 = new MaxPool2D(2, 2); // 32x32 -> 16x16
    
    this.conv2 = new Conv2D(32, 64, 3, 1, 1); // 16x16 -> 16x16
    this.pool2 = new MaxPool2D(2, 2); // 16x16 -> 8x8
    
    this.conv3 = new Conv2D(64, 128, 3, 1, 1); // 8x8 -> 8x8 (bottleneck)

    // Decoder: progressively increase spatial dimensions, reduce channels
    this.upsample1 = new Upsample2D(2); // 8x8 -> 16x16
    this.conv4 = new Conv2D(128, 64, 3, 1, 1); // 16x16 -> 16x16
    
    this.upsample2 = new Upsample2D(2); // 16x16 -> 32x32
    this.conv5 = new Conv2D(64, 32, 3, 1, 1); // 32x32 -> 32x32
    
    this.convOut = new Conv2D(32, outputChannels, 3, 1, 1); // 32x32 -> 32x32
  }

  /**
   * Forward pass through the model
   * Input: [batch, channels, height, width]
   * Output: [batch, channels, height, width]
   */
  forward(input: Tensor): Tensor {
    // Encoder
    let x = this.conv1.forward(input);
    x = relu(x);
    x = this.pool1.forward(x);

    x = this.conv2.forward(x);
    x = relu(x);
    x = this.pool2.forward(x);

    x = this.conv3.forward(x);
    x = relu(x); // Bottleneck

    // Decoder
    x = this.upsample1.forward(x);
    x = this.conv4.forward(x);
    x = relu(x);

    x = this.upsample2.forward(x);
    x = this.conv5.forward(x);
    x = relu(x);

    x = this.convOut.forward(x);
    // Note: No activation on output - we'll apply sigmoid during training if needed

    return x;
  }

  /**
   * Get all trainable parameters
   */
  parameters(): Tensor[] {
    return [
      ...this.conv1.parameters(),
      ...this.conv2.parameters(),
      ...this.conv3.parameters(),
      ...this.conv4.parameters(),
      ...this.conv5.parameters(),
      ...this.convOut.parameters(),
    ];
  }

  /**
   * Zero all gradients
   */
  zeroGrad(): void {
    this.conv1.zeroGrad();
    this.conv2.zeroGrad();
    this.conv3.zeroGrad();
    this.conv4.zeroGrad();
    this.conv5.zeroGrad();
    this.convOut.zeroGrad();
  }

  /**
   * Get model metadata for saving
   */
  getMetadata(): ModelMetadata {
    return {
      version: '1.0.0',
      architecture: 'pixelgen-autoencoder-v1',
      inputChannels: this.inputChannels,
      outputChannels: this.outputChannels,
      inputSize: 32, // Fixed for v1
      layers: [
        { type: 'conv2d', in: this.inputChannels, out: 32, kernel: 3 },
        { type: 'maxpool2d', size: 2 },
        { type: 'conv2d', in: 32, out: 64, kernel: 3 },
        { type: 'maxpool2d', size: 2 },
        { type: 'conv2d', in: 64, out: 128, kernel: 3 },
        { type: 'upsample2d', scale: 2 },
        { type: 'conv2d', in: 128, out: 64, kernel: 3 },
        { type: 'upsample2d', scale: 2 },
        { type: 'conv2d', in: 64, out: 32, kernel: 3 },
        { type: 'conv2d', in: 32, out: this.outputChannels, kernel: 3 },
      ],
    };
  }
}

export interface ModelMetadata {
  version: string;
  architecture: string;
  inputChannels: number;
  outputChannels: number;
  inputSize: number;
  layers: LayerConfig[];
}

interface LayerConfig {
  type: string;
  [key: string]: any;
}
