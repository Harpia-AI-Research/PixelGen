import { Tensor } from '../core/Tensor';
import { relu } from '../core/activations';
import { Conv2D } from '../layers/Conv2D';
import { MaxPool2D, Upsample2D } from '../layers/Pooling';
import { Backend } from '../core/backend';

export class PixelGenModel {
  private conv1: Conv2D;
  private pool1: MaxPool2D;
  private conv2: Conv2D;
  private pool2: MaxPool2D;
  private conv3: Conv2D;

  private upsample1: Upsample2D;
  private conv4: Conv2D;
  private upsample2: Upsample2D;
  private conv5: Conv2D;
  private convOut: Conv2D;

  private inputChannels: number;
  private outputChannels: number;

  constructor(inputChannels: number = 3, outputChannels: number = 3, backend?: Backend) {
    this.inputChannels = inputChannels;
    this.outputChannels = outputChannels;

    this.conv1 = new Conv2D(inputChannels, 32, 3, 1, 1, backend);
    this.pool1 = new MaxPool2D(2, 2, backend);
    
    this.conv2 = new Conv2D(32, 64, 3, 1, 1, backend);
    this.pool2 = new MaxPool2D(2, 2, backend);
    
    this.conv3 = new Conv2D(64, 128, 3, 1, 1, backend);

    this.upsample1 = new Upsample2D(2, backend);
    this.conv4 = new Conv2D(128, 64, 3, 1, 1, backend);
    
    this.upsample2 = new Upsample2D(2, backend);
    this.conv5 = new Conv2D(64, 32, 3, 1, 1, backend);
    
    this.convOut = new Conv2D(32, outputChannels, 3, 1, 1, backend);
  }

  async forward(input: Tensor): Promise<Tensor> {
    let x = await this.conv1.forward(input);
    x = relu(x);
    x = await this.pool1.forward(x);

    x = await this.conv2.forward(x);
    x = relu(x);
    x = await this.pool2.forward(x);

    x = await this.conv3.forward(x);
    x = relu(x);

    x = await this.upsample1.forward(x);
    x = await this.conv4.forward(x);
    x = relu(x);

    x = await this.upsample2.forward(x);
    x = await this.conv5.forward(x);
    x = relu(x);

    x = await this.convOut.forward(x);

    return x;
  }

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

  zeroGrad(): void {
    this.conv1.zeroGrad();
    this.conv2.zeroGrad();
    this.conv3.zeroGrad();
    this.conv4.zeroGrad();
    this.conv5.zeroGrad();
    this.convOut.zeroGrad();
  }

  getMetadata(): ModelMetadata {
    return {
      version: '1.0.0',
      architecture: 'pixelgen-autoencoder-v1',
      inputChannels: this.inputChannels,
      outputChannels: this.outputChannels,
      inputSize: 32,
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
