import { PixelGenModel } from '../model/PixelGenModel';
import { ImageDataset } from '../data/ImageLoader';
import { Optimizer, Adam } from './Optimizer';
import { mseLoss, pixelArtLoss, mseLossGrad, pixelArtLossGrad } from './Loss';
import { Tensor } from '../core/Tensor';
import { Backend } from '../core/backend';

export interface TrainingConfig {
  epochs: number;
  batchSize: number;
  learningRate: number;
  optimizer: 'sgd' | 'adam';
  lossFunction: 'mse' | 'pixelart';
  verbose: boolean;
  backend?: Backend;
}

export interface TrainingCallbacks {
  onEpoch?: (epoch: number, loss: number, model: PixelGenModel) => void;
  onBatch?: (epoch: number, batch: number, loss: number, model: PixelGenModel) => void;
}

export interface TrainingStats {
  epoch: number;
  loss: number;
  time: number;
}

export class Trainer {
  private model: PixelGenModel;
  private optimizer: Optimizer;
  private config: TrainingConfig;
  private stats: TrainingStats[] = [];

  constructor(model: PixelGenModel, config: TrainingConfig) {
    this.model = model;
    this.config = config;

    if (config.optimizer === 'adam') {
      this.optimizer = new Adam(config.learningRate);
    } else {
      this.optimizer = new Adam(config.learningRate);
    }
  }

  async train(dataset: ImageDataset, callbacks?: TrainingCallbacks): Promise<TrainingStats[]> {
    console.log('Starting training...');
    console.log(`Epochs: ${this.config.epochs}`);
    console.log(`Batch size: ${this.config.batchSize}`);
    console.log(`Learning rate: ${this.config.learningRate}`);
    console.log(`Dataset size: ${dataset.size()} images`);
    console.log('---');

    for (let epoch = 0; epoch < this.config.epochs; epoch++) {
      const startTime = Date.now();
      let epochLoss = 0;
      let numBatches = 0;

      dataset.shuffle();

      for (let i = 0; i < dataset.size(); i += this.config.batchSize) {
        const batch = dataset.getBatch(this.config.batchSize, i);
        if (!batch) break;

        this.model.zeroGrad();

        const output = await this.model.forward(batch);

        const loss = this.calculateLoss(output, batch);
        epochLoss += loss;
        numBatches++;

        const outputGrad = this.calculateLossGrad(output, batch);
        output.grad = outputGrad;
        
        await output.backward();

        this.optimizer.step(this.model.parameters());
      }

      const avgLoss = epochLoss / numBatches;
      const epochTime = Date.now() - startTime;

      const stat: TrainingStats = {
        epoch: epoch + 1,
        loss: avgLoss,
        time: epochTime,
      };
      this.stats.push(stat);

      if (this.config.verbose) {
        console.log(
          `Epoch ${epoch + 1}/${this.config.epochs} - Loss: ${avgLoss.toFixed(6)} - Time: ${epochTime}ms`
        );
      }

      if (callbacks?.onEpoch) {
        callbacks.onEpoch(epoch + 1, avgLoss, this.model);
      }
    }

    console.log('---');
    console.log('Training complete!');
    return this.stats;
  }

  private calculateLoss(predicted: Tensor, target: Tensor): number {
    if (this.config.lossFunction === 'pixelart') {
      return pixelArtLoss(predicted, target);
    } else {
      return mseLoss(predicted, target);
    }
  }

  private calculateLossGrad(predicted: Tensor, target: Tensor): Tensor {
    if (this.config.lossFunction === 'pixelart') {
      return pixelArtLossGrad(predicted, target);
    } else {
      return mseLossGrad(predicted, target);
    }
  }

  async evaluate(dataset: ImageDataset): Promise<number> {
    let totalLoss = 0;
    let numBatches = 0;

    for (let i = 0; i < dataset.size(); i += this.config.batchSize) {
      const batch = dataset.getBatch(this.config.batchSize, i);
      if (!batch) break;

      const output = await this.model.forward(batch);
      const loss = this.calculateLoss(output, batch);
      
      totalLoss += loss;
      numBatches++;
    }

    return totalLoss / numBatches;
  }

  getStats(): TrainingStats[] {
    return [...this.stats];
  }
}

export const DEFAULT_TRAINING_CONFIG: TrainingConfig = {
  epochs: 50,
  batchSize: 4,
  learningRate: 0.001,
  optimizer: 'adam',
  lossFunction: 'pixelart',
  verbose: true,
};
