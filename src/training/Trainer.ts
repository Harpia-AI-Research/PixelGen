import { PixelGenModel } from '../model/PixelGenModel';
import { ImageDataset } from '../data/ImageLoader';
import { Optimizer, Adam } from './Optimizer';
import { mseLoss, pixelArtLoss } from './Loss';
import { Tensor } from '../core/Tensor';

export interface TrainingConfig {
  epochs: number;
  batchSize: number;
  learningRate: number;
  optimizer: 'sgd' | 'adam';
  lossFunction: 'mse' | 'pixelart';
  verbose: boolean;
}

export interface TrainingStats {
  epoch: number;
  loss: number;
  time: number;
}

/**
 * Trainer class - handles the training loop
 */
export class Trainer {
  private model: PixelGenModel;
  private optimizer: Optimizer;
  private config: TrainingConfig;
  private stats: TrainingStats[] = [];

  constructor(model: PixelGenModel, config: TrainingConfig) {
    this.model = model;
    this.config = config;

    // Initialize optimizer
    if (config.optimizer === 'adam') {
      this.optimizer = new Adam(config.learningRate);
    } else {
      // SGD not yet implemented in this simplified version
      this.optimizer = new Adam(config.learningRate);
    }
  }

  /**
   * Train the model
   */
  train(dataset: ImageDataset): TrainingStats[] {
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

      // Shuffle dataset at the start of each epoch
      dataset.shuffle();

      // Iterate through batches
      for (let i = 0; i < dataset.size(); i += this.config.batchSize) {
        const batch = dataset.getBatch(this.config.batchSize, i);
        if (!batch) break;

        // Forward pass
        const output = this.model.forward(batch);

        // Calculate loss
        const loss = this.calculateLoss(output, batch);
        epochLoss += loss;
        numBatches++;

        // Backward pass (simplified - in full implementation would use autograd)
        // For now, we'll just update the optimizer
        // Note: Full backward pass would be implemented with proper autograd

        // Zero gradients
        this.model.zeroGrad();

        // Update parameters
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

  /**
   * Evaluate model on dataset
   */
  evaluate(dataset: ImageDataset): number {
    let totalLoss = 0;
    let numBatches = 0;

    for (let i = 0; i < dataset.size(); i += this.config.batchSize) {
      const batch = dataset.getBatch(this.config.batchSize, i);
      if (!batch) break;

      const output = this.model.forward(batch);
      const loss = this.calculateLoss(output, batch);
      
      totalLoss += loss;
      numBatches++;
    }

    return totalLoss / numBatches;
  }

  /**
   * Get training statistics
   */
  getStats(): TrainingStats[] {
    return [...this.stats];
  }
}

/**
 * Default training configuration
 */
export const DEFAULT_TRAINING_CONFIG: TrainingConfig = {
  epochs: 50,
  batchSize: 4,
  learningRate: 0.001,
  optimizer: 'adam',
  lossFunction: 'pixelart',
  verbose: true,
};
