/**
 * PixelGen - Machine Learning for PixelArt
 * Built from scratch in TypeScript
 */

// Core
export { Tensor } from './core/Tensor';
export * from './core/activations';

// Layers
export { Dense } from './layers/Dense';
export { Conv2D } from './layers/Conv2D';
export { MaxPool2D, AvgPool2D, Upsample2D } from './layers/Pooling';

// Model
export { PixelGenModel, ModelMetadata } from './model/PixelGenModel';

// Training
export { SGD, Adam, Optimizer } from './training/Optimizer';
export { mseLoss, maeLoss, bceLoss, pixelArtLoss } from './training/Loss';
export { Trainer, TrainingConfig, TrainingStats, DEFAULT_TRAINING_CONFIG } from './training/Trainer';

// Data
export {
  ImageDataset,
  DatasetConfig,
  ImageData,
  normalizeTensor,
  denormalizeTensor,
  createTestDataset,
} from './data/ImageLoader';

// I/O
export { saveModel, loadModel, inspectModel, Hyperparameters, PGMFile } from './io/ModelIO';
