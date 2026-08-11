/**
 * Basic training example for PixelGen
 * This demonstrates how to train a PixelArt model from scratch
 */

import {
  PixelGenModel,
  Trainer,
  DEFAULT_TRAINING_CONFIG,
  createTestDataset,
  saveModel,
  loadModel,
  inspectModel,
} from '../src';

async function main() {
  console.log('PixelGen - Basic Training Example');
  console.log('==================================\n');

  // Step 1: Create a model
  console.log('Step 1: Creating model...');
  const model = new PixelGenModel(3, 3); // RGB input and output
  console.log(`Model created with ${model.parameters().length} parameter tensors\n`);

  // Step 2: Create or load dataset
  console.log('Step 2: Creating test dataset...');
  const dataset = createTestDataset(
    20, // 20 images
    3,  // RGB
    32, // 32x32 pixels
    32
  );
  console.log(`Dataset created with ${dataset.size()} images\n`);

  // Step 3: Configure training
  console.log('Step 3: Configuring training...');
  const trainingConfig = {
    ...DEFAULT_TRAINING_CONFIG,
    epochs: 10,
    batchSize: 4,
    learningRate: 0.001,
    verbose: true,
  };
  console.log('Training configuration:');
  console.log(JSON.stringify(trainingConfig, null, 2));
  console.log('');

  // Step 4: Create trainer and train
  console.log('Step 4: Training model...');
  const trainer = new Trainer(model, trainingConfig);
  const stats = trainer.train(dataset);
  console.log('');

  // Step 5: Display training results
  console.log('Step 5: Training results');
  console.log('------------------------');
  console.log('Final statistics:');
  const finalStat = stats[stats.length - 1];
  console.log(`  Final loss: ${finalStat.loss.toFixed(6)}`);
  console.log(`  Total epochs: ${stats.length}`);
  console.log(`  Total training time: ${stats.reduce((sum, s) => sum + s.time, 0)}ms`);
  console.log('');

  // Step 6: Save model
  console.log('Step 6: Saving model...');
  const outputPath = 'examples/output/pixelgen-model.pgm';
  saveModel(model, outputPath, {
    learningRate: trainingConfig.learningRate,
    batchSize: trainingConfig.batchSize,
    epochs: trainingConfig.epochs,
    optimizer: trainingConfig.optimizer,
  });
  console.log('');

  // Step 7: Inspect saved model
  console.log('Step 7: Inspecting saved model...');
  const modelInfo = inspectModel(outputPath);
  console.log('Model info:');
  console.log(`  Architecture: ${modelInfo.metadata.architecture}`);
  console.log(`  Version: ${modelInfo.metadata.version}`);
  console.log(`  Total parameters: ${modelInfo.totalParameters}`);
  console.log(`  File size: ${(modelInfo.fileSize / 1024).toFixed(2)} KB`);
  console.log('');

  // Step 8: Load model back
  console.log('Step 8: Loading model from file...');
  const { model: loadedModel, hyperparameters } = loadModel(outputPath);
  console.log('Model loaded successfully!');
  console.log('Hyperparameters:');
  console.log(JSON.stringify(hyperparameters, null, 2));
  console.log('');

  console.log('Example completed successfully! ✓');
}

main().catch(console.error);
