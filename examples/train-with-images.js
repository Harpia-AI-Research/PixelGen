"use strict";
/**
 * Training example using real images from a directory
 * This demonstrates how to train a PixelArt model with PNG/JPG files
 */
Object.defineProperty(exports, "__esModule", { value: true });
const src_1 = require("../src");
async function main() {
    console.log('PixelGen - Training with Real Images');
    console.log('=====================================\n');
    // Step 1: Create a model
    console.log('Step 1: Creating model...');
    const model = new src_1.PixelGenModel(3, 3); // RGB input and output
    console.log('Model created\n');
    // Step 2: Load images from directory
    console.log('Step 2: Loading images from directory...');
    const dataset = new src_1.ImageDataset({
        targetSize: 32,
        normalize: true,
        channels: 3,
    });
    // Change this path to your dataset directory
    const datasetPath = './dataset';
    try {
        dataset.loadFromDirectory(datasetPath);
    }
    catch (error) {
        console.error(`\nError: ${error.message}`);
        console.log('\nTo use this example:');
        console.log('1. Create a directory called "dataset" in the project root');
        console.log('2. Add PNG or JPG images of PixelArt to that directory');
        console.log('3. Run this example again\n');
        console.log('Example structure:');
        console.log('  dataset/');
        console.log('  ├── sprite1.png');
        console.log('  ├── sprite2.png');
        console.log('  └── sprite3.png\n');
        process.exit(1);
    }
    if (dataset.size() === 0) {
        console.error('No images were loaded. Please check your dataset directory.');
        process.exit(1);
    }
    console.log(`Loaded ${dataset.size()} images\n`);
    // Step 3: Configure training
    console.log('Step 3: Configuring training...');
    const trainingConfig = {
        epochs: 50,
        batchSize: Math.min(4, dataset.size()), // Adjust batch size based on dataset
        learningRate: 0.001,
        optimizer: 'adam',
        lossFunction: 'pixelart',
        verbose: true,
    };
    console.log('Training configuration:');
    console.log(JSON.stringify(trainingConfig, null, 2));
    console.log('');
    // Step 4: Create trainer and train
    console.log('Step 4: Training model...');
    console.log('This may take a while depending on your dataset size...\n');
    const trainer = new src_1.Trainer(model, trainingConfig);
    const stats = trainer.train(dataset);
    console.log('');
    // Step 5: Display training results
    console.log('Step 5: Training results');
    console.log('------------------------');
    const finalStat = stats[stats.length - 1];
    console.log(`  Final loss: ${finalStat.loss.toFixed(6)}`);
    console.log(`  Total epochs: ${stats.length}`);
    console.log(`  Total training time: ${stats.reduce((sum, s) => sum + s.time, 0)}ms`);
    console.log('');
    // Step 6: Save model
    console.log('Step 6: Saving model...');
    const outputPath = 'examples/output/trained-model.pgm';
    (0, src_1.saveModel)(model, outputPath, {
        learningRate: trainingConfig.learningRate,
        batchSize: trainingConfig.batchSize,
        epochs: trainingConfig.epochs,
        optimizer: trainingConfig.optimizer,
    });
    console.log('');
    console.log('Training completed successfully! ✓');
    console.log(`Model saved to: ${outputPath}`);
}
main().catch(console.error);
//# sourceMappingURL=train-with-images.js.map