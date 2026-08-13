"use strict";
/**
 * Advanced PixelGen usage example
 * Demonstrates model evaluation, fine-tuning, and model inspection
 */
Object.defineProperty(exports, "__esModule", { value: true });
const src_1 = require("../src");
async function main() {
    console.log('PixelGen - Advanced Usage Example');
    console.log('==================================\n');
    // ============================================
    // Part 1: Train Initial Model
    // ============================================
    console.log('Part 1: Training initial model');
    console.log('------------------------------');
    const model = new src_1.PixelGenModel(3, 3);
    const trainDataset = (0, src_1.createTestDataset)(30, 3, 32, 32);
    const trainer = new src_1.Trainer(model, {
        epochs: 20,
        batchSize: 4,
        learningRate: 0.001,
        optimizer: 'adam',
        lossFunction: 'pixelart',
        verbose: false,
    });
    console.log('Training...');
    const initialStats = trainer.train(trainDataset);
    console.log(`Initial training complete!`);
    console.log(`Final loss: ${initialStats[initialStats.length - 1].loss.toFixed(6)}`);
    console.log('');
    // ============================================
    // Part 2: Save and Inspect Model
    // ============================================
    console.log('Part 2: Saving and inspecting model');
    console.log('------------------------------------');
    const modelPath = 'examples/output/advanced-model.pgm';
    (0, src_1.saveModel)(model, modelPath, {
        learningRate: 0.001,
        batchSize: 4,
        epochs: 20,
        optimizer: 'adam',
    });
    // Inspect without loading weights (efficient)
    const info = (0, src_1.inspectModel)(modelPath);
    console.log('\nModel Information:');
    console.log(`  Architecture: ${info.metadata.architecture}`);
    console.log(`  Version: ${info.metadata.version}`);
    console.log(`  Parameters: ${info.totalParameters.toLocaleString()}`);
    console.log(`  File size: ${(info.fileSize / 1024).toFixed(2)} KB`);
    console.log(`  Layers: ${info.metadata.layers.length}`);
    console.log('');
    // ============================================
    // Part 3: Load Model and Evaluate
    // ============================================
    console.log('Part 3: Loading and evaluating model');
    console.log('-------------------------------------');
    const { model: loadedModel } = (0, src_1.loadModel)(modelPath);
    // Create test dataset for evaluation
    const testDataset = (0, src_1.createTestDataset)(10, 3, 32, 32);
    const evaluator = new src_1.Trainer(loadedModel, {
        epochs: 1,
        batchSize: 4,
        learningRate: 0.001,
        optimizer: 'adam',
        lossFunction: 'pixelart',
        verbose: false,
    });
    const testLoss = evaluator.evaluate(testDataset);
    console.log(`Test loss: ${testLoss.toFixed(6)}`);
    console.log('');
    // ============================================
    // Part 4: Fine-tuning
    // ============================================
    console.log('Part 4: Fine-tuning on new data');
    console.log('--------------------------------');
    // Simulate new data for fine-tuning
    const finetuneDataset = (0, src_1.createTestDataset)(15, 3, 32, 32);
    const finetuneTrainer = new src_1.Trainer(loadedModel, {
        epochs: 10,
        batchSize: 2,
        learningRate: 0.0001, // Lower learning rate for fine-tuning
        optimizer: 'adam',
        lossFunction: 'pixelart',
        verbose: false,
    });
    console.log('Fine-tuning...');
    const finetuneStats = finetuneTrainer.train(finetuneDataset);
    console.log(`Fine-tuning complete!`);
    console.log(`Final loss: ${finetuneStats[finetuneStats.length - 1].loss.toFixed(6)}`);
    console.log('');
    // Save fine-tuned model
    const finetunedPath = 'examples/output/advanced-model-finetuned.pgm';
    (0, src_1.saveModel)(loadedModel, finetunedPath, {
        learningRate: 0.0001,
        batchSize: 2,
        epochs: 10,
        optimizer: 'adam',
    });
    console.log(`Fine-tuned model saved to ${finetunedPath}`);
    console.log('');
    // ============================================
    // Part 5: Model Inference
    // ============================================
    console.log('Part 5: Running inference');
    console.log('-------------------------');
    // Create a single test image
    const testImage = new src_1.Tensor(new Float32Array(3 * 32 * 32).map(() => Math.random()), [1, 3, 32, 32]);
    console.log('Input image shape:', testImage.shape);
    // Run forward pass
    const output = loadedModel.forward(testImage);
    console.log('Output image shape:', output.shape);
    // Get some statistics
    let min = Infinity, max = -Infinity, sum = 0;
    for (let i = 0; i < output.data.length; i++) {
        const val = output.data[i];
        if (val < min)
            min = val;
        if (val > max)
            max = val;
        sum += val;
    }
    const mean = sum / output.data.length;
    console.log('Output statistics:');
    console.log(`  Min: ${min.toFixed(4)}`);
    console.log(`  Max: ${max.toFixed(4)}`);
    console.log(`  Mean: ${mean.toFixed(4)}`);
    console.log('');
    // ============================================
    // Part 6: Training Progress Analysis
    // ============================================
    console.log('Part 6: Training progress analysis');
    console.log('-----------------------------------');
    console.log('\nInitial Training Loss Progression:');
    const milestones = [0, 4, 9, 14, 19]; // Show epochs 1, 5, 10, 15, 20
    for (const idx of milestones) {
        if (idx < initialStats.length) {
            const stat = initialStats[idx];
            console.log(`  Epoch ${stat.epoch}: ${stat.loss.toFixed(6)} (${stat.time}ms)`);
        }
    }
    console.log('\nFine-tuning Loss Progression:');
    for (let i = 0; i < Math.min(5, finetuneStats.length); i++) {
        const stat = finetuneStats[i];
        console.log(`  Epoch ${stat.epoch}: ${stat.loss.toFixed(6)} (${stat.time}ms)`);
    }
    console.log('');
    // ============================================
    // Summary
    // ============================================
    console.log('Summary');
    console.log('-------');
    console.log('✓ Trained initial model (20 epochs)');
    console.log('✓ Saved and inspected model');
    console.log('✓ Evaluated on test data');
    console.log('✓ Fine-tuned on new data (10 epochs)');
    console.log('✓ Ran inference on test image');
    console.log('✓ Analyzed training progress');
    console.log('');
    console.log('Advanced example completed successfully! ✓');
}
main().catch(console.error);
//# sourceMappingURL=advanced-usage.js.map