/**
 * Real-time Checkpointing Example
 * Demonstrates how to save model checkpoints during training
 */

import {
  PixelGenModel,
  Trainer,
  createTestDataset,
  saveModel,
  loadModel,
  inspectModel,
} from '../src';

import * as fs from 'fs';
import * as path from 'path';

async function main() {
  console.log('PixelGen - Real-time Checkpointing Example');
  console.log('==========================================\n');

  // ============================================
  // Setup: Create directories
  // ============================================
  const checkpointDir = path.join('examples', 'output', 'checkpoints');
  
  if (!fs.existsSync(checkpointDir)) {
    fs.mkdirSync(checkpointDir, { recursive: true });
  }

  // ============================================
  // Step 1: Create Model and Dataset
  // ============================================
  console.log('Step 1: Creating model and dataset');
  console.log('----------------------------------');
  
  const model = new PixelGenModel(3, 3);
  const dataset = createTestDataset(40, 3, 32, 32);
  
  console.log('✓ Model created');
  console.log(`✓ Dataset created with ${dataset.size()} images\n`);

  // ============================================
  // Step 2: Configure Training with Checkpoints
  // ============================================
  console.log('Step 2: Configuring training with checkpointing');
  console.log('------------------------------------------------');
  
  const trainingConfig = {
    epochs: 60,
    batchSize: 4,
    learningRate: 0.001,
    optimizer: 'adam' as const,
    lossFunction: 'pixelart' as const,
    verbose: true,
  };

  const trainer = new Trainer(model, trainingConfig);
  
  // Checkpoint callback - saves every 10 epochs
  const checkpointCallback = (epoch: number, loss: number, currentModel: PixelGenModel) => {
    if (epoch % 10 === 0) {
      const filename = `checkpoint-epoch-${epoch.toString().padStart(3, '0')}.pgm`;
      const filepath = path.join(checkpointDir, filename);
      
      try {
        saveModel(currentModel, filepath, {
          learningRate: trainingConfig.learningRate,
          batchSize: trainingConfig.batchSize,
          epochs: epoch,
          optimizer: trainingConfig.optimizer,
        });
        
        console.log(`💾 Checkpoint saved: ${filename}`);
        console.log(`   📁 Path: ${filepath}`);
        console.log(`   📊 Loss: ${loss.toFixed(6)}`);
        
        // Show file size
        const stats = fs.statSync(filepath);
        console.log(`   💽 Size: ${(stats.size / 1024).toFixed(2)} KB\n`);
        
      } catch (error: any) {
        console.warn(`⚠️  Failed to save checkpoint: ${error.message}\n`);
      }
    }
  };

  // Progress callback - shows progress every 5 epochs
  const progressCallback = (epoch: number, loss: number) => {
    if (epoch % 5 === 0 && epoch % 10 !== 0) { // Don't duplicate with checkpoint messages
      console.log(`📈 Progress: Epoch ${epoch}/${trainingConfig.epochs} - Loss: ${loss.toFixed(6)}`);
    }
  };

  // Combined callback
  const combinedCallback = (epoch: number, loss: number, currentModel: PixelGenModel) => {
    checkpointCallback(epoch, loss, currentModel);
    progressCallback(epoch, loss);
  };

  console.log('Training configuration:');
  console.log(JSON.stringify(trainingConfig, null, 2));
  console.log(`\nCheckpoints will be saved every 10 epochs to: ${checkpointDir}\n`);

  // ============================================
  // Step 3: Train with Real-time Checkpointing
  // ============================================
  console.log('Step 3: Training with real-time checkpointing');
  console.log('----------------------------------------------');
  console.log('This will take a few minutes. Watch for checkpoint saves!\n');
  
  const startTime = Date.now();
  const stats = trainer.train(dataset, { onEpoch: combinedCallback });
  const totalTime = Date.now() - startTime;

  console.log('\n============================================');
  console.log('Training Complete!');
  console.log('============================================');
  console.log(`Total training time: ${(totalTime / 1000).toFixed(1)}s`);
  console.log(`Final loss: ${stats[stats.length - 1].loss.toFixed(6)}\n`);

  // ============================================
  // Step 4: Analyze Saved Checkpoints
  // ============================================
  console.log('Step 4: Analyzing saved checkpoints');
  console.log('------------------------------------');
  
  const checkpointFiles = fs.readdirSync(checkpointDir)
    .filter(file => file.endsWith('.pgm'))
    .sort();

  if (checkpointFiles.length === 0) {
    console.log('No checkpoints found');
    return;
  }

  console.log(`Found ${checkpointFiles.length} checkpoints:\n`);

  const checkpointAnalysis: any[] = [];

  for (const filename of checkpointFiles) {
    const filepath = path.join(checkpointDir, filename);
    const info = inspectModel(filepath);
    
    // Extract epoch from filename
    const epochMatch = filename.match(/checkpoint-epoch-(\d+)/);
    const epoch = epochMatch ? parseInt(epochMatch[1]) : 0;
    
    // Get corresponding training stats
    const statIndex = epoch - 1; // stats array is 0-indexed
    const trainingLoss = statIndex >= 0 && statIndex < stats.length ? 
      stats[statIndex].loss : null;

    const analysis = {
      filename,
      epoch,
      trainingLoss,
      parameters: info.totalParameters,
      fileSize: info.fileSize,
      sizeMB: (info.fileSize / (1024 * 1024)).toFixed(2),
    };

    checkpointAnalysis.push(analysis);

    console.log(`📄 ${filename}`);
    console.log(`   🔢 Epoch: ${epoch}`);
    console.log(`   📊 Training Loss: ${trainingLoss ? trainingLoss.toFixed(6) : 'N/A'}`);
    console.log(`   ⚙️  Parameters: ${info.totalParameters.toLocaleString()}`);
    console.log(`   💾 File Size: ${analysis.sizeMB} MB\n`);
  }

  // ============================================
  // Step 5: Demonstrate Loading Checkpoints
  // ============================================
  console.log('Step 5: Loading and testing checkpoints');
  console.log('---------------------------------------');

  // Load the best checkpoint (lowest loss)
  const bestCheckpoint = checkpointAnalysis
    .filter(c => c.trainingLoss !== null)
    .sort((a, b) => a.trainingLoss! - b.trainingLoss!)[0];

  if (bestCheckpoint) {
    console.log(`Loading best checkpoint: ${bestCheckpoint.filename}`);
    console.log(`Best loss: ${bestCheckpoint.trainingLoss!.toFixed(6)}\n`);
    
    const bestPath = path.join(checkpointDir, bestCheckpoint.filename);
    const { model: loadedModel, hyperparameters } = loadModel(bestPath);
    
    console.log('✓ Checkpoint loaded successfully');
    console.log('Hyperparameters from checkpoint:');
    console.log(JSON.stringify(hyperparameters, null, 2));
    
    // Quick test inference
    const testInput = dataset.getBatch(1, 0);
    if (testInput) {
      const output = loadedModel.forward(testInput);
      console.log(`\nInference test:`);
      console.log(`  Input shape: [${testInput.shape.join(', ')}]`);
      console.log(`  Output shape: [${output.shape.join(', ')}]`);
      
      // Calculate reconstruction error
      let mse = 0;
      for (let i = 0; i < testInput.data.length; i++) {
        const diff = testInput.data[i] - output.data[i];
        mse += diff * diff;
      }
      mse /= testInput.data.length;
      
      console.log(`  Reconstruction MSE: ${mse.toFixed(6)}`);
    }
  }

  // ============================================
  // Summary
  // ============================================
  console.log('\n============================================');
  console.log('Summary');
  console.log('============================================');
  console.log(`✓ Trained for ${trainingConfig.epochs} epochs`);
  console.log(`✓ Saved ${checkpointFiles.length} checkpoints`);
  console.log(`✓ Checkpoints directory: ${checkpointDir}`);
  console.log(`✓ Best model at epoch ${bestCheckpoint?.epoch} with loss ${bestCheckpoint?.trainingLoss?.toFixed(6)}`);
  console.log('\nCheckpointing allows you to:');
  console.log('• Resume training from any point');
  console.log('• Compare model performance over time');
  console.log('• Choose the best performing checkpoint');
  console.log('• Never lose training progress');
  console.log('\nReal-time checkpointing example completed! ✓');
}

main().catch(console.error);