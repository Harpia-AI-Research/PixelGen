# Quick Start Guide

This guide will walk you through training your first PixelGen model in under 5 minutes.

## Option 1: Training with Test Data

Perfect for understanding how PixelGen works before using real images.

### Step 1: Install PixelGen

```bash
npm install pixelgen
```

### Step 2: Create a Training Script

Create a new file `my-first-model.js`:

```javascript
const {
  PixelGenModel,
  Trainer,
  createTestDataset,
  saveModel,
} = require('pixelgen');

// Create model
console.log('Creating model...');
const model = new PixelGenModel(3, 3);
console.log('✓ Model created');

// Create synthetic dataset
console.log('Generating test dataset...');
const dataset = createTestDataset(20, 3, 32, 32);
console.log(`✓ Created ${dataset.size()} test images`);

// Configure training
const trainer = new Trainer(model, {
  epochs: 20,
  batchSize: 4,
  learningRate: 0.001,
  optimizer: 'adam',
  lossFunction: 'pixelart',
  verbose: true,
});

// Train
console.log('\nTraining...\n');
const stats = trainer.train(dataset);

// Display results
const finalLoss = stats[stats.length - 1].loss;
console.log(`\n✓ Training complete!`);
console.log(`Final loss: ${finalLoss.toFixed(6)}`);

// Save model
saveModel(model, 'my-first-model.pgm', {
  learningRate: 0.001,
  batchSize: 4,
  epochs: 20,
  optimizer: 'adam',
});

console.log('✓ Model saved to my-first-model.pgm');
```

### Step 2: Run the Script

```bash
node my-first-model.js
```

### Expected Output

```
Creating model...
✓ Model created
Generating test dataset...
✓ Created 20 test images

Training...

Epoch 1/20 - Loss: 0.234567 - Time: 1234ms
Epoch 2/20 - Loss: 0.198432 - Time: 1198ms
Epoch 3/20 - Loss: 0.167821 - Time: 1205ms
...
Epoch 20/20 - Loss: 0.045123 - Time: 1211ms

✓ Training complete!
Final loss: 0.045123
✓ Model saved to my-first-model.pgm
```

**Congratulations!** You've trained your first PixelGen model.

---

## Option 2: Training with Real Images

Now let's train on actual PixelArt images.

### Step 1: Prepare Your Dataset

Create a folder structure:

```
PixelGen/
└── dataset/
    ├── sprite1.png
    ├── sprite2.png
    ├── sprite3.png
    └── ...
```

**Requirements:**
- At least 20 images (more is better)
- PNG or JPEG format
- Any size (will be resized to 32×32)
- PixelArt style recommended

**Where to get PixelArt:**
- Create your own in Aseprite, Piskel, or similar
- Use royalty-free sprites from OpenGameArt.org
- Download free sprite packs
- Take screenshots from retro games (for personal use only)

### Step 2: Create Training Script

Create `train-with-images.js`:

```javascript
const {
  PixelGenModel,
  Trainer,
  ImageDataset,
  saveModel,
} = require('pixelgen');

// Create model
const model = new PixelGenModel(3, 3);

// Load images
console.log('Loading images from dataset/...');
const dataset = new ImageDataset({
  targetSize: 32,
  normalize: true,
  channels: 3,
});

try {
  dataset.loadFromDirectory('./dataset');
} catch (error) {
  console.error('Error loading dataset:', error.message);
  console.log('\nMake sure you have:');
  console.log('1. Created a "dataset" folder');
  console.log('2. Added PNG or JPG images to it');
  process.exit(1);
}

console.log(`✓ Loaded ${dataset.size()} images\n`);

// Configure training
const trainer = new Trainer(model, {
  epochs: 50,
  batchSize: 4,
  learningRate: 0.001,
  optimizer: 'adam',
  lossFunction: 'pixelart',
  verbose: true,
});

// Train
console.log('Training...\n');
const stats = trainer.train(dataset);

// Save
const finalLoss = stats[stats.length - 1].loss;
console.log(`\n✓ Training complete! Final loss: ${finalLoss.toFixed(6)}`);

saveModel(model, 'pixelart-model.pgm', {
  learningRate: 0.001,
  batchSize: 4,
  epochs: 50,
  optimizer: 'adam',
});

console.log('✓ Model saved to pixelart-model.pgm');
```

### Step 3: Run Training

```bash
node train-with-images.js
```

### Expected Output

```
Loading images from dataset/...
Found 50 images in ./dataset
Successfully loaded 50 images
✓ Loaded 50 images

Training...

Epoch 1/50 - Loss: 0.456789 - Time: 2345ms
Epoch 2/50 - Loss: 0.398765 - Time: 2312ms
...
Epoch 50/50 - Loss: 0.067891 - Time: 2298ms

✓ Training complete! Final loss: 0.067891
Model saved to pixelart-model.pgm
✓ Model saved to pixelart-model.pgm
```

---

## Understanding the Output

### Training Metrics

Each epoch shows:
- **Epoch X/Y**: Current epoch out of total
- **Loss**: How different the output is from the input (lower is better)
- **Time**: Milliseconds for this epoch

### Good Loss Values

| Loss | Interpretation |
|------|---------------|
| > 0.3 | Early training, model still learning basics |
| 0.1 - 0.3 | Model learning patterns |
| 0.05 - 0.1 | Good reconstruction quality |
| < 0.05 | Excellent reconstruction |

### Model File

The `.pgm` file contains:
- Trained weights (all learned parameters)
- Model architecture metadata
- Training hyperparameters
- Parameter shapes

Typical size: 0.5 - 2 MB

---

## Loading and Using a Trained Model

### Load Model

```javascript
const { loadModel, Tensor } = require('pixelgen');

// Load trained model
const { model, hyperparameters } = loadModel('pixelart-model.pgm');

console.log('Model loaded!');
console.log('Architecture:', hyperparameters);
```

### Run Inference

```typescript
// Prepare input (32×32 RGB image as flat array)
const imageData = new Float32Array(32 * 32 * 3);
// ... fill with image data (normalized 0-1) ...

// Create tensor [1, 3, 32, 32]
const input = new Tensor(imageData, [1, 3, 32, 32]);

// Forward pass
const output = model.forward(input);

// output.data contains the generated image (Float32Array)
// output.shape is [1, 3, 32, 32]
```

---

## Next Steps

### Experiment with Hyperparameters

Try changing these values to see how they affect training:

```typescript
const trainer = new Trainer(model, {
  epochs: 100,        // Try 20, 50, 100, 200
  batchSize: 8,       // Try 2, 4, 8, 16
  learningRate: 0.001, // Try 0.0001, 0.001, 0.01
  optimizer: 'adam',   // Try 'sgd' or 'adam'
  lossFunction: 'pixelart', // Try 'mse' or 'pixelart'
  verbose: true,
});
```

**Guidelines:**
- **More epochs**: Better learning but takes longer
- **Larger batches**: Faster but uses more memory
- **Smaller learning rate**: More stable but slower
- **'pixelart' loss**: Better for sharp edges
- **'adam' optimizer**: Usually better than SGD

### Advanced Topics

Continue learning:

- **[Core Concepts](04-CoreConcepts.md)**: Understand how it works
- **[Training Guide](06-Training-Guide.md)**: Master hyperparameters
- **[API Reference](05-API-Reference.md)**: Explore all functions
- **[Advanced Usage](08-Advanced-Usage.md)**: Fine-tuning, evaluation

### Build Something Cool

Ideas for projects:
1. Train on your game's sprites to learn your art style
2. Create variations of existing sprites
3. Restore old, corrupted PixelArt
4. Learn patterns from retro game screenshots
5. Build a sprite generation tool

---

[← Installation](02-Installation.md) | [Core Concepts →](04-CoreConcepts.md)
