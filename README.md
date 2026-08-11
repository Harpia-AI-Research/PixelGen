<div align="center">
  <img src="Assets/PixelGenLogo.png" alt="PixelGen Logo" width="400">
  
  <h1>PixelGen</h1>
  <p><strong>A Ground-Up Machine Learning Architecture for PixelArt Generation</strong></p>
  
  [![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
  [![TypeScript](https://img.shields.io/badge/TypeScript-5.0+-blue.svg)](https://www.typescriptlang.org/)
  [![Node](https://img.shields.io/badge/Node.js-18.0+-green.svg)](https://nodejs.org/)
</div>

---

## Overview

**PixelGen** is a specialized machine learning architecture designed exclusively for PixelArt generation and transformation. Unlike generic ML frameworks, PixelGen is built from the ground up to understand and produce pixel-perfect artwork.

The entire stack—from tensor operations to neural network layers—is implemented in pure TypeScript without external ML dependencies, providing full transparency and control over the learning process.

### Key Differentiators

- **Domain-Specific Design**: Architecture optimized for PixelArt's unique characteristics (discrete pixels, limited palettes, sharp edges)
- **Zero ML Dependencies**: Complete implementation from scratch—no TensorFlow, PyTorch, or other frameworks
- **CPU-First Approach**: Efficient training on commodity hardware without GPU requirements
- **Transparent Learning**: Full visibility into model internals and training dynamics
- **Custom File Format**: `.pgm` (Pixel Gen Model) format designed specifically for PixelArt models

---

## Architecture

### Neural Network Design

PixelGen implements a **convolutional autoencoder** architecture that learns compressed representations of PixelArt:

```
Input (32×32×3 RGB)
        ↓
    ┌─────────┐
    │ ENCODER │  
    └─────────┘
        ↓
   Conv2D(3→32) + ReLU
   MaxPool(2×2) → 16×16
        ↓
   Conv2D(32→64) + ReLU
   MaxPool(2×2) → 8×8
        ↓
   Conv2D(64→128) + ReLU
        ↓
    ┌───────────┐
    │ BOTTLENECK│ (8×8×128)
    └───────────┘
        ↓
    ┌─────────┐
    │ DECODER │
    └─────────┘
        ↓
   Upsample(2×) → 16×16
   Conv2D(128→64) + ReLU
        ↓
   Upsample(2×) → 32×32
   Conv2D(64→32) + ReLU
        ↓
   Conv2D(32→3)
        ↓
Output (32×32×3 RGB)
```

### Core Components

#### 1. Tensor Engine

A complete n-dimensional array implementation with automatic differentiation:

- **Data Structure**: `Float32Array`-backed tensors for optimal performance
- **Operations**: Matrix multiplication, element-wise ops, reshaping, transposition
- **Autograd**: Computational graph-based backpropagation with topological sorting
- **Memory Efficient**: In-place operations where safe, minimal allocations

#### 2. Neural Network Layers

**Convolutional Layer (Conv2D)**
- Configurable kernel size, stride, and padding
- Xavier/Glorot weight initialization for training stability
- Efficient convolution implementation for small images

**Pooling Layers**
- MaxPool2D: Maximum value selection for downsampling
- AvgPool2D: Average pooling for smooth reduction
- Upsample2D: Nearest-neighbor upsampling for decoder

**Dense Layer**
- Fully-connected layer for feature transformation
- Bias terms with automatic gradient computation

#### 3. Optimization

**Adam Optimizer** (Primary)
- Adaptive learning rates per parameter
- Momentum and velocity accumulation
- Bias correction for initial timesteps
- Default: `lr=0.001, β₁=0.9, β₂=0.999`

**SGD with Momentum**
- Classical gradient descent with momentum term
- Configurable learning rate and momentum factor

#### 4. Loss Functions

**PixelArt Loss** (Recommended)
```
L_total = L_MSE + λ·L_edge
```
- **L_MSE**: Mean squared error for pixel reconstruction
- **L_edge**: Edge preservation penalty for sharp transitions
- **λ**: Edge weight (default: 0.3)

This custom loss function encourages the model to maintain the crisp boundaries characteristic of PixelArt.

**Standard Losses**
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Binary Cross-Entropy (BCE)

---

## Installation

### Prerequisites

- Node.js ≥ 18.0.0
- npm or yarn package manager
- 4GB+ RAM recommended for training

### Setup

```bash
# Clone the repository
git clone https://github.com/Harpia-AI-Research/PixelGen.git
cd PixelGen

# Install dependencies
npm install

# Build the project
npm run build
```

---

## Quick Start

### Training with Test Data

For initial experimentation, use the built-in test dataset generator:

```typescript
import {
  PixelGenModel,
  Trainer,
  createTestDataset,
  saveModel,
} from 'pixelgen';

// Initialize model
const model = new PixelGenModel(3, 3); // RGB input → RGB output

// Generate synthetic dataset
const dataset = createTestDataset(50, 3, 32, 32);

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
const stats = trainer.train(dataset);

// Save trained model
saveModel(model, 'model.pgm', {
  learningRate: 0.001,
  batchSize: 4,
  epochs: 50,
  optimizer: 'adam',
});
```

### Training with Real Images

Load your own PixelArt sprites, tiles, or scenes:

```typescript
import {
  PixelGenModel,
  Trainer,
  ImageDataset,
  saveModel,
} from 'pixelgen';

// Initialize model
const model = new PixelGenModel(3, 3);

// Load images from directory
const dataset = new ImageDataset({
  targetSize: 32,      // Resize to 32×32
  normalize: true,     // Normalize to [0,1]
  channels: 3,         // RGB
});

dataset.loadFromDirectory('./my-pixelart');

// Train
const trainer = new Trainer(model, {
  epochs: 100,
  batchSize: 8,
  learningRate: 0.0005,
  optimizer: 'adam',
  lossFunction: 'pixelart',
  verbose: true,
});

trainer.train(dataset);
saveModel(model, 'trained-model.pgm', { /* ... */ });
```

### Inference

```typescript
import { loadModel, Tensor } from 'pixelgen';

// Load trained model
const { model } = loadModel('trained-model.pgm');

// Prepare input (32×32×3 image as tensor)
const input = new Tensor(imageData, [1, 3, 32, 32]);

// Generate output
const output = model.forward(input);

// Output is a tensor [1, 3, 32, 32] ready for visualization
```

---

## Project Structure

```
pixelgen/
├── src/
│   ├── core/              # Foundation layer
│   │   ├── Tensor.ts      # N-dimensional array with autograd
│   │   └── activations.ts # ReLU, Sigmoid, Tanh, etc.
│   │
│   ├── layers/            # Neural network building blocks
│   │   ├── Conv2D.ts      # 2D convolution
│   │   ├── Dense.ts       # Fully-connected layer
│   │   └── Pooling.ts     # MaxPool, AvgPool, Upsample
│   │
│   ├── model/             # Model architecture
│   │   └── PixelGenModel.ts
│   │
│   ├── training/          # Training infrastructure
│   │   ├── Optimizer.ts   # SGD, Adam
│   │   ├── Loss.ts        # Loss functions
│   │   └── Trainer.ts     # Training loop
│   │
│   ├── data/              # Data loading
│   │   └── ImageLoader.ts # PNG/JPEG parsing
│   │
│   └── io/                # Model persistence
│       └── ModelIO.ts     # .pgm format handler
│
├── examples/              # Usage examples
│   ├── basic-training.ts
│   ├── advanced-usage.ts
│   └── train-with-images.ts
│
├── docs/                  # Documentation
│   ├── ARCHITECTURE.md    # Technical architecture
│   └── GETTING_STARTED.md # Tutorial
│
└── VISION/                # Project vision
    └── README.md
```

---

## Model File Format (.pgm)

PixelGen models are saved in a custom binary format optimized for PixelArt models:

### Structure

```
[JSON Header]
---WEIGHTS---
[Binary Float32Array]
```

### Header Schema

```json
{
  "metadata": {
    "version": "1.0.0",
    "architecture": "pixelgen-autoencoder-v1",
    "inputChannels": 3,
    "outputChannels": 3,
    "inputSize": 32,
    "layers": [...]
  },
  "hyperparameters": {
    "learningRate": 0.001,
    "batchSize": 4,
    "epochs": 50,
    "optimizer": "adam"
  },
  "parameterShapes": [[32,3,3,3], ...],
  "totalParameters": 123456
}
```

### Advantages

- **Compact**: Binary weight storage
- **Self-Documenting**: JSON metadata for model inspection
- **Fast Loading**: Direct Float32Array deserialization
- **Complete**: All information needed to reconstruct the model

---

## Examples

### Run Built-in Examples

```bash
# Basic training with synthetic data
npm run example:basic

# Advanced usage (fine-tuning, evaluation)
npm run example:advanced

# Training with real images
npm run example:images
```

### Example Output

```
PixelGen - Training Example
===========================

Step 1: Creating model...
Model created with 12 parameter tensors

Step 2: Loading images...
Found 50 images in ./dataset
Successfully loaded 50 images

Step 3: Training model...
Epoch 1/50 - Loss: 0.234567 - Time: 1234ms
Epoch 2/50 - Loss: 0.198432 - Time: 1198ms
...
Epoch 50/50 - Loss: 0.023456 - Time: 1211ms

Training complete!
Model saved to: model.pgm (1.2 MB)
```

---

## Performance Considerations

### Training Speed

| Dataset Size | Epochs | Batch Size | Time (CPU) |
|--------------|--------|------------|------------|
| 20 images    | 50     | 4          | ~2 minutes |
| 100 images   | 100    | 8          | ~15 minutes|
| 500 images   | 200    | 16         | ~2 hours   |

*Measured on Intel i5-10400 @ 2.9GHz*

### Memory Usage

- **Model**: ~5 MB in memory (128k parameters)
- **Training**: ~50 MB per batch of 8 images
- **Minimum RAM**: 2GB
- **Recommended RAM**: 4GB+

### Optimization Tips

1. **Batch Size**: Larger batches are faster but use more memory
2. **Image Count**: Start with 50-100 images to validate approach
3. **Epochs**: Monitor loss—stop early if plateaus
4. **Learning Rate**: Reduce if loss oscillates, increase if learning is slow

---

## Technical Details

### Autograd Implementation

PixelGen uses reverse-mode automatic differentiation:

1. **Forward Pass**: Build computational graph
2. **Backward Pass**: Traverse graph in reverse topological order
3. **Gradient Accumulation**: Apply chain rule at each node
4. **Parameter Update**: Use optimizer to update weights

### Initialization Strategy

- **Weights**: Xavier/Glorot initialization (`U(-√(6/(fan_in+fan_out)), √(6/(fan_in+fan_out)))`)
- **Biases**: Zero initialization
- **Rationale**: Prevents vanishing/exploding gradients in deep networks

### Numerical Stability

- **Softmax**: Max subtraction before exp for numerical stability
- **Adam Epsilon**: 1e-8 added to denominator to prevent division by zero
- **Gradient Clipping**: Not implemented (models are shallow enough)

---

## API Reference

### Core Classes

#### `Tensor`
```typescript
class Tensor {
  constructor(data: number[] | Float32Array, shape: number[], requiresGrad?: boolean)
  
  // Operations
  add(other: Tensor | number): Tensor
  multiply(other: Tensor | number): Tensor
  matmul(other: Tensor): Tensor
  reshape(newShape: number[]): Tensor
  transpose(): Tensor
  
  // Autograd
  backward(): void
  zeroGrad(): void
  
  // Factory methods
  static zeros(shape: number[]): Tensor
  static ones(shape: number[]): Tensor
  static random(shape: number[]): Tensor
  static xavier(shape: number[]): Tensor
}
```

#### `PixelGenModel`
```typescript
class PixelGenModel {
  constructor(inputChannels: number, outputChannels: number)
  
  forward(input: Tensor): Tensor
  parameters(): Tensor[]
  zeroGrad(): void
  getMetadata(): ModelMetadata
}
```

#### `Trainer`
```typescript
interface TrainingConfig {
  epochs: number
  batchSize: number
  learningRate: number
  optimizer: 'sgd' | 'adam'
  lossFunction: 'mse' | 'pixelart'
  verbose: boolean
}

class Trainer {
  constructor(model: PixelGenModel, config: TrainingConfig)
  
  train(dataset: ImageDataset): TrainingStats[]
  evaluate(dataset: ImageDataset): number
}
```

#### `ImageDataset`
```typescript
interface DatasetConfig {
  targetSize: number
  normalize: boolean
  channels: number
}

class ImageDataset {
  constructor(config: DatasetConfig)
  
  loadFromDirectory(path: string): void
  addTensor(tensor: Tensor): void
  getBatch(batchSize: number, startIdx?: number): Tensor | null
  shuffle(): void
  size(): number
}
```

---

## Limitations & Future Work

### Current Limitations

- **Fixed Input Size**: 32×32 pixels only
- **RGB Only**: No alpha channel support
- **CPU Only**: No GPU acceleration
- **Single Architecture**: Fixed autoencoder (not customizable)
- **English Documentation**: Limited multilingual support

### Planned Enhancements

- Variable input sizes (16×16, 64×64, 128×128)
- RGBA support with alpha channel handling
- WebAssembly SIMD for faster computation
- Customizable architectures (define your own layers)
- CLI tools for training and generation
- Model zoo with pre-trained models

---

## Documentation

- **[Getting Started](docs/GETTING_STARTED.md)**: Step-by-step tutorial
- **[Architecture Guide](docs/ARCHITECTURE.md)**: Deep technical overview
- **[Project Vision](VISION/README.md)**: Philosophy and design principles
- **[Contributing](CONTRIBUTING.md)**: How to contribute
- **[Changelog](CHANGELOG.md)**: Version history

---

## Citation

If you use PixelGen in your research or project, please cite:

```bibtex
@software{pixelgen2026,
  title = {PixelGen: A Ground-Up Machine Learning Architecture for PixelArt},
  author = {Harpia AI Research},
  year = {2026},
  url = {https://github.com/Harpia-AI-Research/PixelGen}
}
```

---

## License

This project is licensed under the **BSD 3-Clause License** - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2026, Harpia AI Research

---

## Acknowledgments

PixelGen is an independent research project exploring domain-specific machine learning architectures. Built with transparency and education in mind.

**Technologies**: TypeScript, Node.js, pngjs, jpeg-js

---

<div align="center">
  <strong>PixelGen</strong> — Crafting PixelArt through Machine Learning
  <br><br>
  <a href="https://github.com/Harpia-AI-Research/PixelGen">GitHub</a> •
  <a href="docs/GETTING_STARTED.md">Tutorial</a> •
  <a href="docs/ARCHITECTURE.md">Architecture</a> •
  <a href="CONTRIBUTING.md">Contributing</a>
</div>
