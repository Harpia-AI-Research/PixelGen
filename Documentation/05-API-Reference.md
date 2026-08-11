# API Reference

Complete reference for all PixelGen classes, functions, and types.

## Table of Contents

- [Core](#core)
  - [Tensor](#tensor)
  - [Activation Functions](#activation-functions)
- [Layers](#layers)
  - [Conv2D](#conv2d)
  - [Dense](#dense)
  - [MaxPool2D](#maxpool2d)
  - [AvgPool2D](#avgpool2d)
  - [Upsample2D](#upsample2d)
- [Model](#model)
  - [PixelGenModel](#pixelgenmodel)
- [Training](#training)
  - [Trainer](#trainer)
  - [Optimizers](#optimizers)
  - [Loss Functions](#loss-functions-1)
- [Data](#data)
  - [ImageDataset](#imagedataset)
  - [Utility Functions](#utility-functions)
- [I/O](#io)
  - [Model Persistence](#model-persistence)

---

## Core

### Tensor

N-dimensional array with automatic differentiation support.

#### Constructor

```typescript
new Tensor(
  data: number[] | Float32Array | number[][],
  shape?: number[],
  requiresGrad?: boolean
)
```

**Parameters:**
- `data`: Array data (nested arrays will be flattened)
- `shape`: Optional shape (inferred if not provided)
- `requiresGrad`: Enable gradient tracking (default: `false`)

**Example:**
```typescript
const t1 = new Tensor([1, 2, 3, 4], [2, 2]);
const t2 = new Tensor([[1, 2], [3, 4]]); // shape inferred
const t3 = new Tensor([1, 2], [2], true); // with gradients
```

#### Properties

```typescript
tensor.data: Float32Array      // Raw data
tensor.shape: number[]          // Shape
tensor.grad?: Tensor            // Gradients (if requiresGrad)
tensor.requiresGrad: boolean    // Gradient tracking flag
```

#### Methods

##### Element Access

```typescript
tensor.get(...indices: number[]): number
tensor.set(value: number, ...indices: number[]): void
```

**Example:**
```typescript
const t = new Tensor([1, 2, 3, 4], [2, 2]);
console.log(t.get(0, 1)); // 2
t.set(5, 1, 0);
console.log(t.get(1, 0)); // 5
```

##### Shape Operations

```typescript
tensor.reshape(newShape: number[]): Tensor
tensor.transpose(): Tensor  // 2D only
tensor.clone(): Tensor
```

**Example:**
```typescript
const t = new Tensor([1, 2, 3, 4, 5, 6], [2, 3]);
const reshaped = t.reshape([3, 2]);
const transposed = t.transpose(); // [3, 2]
```

##### Arithmetic Operations

```typescript
tensor.add(other: Tensor | number): Tensor
tensor.multiply(other: Tensor | number): Tensor
tensor.matmul(other: Tensor): Tensor  // Matrix multiplication (2D only)
```

**Example:**
```typescript
const a = new Tensor([1, 2, 3], [3]);
const b = new Tensor([4, 5, 6], [3]);

const sum = a.add(b);        // [5, 7, 9]
const product = a.multiply(2); // [2, 4, 6]
```

##### Autograd

```typescript
tensor.backward(): void
tensor.zeroGrad(): void
```

**Example:**
```typescript
const x = new Tensor([2], [1], true);
const y = x.multiply(x); // y = x²

y.backward();
console.log(x.grad); // dy/dx = 2x = 4
```

#### Static Factory Methods

```typescript
Tensor.zeros(shape: number[], requiresGrad?: boolean): Tensor
Tensor.ones(shape: number[], requiresGrad?: boolean): Tensor
Tensor.random(shape: number[], requiresGrad?: boolean): Tensor
Tensor.xavier(shape: number[], requiresGrad?: boolean): Tensor
Tensor.zerosLike(tensor: Tensor): Tensor
Tensor.onesLike(tensor: Tensor): Tensor
```

**Example:**
```typescript
const zeros = Tensor.zeros([3, 3]);
const ones = Tensor.ones([2, 2], true);
const random = Tensor.random([5]);
const weights = Tensor.xavier([64, 32], true);
```

---

### Activation Functions

```typescript
import { relu, leakyRelu, sigmoid, tanh, softmax } from 'pixelgen';
```

#### relu

```typescript
function relu(input: Tensor): Tensor
```

ReLU activation: `f(x) = max(0, x)`

#### leakyRelu

```typescript
function leakyRelu(input: Tensor, alpha?: number): Tensor
```

Leaky ReLU: `f(x) = x if x > 0 else α*x`

Default `alpha = 0.01`

#### sigmoid

```typescript
function sigmoid(input: Tensor): Tensor
```

Sigmoid: `f(x) = 1 / (1 + e^(-x))`

#### tanh

```typescript
function tanh(input: Tensor): Tensor
```

Hyperbolic tangent: `f(x) = tanh(x)`

#### softmax

```typescript
function softmax(input: Tensor): Tensor
```

Softmax (for classification)

---

## Layers

### Conv2D

2D Convolutional layer.

#### Constructor

```typescript
new Conv2D(
  inChannels: number,
  outChannels: number,
  kernelSize?: number,
  stride?: number,
  padding?: number
)
```

**Parameters:**
- `inChannels`: Input channels
- `outChannels`: Output channels (number of filters)
- `kernelSize`: Kernel size (default: 3)
- `stride`: Stride (default: 1)
- `padding`: Padding (default: 1)

**Example:**
```typescript
const conv = new Conv2D(3, 64, 3, 1, 1);
// Input: 3 channels (RGB)
// Output: 64 feature maps
// 3×3 kernel, stride 1, padding 1
```

#### Methods

```typescript
conv.forward(input: Tensor): Tensor
conv.parameters(): Tensor[]
conv.zeroGrad(): void
```

**Shape:**
- Input: `[batch, inChannels, height, width]`
- Output: `[batch, outChannels, outHeight, outWidth]`

---

### Dense

Fully-connected (dense) layer.

#### Constructor

```typescript
new Dense(inputSize: number, outputSize: number)
```

#### Methods

```typescript
dense.forward(input: Tensor): Tensor
dense.parameters(): Tensor[]
dense.zeroGrad(): void
```

**Shape:**
- Input: `[batch, inputSize]`
- Output: `[batch, outputSize]`

---

### MaxPool2D

Max pooling layer.

#### Constructor

```typescript
new MaxPool2D(poolSize?: number, stride?: number)
```

**Defaults:** `poolSize = 2`, `stride = poolSize`

#### Methods

```typescript
pool.forward(input: Tensor): Tensor
pool.parameters(): Tensor[]  // Always []
pool.zeroGrad(): void
```

---

### AvgPool2D

Average pooling layer.

#### Constructor

```typescript
new AvgPool2D(poolSize?: number, stride?: number)
```

Same as MaxPool2D but takes average instead of maximum.

---

### Upsample2D

Nearest-neighbor upsampling.

#### Constructor

```typescript
new Upsample2D(scale?: number)
```

**Default:** `scale = 2`

#### Methods

```typescript
upsample.forward(input: Tensor): Tensor
```

---

## Model

### PixelGenModel

Main autoencoder model for PixelArt.

#### Constructor

```typescript
new PixelGenModel(
  inputChannels?: number,
  outputChannels?: number
)
```

**Defaults:** Both default to 3 (RGB)

**Example:**
```typescript
const model = new PixelGenModel(3, 3);
```

#### Methods

##### forward

```typescript
model.forward(input: Tensor): Tensor
```

Run forward pass.

**Shape:**
- Input: `[batch, 3, 32, 32]`
- Output: `[batch, 3, 32, 32]`

##### parameters

```typescript
model.parameters(): Tensor[]
```

Get all trainable parameters.

##### zeroGrad

```typescript
model.zeroGrad(): void
```

Zero all gradients.

##### getMetadata

```typescript
model.getMetadata(): ModelMetadata
```

Get model configuration and architecture info.

**Returns:**
```typescript
interface ModelMetadata {
  version: string
  architecture: string
  inputChannels: number
  outputChannels: number
  inputSize: number
  layers: LayerConfig[]
}
```

---

## Training

### Trainer

Handles the training loop.

#### Constructor

```typescript
new Trainer(model: PixelGenModel, config: TrainingConfig)
```

**TrainingConfig:**
```typescript
interface TrainingConfig {
  epochs: number
  batchSize: number
  learningRate: number
  optimizer: 'sgd' | 'adam'
  lossFunction: 'mse' | 'mae' | 'pixelart'
  verbose: boolean
}
```

**Example:**
```typescript
const trainer = new Trainer(model, {
  epochs: 50,
  batchSize: 4,
  learningRate: 0.001,
  optimizer: 'adam',
  lossFunction: 'pixelart',
  verbose: true,
});
```

#### Methods

##### train

```typescript
trainer.train(dataset: ImageDataset): TrainingStats[]
```

Train the model on a dataset.

**Returns:** Array of statistics for each epoch:
```typescript
interface TrainingStats {
  epoch: number
  loss: number
  time: number  // milliseconds
}
```

##### evaluate

```typescript
trainer.evaluate(dataset: ImageDataset): number
```

Evaluate model on a dataset without training.

**Returns:** Average loss

##### getStats

```typescript
trainer.getStats(): TrainingStats[]
```

Get training history.

---

### Optimizers

#### SGD

```typescript
import { SGD } from 'pixelgen';

const optimizer = new SGD(learningRate, momentum);
```

**Parameters:**
- `learningRate`: Learning rate (default: 0.01)
- `momentum`: Momentum factor (default: 0)

#### Adam

```typescript
import { Adam } from 'pixelgen';

const optimizer = new Adam(learningRate, beta1, beta2, epsilon);
```

**Parameters:**
- `learningRate`: Learning rate (default: 0.001)
- `beta1`: First moment decay (default: 0.9)
- `beta2`: Second moment decay (default: 0.999)
- `epsilon`: Numerical stability (default: 1e-8)

---

### Loss Functions

#### mseLoss

```typescript
function mseLoss(predicted: Tensor, target: Tensor): number
```

Mean Squared Error: `(1/n) * Σ(predicted - target)²`

#### maeLoss

```typescript
function maeLoss(predicted: Tensor, target: Tensor): number
```

Mean Absolute Error: `(1/n) * Σ|predicted - target|`

#### bceLoss

```typescript
function bceLoss(predicted: Tensor, target: Tensor): number
```

Binary Cross-Entropy Loss.

#### pixelArtLoss

```typescript
function pixelArtLoss(
  predicted: Tensor,
  target: Tensor,
  edgeWeight?: number
): number
```

Custom loss for PixelArt: `MSE + edgeWeight * EdgeLoss`

**Default:** `edgeWeight = 0.3`

---

## Data

### ImageDataset

Dataset loader for images.

#### Constructor

```typescript
new ImageDataset(config: DatasetConfig)
```

**DatasetConfig:**
```typescript
interface DatasetConfig {
  targetSize: number    // Resize to this size (32)
  normalize: boolean    // Normalize to [0,1]
  channels: number      // RGB = 3, RGBA = 4
}
```

**Example:**
```typescript
const dataset = new ImageDataset({
  targetSize: 32,
  normalize: true,
  channels: 3,
});
```

#### Methods

##### loadFromDirectory

```typescript
dataset.loadFromDirectory(directoryPath: string): void
```

Load all PNG/JPG images from a directory.

##### addTensor

```typescript
dataset.addTensor(tensor: Tensor): void
```

Manually add a pre-processed tensor.

##### getBatch

```typescript
dataset.getBatch(batchSize: number, startIdx?: number): Tensor | null
```

Get a batch of images.

**Returns:** Tensor `[batchSize, channels, height, width]` or `null` if no more data.

##### shuffle

```typescript
dataset.shuffle(): void
```

Randomly shuffle the dataset.

##### size

```typescript
dataset.size(): number
```

Get number of images in dataset.

##### getImagePaths

```typescript
dataset.getImagePaths(): string[]
```

Get list of loaded image file paths.

---

### Utility Functions

#### normalizeTensor

```typescript
function normalizeTensor(tensor: Tensor): Tensor
```

Normalize from [0, 255] to [0, 1].

#### denormalizeTensor

```typescript
function denormalizeTensor(tensor: Tensor): Tensor
```

Denormalize from [0, 1] to [0, 255].

#### createTestDataset

```typescript
function createTestDataset(
  numImages: number,
  channels: number,
  height: number,
  width: number
): ImageDataset
```

Create synthetic test dataset with random data.

---

## I/O

### Model Persistence

#### saveModel

```typescript
function saveModel(
  model: PixelGenModel,
  filepath: string,
  hyperparameters: Hyperparameters
): void
```

Save model to `.pgm` file.

**Hyperparameters:**
```typescript
interface Hyperparameters {
  learningRate: number
  batchSize: number
  epochs: number
  optimizer: string
}
```

**Example:**
```typescript
saveModel(model, 'model.pgm', {
  learningRate: 0.001,
  batchSize: 4,
  epochs: 50,
  optimizer: 'adam',
});
```

#### loadModel

```typescript
function loadModel(filepath: string): {
  model: PixelGenModel
  hyperparameters: Hyperparameters
}
```

Load model from `.pgm` file.

**Example:**
```typescript
const { model, hyperparameters } = loadModel('model.pgm');
```

#### inspectModel

```typescript
function inspectModel(filepath: string): {
  metadata: ModelMetadata
  hyperparameters: Hyperparameters
  fileSize: number
  totalParameters: number
}
```

Inspect model file without loading weights (fast).

**Example:**
```typescript
const info = inspectModel('model.pgm');
console.log(`Parameters: ${info.totalParameters}`);
console.log(`File size: ${info.fileSize} bytes`);
```

---

[← Core Concepts](04-CoreConcepts.md) | [Training Guide →](06-Training-Guide.md)
