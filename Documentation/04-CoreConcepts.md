# Core Concepts

This guide explains the fundamental concepts behind PixelGen's architecture and operation.

## Table of Contents

1. [Tensors](#tensors)
2. [Automatic Differentiation](#automatic-differentiation)
3. [Neural Network Layers](#neural-network-layers)
4. [The Autoencoder Architecture](#the-autoencoder-architecture)
5. [Training Loop](#training-loop)
6. [Loss Functions](#loss-functions)
7. [Optimizers](#optimizers)

---

## Tensors

### What is a Tensor?

A **tensor** is an n-dimensional array of numbers. Think of it as a generalization:

- **0D tensor** (scalar): Single number `5`
- **1D tensor** (vector): `[1, 2, 3, 4]`
- **2D tensor** (matrix): `[[1,2], [3,4]]`
- **3D tensor**: `[[[1,2], [3,4]], [[5,6], [7,8]]]`
- **4D tensor**: Used for batches of images

### Images as Tensors

In PixelGen, images are represented as 4D tensors:

```
Shape: [batch, channels, height, width]
Example: [1, 3, 32, 32]
```

- **batch**: Number of images (1 for single image)
- **channels**: Color channels (3 for RGB)
- **height**: Image height in pixels (32)
- **width**: Image width in pixels (32)

### Why Float32Array?

PixelGen uses `Float32Array` internally:

```typescript
// Regular JavaScript array (slow)
const arr = [1.0, 2.0, 3.0];

// Float32Array (fast)
const tensor = new Float32Array([1.0, 2.0, 3.0]);
```

**Benefits:**
- ~10x faster for mathematical operations
- Fixed memory layout (better CPU cache usage)
- Native support for SIMD operations

---

## Automatic Differentiation

### The Problem: Computing Gradients

To train neural networks, we need **gradients** — the derivatives of the loss with respect to each parameter.

For a simple function like `f(x) = x²`:
- Derivative: `f'(x) = 2x`
- Easy to compute manually

For a neural network with 128,000 parameters:
- Computing derivatives manually is impossible
- We need **automatic differentiation (autograd)**

### How Autograd Works

#### 1. Forward Pass

Build a computational graph:

```
x = 2
y = 3
z = x * y = 6
w = z + x = 8
```

Each operation stores:
- Its inputs
- How to compute its gradient

#### 2. Backward Pass

Walk backwards through the graph, applying chain rule:

```
∂w/∂w = 1           (start)
∂w/∂z = 1           (derivative of addition)
∂w/∂x = ∂w/∂z * ∂z/∂x + ∂w/∂x
      = 1 * y + 1 = 4

∂w/∂y = ∂w/∂z * ∂z/∂y
      = 1 * x = 2
```

### Implementation in PixelGen

```typescript
const a = new Tensor([2], [1], true); // requiresGrad=true
const b = new Tensor([3], [1], true);

const c = a.multiply(b); // c = 6
const d = c.add(a);      // d = 8

d.backward(); // Compute gradients

console.log(a.grad); // ∂d/∂a = 4
console.log(b.grad); // ∂d/∂b = 2
```

---

## Neural Network Layers

### Convolutional Layer (Conv2D)

Applies a **sliding window** (kernel) across the image:

```
Input: 32×32 image
Kernel: 3×3 filter
Output: 32×32 feature map (with padding)
```

**What it does:**
- Detects local patterns (edges, corners, textures)
- Each kernel learns to detect a specific feature
- Shared weights across the image (parameter efficient)

**Example:**
```typescript
const conv = new Conv2D(
  3,    // input channels (RGB)
  32,   // output channels (32 features)
  3,    // kernel size (3×3)
  1,    // stride
  1     // padding
);

const output = conv.forward(input);
// Input: [1, 3, 32, 32]
// Output: [1, 32, 32, 32]
```

### Pooling Layers

**MaxPool2D**: Take maximum value in each region

```
Input:  [1,2,3,4]      Output: [3,4]
        [5,6,7,8]  →          [7,8]
        
(2×2 pooling with stride 2)
```

**Purpose:**
- Reduce spatial dimensions
- Make features more robust to small translations
- Reduce computation in later layers

### Upsampling Layer

**Nearest Neighbor Upsampling**: Repeat pixel values

```
Input:  [1,2]      Output: [1,1,2,2]
        [3,4]  →          [1,1,2,2]
                          [3,3,4,4]
                          [3,3,4,4]
```

**Purpose:**
- Increase spatial dimensions (for decoder)
- Simple and fast
- Preserves sharp edges (good for PixelArt)

### Dense (Fully Connected) Layer

Connects every input to every output:

```
Input: [x1, x2, x3]
Weights: W (matrix)
Bias: b
Output: W @ input + b
```

**Not used in PixelGen's main model** (fully convolutional), but available for custom architectures.

---

## The Autoencoder Architecture

### Conceptual Overview

```
Original Image → Encoder → Compressed Code → Decoder → Reconstructed Image
    32×32           ↓           8×8×128         ↓            32×32
```

### Why Compress and Decompress?

By forcing information through a **bottleneck**, the model must learn:
- Which features are important
- How to efficiently encode PixelArt
- Common patterns and structures

### Encoder Path

```
Input: 32×32×3 (3,072 values)
   ↓
Conv(3→32) + ReLU
   ↓
MaxPool(2×) → 16×16×32
   ↓
Conv(32→64) + ReLU
   ↓
MaxPool(2×) → 8×8×64
   ↓
Conv(64→128) + ReLU
   ↓
Bottleneck: 8×8×128 (8,192 values)
```

**Compression ratio**: 3,072 → 8,192 (actually expanded in channels, but spatially compressed 16×)

### Decoder Path

```
Bottleneck: 8×8×128
   ↓
Conv(128→64) + ReLU
   ↓
Upsample(2×) → 16×16×64
   ↓
Conv(64→32) + ReLU
   ↓
Upsample(2×) → 32×32×32
   ↓
Conv(32→3) (no activation)
   ↓
Output: 32×32×3
```

### What the Model Learns

- **Early layers**: Low-level features (edges, colors)
- **Middle layers**: Mid-level patterns (shapes, structures)
- **Bottleneck**: High-level understanding (this is a character, a tree, etc.)
- **Decoder**: How to reconstruct from this understanding

---

## Training Loop

### The Four Steps

```
1. Forward Pass  → Compute output
2. Loss Calculation → Measure error
3. Backward Pass → Compute gradients
4. Parameter Update → Adjust weights
```

### Detailed Walkthrough

#### Step 1: Forward Pass

```typescript
const output = model.forward(input);
// Input: [4, 3, 32, 32]  (batch of 4 images)
// Output: [4, 3, 32, 32] (reconstructions)
```

#### Step 2: Compute Loss

```typescript
const loss = pixelArtLoss(output, input);
// Compare reconstruction to original
// Result: single number (e.g., 0.123)
```

#### Step 3: Backward Pass

```typescript
loss.backward();
// Compute ∂loss/∂weight for every weight
// Stored in weight.grad
```

#### Step 4: Update Parameters

```typescript
optimizer.step(model.parameters());
// For each parameter:
//   weight = weight - learningRate * gradient
```

### One Epoch

An **epoch** is one complete pass through the dataset:

```
For each batch in dataset:
    1. Forward pass
    2. Compute loss
    3. Backward pass
    4. Update parameters
```

After one epoch, the model has seen every training image once.

---

## Loss Functions

### Mean Squared Error (MSE)

```
loss = (1/n) * Σ(predicted - target)²
```

**Properties:**
- Simple and fast
- Penalizes large errors more than small ones
- Can lead to blurry outputs

### PixelArt Loss

```
loss = MSE + λ * EdgeLoss
```

**EdgeLoss** measures difference in gradients:

```
For each pixel:
    horizontal_gradient = |pixel[x+1] - pixel[x]|
    vertical_gradient = |pixel[y+1] - pixel[y]|
    
EdgeLoss = |predicted_gradients - target_gradients|
```

**Effect:**
- Encourages sharp transitions
- Preserves edges and outlines
- Better for PixelArt than plain MSE

---

## Optimizers

### Gradient Descent

Basic idea:
```
weight = weight - learningRate * gradient
```

If gradient is positive → weight decreases
If gradient is negative → weight increases

### Stochastic Gradient Descent (SGD)

Update weights after each **batch** (not whole dataset):

```
For each batch:
    compute gradients
    update weights
```

**With Momentum:**
```
velocity = momentum * velocity + gradient
weight = weight - learningRate * velocity
```

Benefits:
- Faster convergence
- Smooths out noisy gradients
- Can escape local minima

### Adam (Recommended)

Adaptive learning rates per parameter:

```
m = β₁ * m + (1-β₁) * gradient      // momentum
v = β₂ * v + (1-β₂) * gradient²     // velocity

m_hat = m / (1 - β₁^t)              // bias correction
v_hat = v / (1 - β₂^t)

weight = weight - lr * m_hat / (√v_hat + ε)
```

**Benefits:**
- Adapts learning rate per parameter
- Usually works well with default settings
- Less sensitive to learning rate choice

**Default hyperparameters:**
- β₁ = 0.9
- β₂ = 0.999
- ε = 1e-8

---

## Putting It All Together

### Training Flow

```
1. Load dataset
2. Create model
3. For each epoch:
   a. Shuffle dataset
   b. For each batch:
      i.   Forward pass
      ii.  Compute loss
      iii. Backward pass
      iv.  Update weights
   c. Report epoch loss
4. Save trained model
```

### What the Model Learns

- **Epoch 1-5**: Basic colors and rough shapes
- **Epoch 5-20**: Patterns, structures, common elements
- **Epoch 20-50**: Fine details, style consistency
- **Epoch 50+**: Over-fitting (memorizes training data)

### When to Stop Training

Stop when:
- Loss stops decreasing
- Validation loss starts increasing (if you have validation set)
- You're happy with the results
- Model starts memorizing instead of generalizing

---

## Summary

- **Tensors**: n-dimensional arrays, the fundamental data structure
- **Autograd**: Automatic gradient computation via computational graphs
- **Layers**: Building blocks (Conv2D, Pooling, Upsampling)
- **Autoencoder**: Compress → Bottleneck → Decompress
- **Training**: Forward → Loss → Backward → Update
- **Loss Functions**: MSE for reconstruction, PixelArt loss for edges
- **Optimizers**: Adam for adaptive learning rates

---

[← Quick Start](03-QuickStart.md) | [API Reference →](05-API-Reference.md)
