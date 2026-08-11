# PixelGen - Project Vision

## What is PixelGen?

PixelGen is a machine learning architecture specialized in PixelArt, built completely from scratch in TypeScript. The goal is to create models that learn to generate and transform PixelArt through examples, without external ML framework dependencies.

## Philosophy

### Why "Rule-Based"?
PixelArt is not free-form art — it has clear mathematical structure:
- Discrete pixels on a grid
- Limited color palettes
- Repetitive patterns and symmetries
- Well-defined visual conventions (outlines, dithering, etc.)

PixelGen uses these properties as **architectural constraints** — the model doesn't learn to draw anything, it learns to operate within the specific space of PixelArt.

### Why TypeScript?
- Accessible to web developers and game devs
- Readable and typed code
- Enables future integration in JS tooling

### Why CPU?
CPU training makes the project accessible to everyone, without requiring specialized hardware. The restricted domain of PixelArt (small images, clear rules) makes this viable.

### Why From Scratch?
Building everything from scratch gives us total control over:
- How the model learns
- How it represents PixelArt internally  
- Domain-specific optimizations
- Zero external dependencies

## Goals

### v1.0 (MVP)
- ✨ Fixed and opinionated architecture
- 📁 User-provided dataset (folder with PNGs)
- 🧠 CPU training
- 💾 Custom model format (.pgm - Pixel Gen Model)
- 🎨 PixelArt generation from scratch after training
- 🔄 Transformation of normal images into PixelArt

### Future
- 🔧 Flexible framework where users define custom architectures
- ⚡ Performance optimizations (multi-threading, SIMD)
- 🎯 Specialized architectures (sprites, tilesets, animations)
- 🌐 Bindings for other languages

## Technical Structure

### Core Components
- **Tensor Engine** — matrix operations with TypedArrays
- **Autograd** — automatic differentiation for backpropagation
- **Layers** — Conv2D, Dense, Pooling, Activation
- **Optimizers** — SGD, Adam
- **Model** — encoder-decoder network architecture
- **Training Loop** — forward, backward, update
- **Data Loader** — image preprocessing
- **I/O** — save/load .pgm models

### .pgm Format (Pixel Gen Model)
Binary with JSON header:
```
[JSON Header with metadata]
[Separator]
[Binary weights in Float32Array]
```

The header contains:
- Architecture version
- Hyperparameters (input/output size, palette, learning rate)
- Tensor shapes
- Training metadata

## Expected Usage

```bash
# Train a model
pixelgen train --dataset ./my-dataset --output model.pgm

# Generate new PixelArt
pixelgen generate --model model.pgm --output sprite.png

# Transform image to PixelArt
pixelgen transform --model model.pgm --input photo.jpg --output pixel.png
```

## Design Principles

1. **Simplicity first** — readable code is more important than micro-optimizations
2. **Restricted domain** — leverage the unique characteristics of PixelArt
3. **Accessibility** — anyone with Node.js should be able to train
4. **Future extensibility** — architecture prepared to evolve into a flexible framework

## Non-Goals (at least in v1)

- ❌ Compete in performance with PyTorch/TensorFlow
- ❌ Be a general-purpose ML framework
- ❌ Run in browser (focus is training, not web inference)
- ❌ Generate photorealistic art or domains outside PixelArt

---

**PixelGen** — Machine Learning for PixelArt, built from scratch, for everyone.
