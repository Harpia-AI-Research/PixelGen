# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- PNG and JPEG image parser
- Support for loading images from directories
- Automatic image resizing to 32×32
- Training example with real images (`train-with-images.ts`)
- Automatic conversion from image formats to tensors
- Dependencies: `pngjs` and `jpeg-js` for image processing

---

## [0.1.0] - 2026-08-10

### Added

#### Core
- Implementation of `Tensor` class with support for n-dimensional operations
- Automatic differentiation system (autograd)
- Activation functions: ReLU, Leaky ReLU, Sigmoid, Tanh, Softmax
- Use of `Float32Array` for performance

#### Layers
- `Dense` - Fully connected layer
- `Conv2D` - 2D Convolution with configurable padding and stride
- `MaxPool2D` and `AvgPool2D` - Pooling layers
- `Upsample2D` - Upsampling for decoder

#### Model
- `PixelGenModel` - Fixed convolutional autoencoder for PixelArt
- Encoder-decoder architecture optimized for 32×32 images
- Support for metadata and configuration

#### Training
- Optimizers: `SGD` (with momentum) and `Adam`
- Loss functions: MSE, MAE, BCE, and PixelArt-specific loss
- `Trainer` class with complete training loop
- Training statistics tracking

#### Data
- `ImageDataset` for dataset management
- Normalization and denormalization utilities
- Function to create test datasets
- Support for batching and shuffling

#### I/O
- `.pgm` format (Pixel Gen Model) - binary with JSON header
- Functions to save and load models
- Function to inspect models without loading full weights

#### Documentation
- Main README with usage guide
- Project vision document (VISION/)
- Getting Started guide
- Technical architecture documentation
- Examples: basic and advanced
- BSD-3-Clause License

#### Tooling
- TypeScript configuration
- npm scripts for build, dev, and examples
- Configured .gitignore
- package.json with complete metadata

### Notes

This is the first public version of PixelGen. The architecture is fixed and focused on demonstrating the concept of ML for PixelArt built from scratch.

### Known Limitations

- PNG/JPEG image parser not implemented (uses test data)
- Fixed architecture at 32×32 pixels
- RGB only (3 channels)
- Simplified autograd
- Performance limited by pure CPU execution

---

[0.1.0]: https://github.com/Harpia-AI-Research/PixelGen/releases/tag/v0.1.0
