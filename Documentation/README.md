# PixelGen Documentation

Complete guide to using PixelGen for PixelArt generation and transformation.

## Getting Started

New to PixelGen? Start here:

1. **[Introduction](01-Introduction.md)** - What is PixelGen and why use it?
2. **[Installation](02-Installation.md)** - Set up PixelGen on your system
3. **[Quick Start](03-QuickStart.md)** - Train your first model in 5 minutes

## Core Documentation

Understand how PixelGen works:

4. **[Core Concepts](04-CoreConcepts.md)** - Tensors, autograd, layers, and architecture
5. **[API Reference](05-API-Reference.md)** - Complete API documentation
6. **[Training Guide](06-Training-Guide.md)** *(Coming Soon)* - Master hyperparameters and training
7. **[Model Architecture](07-Model-Architecture.md)** *(Coming Soon)* - Deep dive into the architecture
8. **[Advanced Usage](08-Advanced-Usage.md)** *(Coming Soon)* - Fine-tuning, evaluation, custom architectures

## Additional Resources

- **[examples/](../examples/)** - Example scripts
  - `basic-training.ts` - Simple training with test data
  - `advanced-usage.ts` - Fine-tuning and evaluation
  - `train-with-images.ts` - Training with real images

- **[VISION/](../VISION/)** - Project vision and philosophy
- **[CONTRIBUTING.md](../CONTRIBUTING.md)** - Contribution guidelines
- **[CHANGELOG.md](../CHANGELOG.md)** - Version history

## Quick Links

### Common Tasks

- [Train a model with test data](03-QuickStart.md#option-1-training-with-test-data)
- [Train with your own images](03-QuickStart.md#option-2-training-with-real-images)
- [Load a trained model](03-QuickStart.md#loading-and-using-a-trained-model)
- [Save and load models](05-API-Reference.md#model-persistence)
- [Adjust hyperparameters](03-QuickStart.md#experiment-with-hyperparameters)

### Technical Deep Dives

- [How tensors work](04-CoreConcepts.md#tensors)
- [Automatic differentiation explained](04-CoreConcepts.md#automatic-differentiation)
- [Neural network layers](04-CoreConcepts.md#neural-network-layers)
- [The autoencoder architecture](04-CoreConcepts.md#the-autoencoder-architecture)
- [Training loop breakdown](04-CoreConcepts.md#training-loop)
- [Loss functions](04-CoreConcepts.md#loss-functions)
- [Optimizers (SGD vs Adam)](04-CoreConcepts.md#optimizers)

### API Quick Reference

- [Tensor class](05-API-Reference.md#tensor)
- [PixelGenModel class](05-API-Reference.md#pixelgenmodel)
- [Trainer class](05-API-Reference.md#trainer)
- [ImageDataset class](05-API-Reference.md#imagedataset)
- [All activation functions](05-API-Reference.md#activation-functions)
- [All loss functions](05-API-Reference.md#loss-functions-1)

## Documentation Status

| Document | Status |
|----------|--------|
| Introduction | ✅ Complete |
| Installation | ✅ Complete |
| Quick Start | ✅ Complete |
| Core Concepts | ✅ Complete |
| API Reference | ✅ Complete |
| Training Guide | 🚧 Coming Soon |
| Model Architecture | 🚧 Coming Soon |
| Advanced Usage | 🚧 Coming Soon |

## Need Help?

- **Found a bug?** Open an issue on GitHub
- **Have a question?** Check existing issues or create a new one
- **Want to contribute?** See [CONTRIBUTING.md](../CONTRIBUTING.md)

## License

PixelGen is licensed under the BSD-3-Clause License.  
Copyright (c) 2026, Harpia AI Research

---

<div align="center">
  <strong>Happy PixelArt Generation!</strong>
</div>
