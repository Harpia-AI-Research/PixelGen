# Contributing to PixelGen

Thank you for considering contributing to PixelGen! This document provides guidelines for contributions.

## 🎯 How to Contribute

### Reporting Bugs

If you found a bug, open an issue with:

1. **Clear title** - Describe the problem briefly
2. **Detailed description** - Explain what happened
3. **Steps to reproduce** - How to reproduce the problem
4. **Expected behavior** - What should happen
5. **Environment** - Node.js version, OS, etc.
6. **Code example** - If possible, a minimal snippet that reproduces the issue

### Suggesting Features

To suggest a new feature:

1. **Check existing issues** - Maybe it's already planned
2. **Describe the use case** - Why is this feature useful?
3. **Propose a solution** - How do you imagine it would work?
4. **Consider alternatives** - Other ways to solve the problem?

### Pull Requests

1. **Fork the repository**
2. **Create a branch** - `git checkout -b feature/my-feature`
3. **Make your changes** - Follow code conventions
4. **Test your changes** - Make sure it works
5. **Commit your changes** - Clear and descriptive messages
6. **Push to fork** - `git push origin feature/my-feature`
7. **Open a Pull Request** - Describe your changes

## 📝 Code Conventions

### TypeScript

- Use **TypeScript strict mode**
- Prefer **interfaces** for public types
- Use **explicit types** in function parameters
- Avoid `any` - use specific types or `unknown`

### Naming

```typescript
// Classes: PascalCase
class PixelGenModel { }

// Functions and variables: camelCase
function trainModel() { }
const learningRate = 0.001;

// Constants: UPPER_SNAKE_CASE
const DEFAULT_BATCH_SIZE = 4;

// Types and interfaces: PascalCase
interface ModelConfig { }
type TensorShape = number[];
```

### Comments

```typescript
/**
 * Function description
 * @param input - Parameter description
 * @returns Return value description
 */
function forward(input: Tensor): Tensor {
  // Inline comments when necessary
  return output;
}
```

### File Structure

```
src/
├── core/          # Foundation (Tensor, autograd)
├── layers/        # Building blocks (Conv2D, Dense)
├── model/         # Model architectures
├── training/      # Optimizers, loss, trainer
├── data/          # Data loading and preprocessing
└── io/            # Save/load models
```

## 🧪 Testing

Currently, the project does not have automated tests. Contributions to add tests are very welcome!

### Manual Testing

```bash
# Build
npm run build

# Run examples
npm run example:basic
npm run example:advanced
```

## 📚 Documentation

If you add a new feature, please:

1. Update the README if necessary
2. Add code comments
3. Create or update documentation in `Documentation/`
4. Add examples if appropriate

## 🎨 Areas That Need Help

### High Priority

- [ ] **PNG/JPEG image parser** - Implement native image loading
- [ ] **Automated tests** - Create test suite with Jest or similar
- [ ] **CLI** - Command-line interface for training/generation
- [ ] **Performance optimizations** - SIMD, multi-threading, etc.

### Medium Priority

- [ ] **Data validation** - Better input validation
- [ ] **Logging** - Structured logging system
- [ ] **Visualization** - Tools to visualize training
- [ ] **Documentation** - More examples and tutorials

### Low Priority

- [ ] **RGBA support** - Transparency in images
- [ ] **Export to other formats** - ONNX, TensorFlow.js
- [ ] **Pre-trained models** - Provide base models

## 🤔 Questions?

If you have questions about how to contribute:

1. Open an issue with the `question` tag
2. Describe your question clearly
3. We'll respond as soon as possible

## 📜 Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive experience for everyone.

### Expected Behavior

- Use welcoming and inclusive language
- Be respectful of different viewpoints
- Accept constructive criticism
- Focus on what's best for the community
- Show empathy towards other members

### Unacceptable Behavior

- Sexualized language or images
- Trolling, insulting or derogatory comments
- Public or private harassment
- Publishing others' private information without permission
- Other conduct that would be considered inappropriate in a professional setting

## 📄 License

By contributing, you agree that your contributions will be licensed under the same BSD-3-Clause license as the project.

---

Thank you for helping make PixelGen better! 🎨✨
