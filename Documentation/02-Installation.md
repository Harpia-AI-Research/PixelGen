# Installation Guide

## Prerequisites

### Required Software

1. **Node.js** (version 18.0.0 or higher)
   - Download from: https://nodejs.org/
   - Verify installation: `node --version`

2. **npm** (comes with Node.js) or **yarn**
   - Verify npm: `npm --version`
   - Or install yarn: `npm install -g yarn`

3. **Git** (for cloning the repository)
   - Download from: https://git-scm.com/
   - Verify: `git --version`

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 2 GB | 4 GB+ |
| CPU | 2 cores | 4+ cores |
| Storage | 500 MB | 1 GB |
| OS | Windows, macOS, Linux | Any modern OS |

## Installation Steps

### Method 1: Clone from GitHub (Recommended)

```bash
# Clone the repository
git clone https://github.com/Harpia-AI-Research/PixelGen.git

# Navigate to directory
cd PixelGen

# Install dependencies
npm install

# Build the project
npm run build
```

### Method 2: Download ZIP

1. Go to https://github.com/Harpia-AI-Research/PixelGen
2. Click "Code" → "Download ZIP"
3. Extract the ZIP file
4. Open terminal in extracted folder
5. Run:
```bash
npm install
npm run build
```

## Verify Installation

After installation, verify everything works:

```bash
# Check if TypeScript compiled successfully
ls dist/

# You should see:
# - core/
# - layers/
# - model/
# - training/
# - data/
# - io/
# - index.js
# - index.d.ts
```

### Run Test Example

```bash
npm run example:basic
```

Expected output:
```
PixelGen - Basic Training Example
==================================

Step 1: Creating model...
Model created with 12 parameter tensors

Step 2: Creating test dataset...
Dataset created with 20 images
...
```

If you see this, installation was successful!

## Dependency Overview

PixelGen uses minimal external dependencies:

### Production Dependencies

```json
{
  "pngjs": "^7.0.0",      // PNG image parsing
  "jpeg-js": "^0.4.4"     // JPEG image parsing
}
```

### Development Dependencies

```json
{
  "@types/node": "^20.0.0",
  "@types/pngjs": "^6.0.0",
  "typescript": "^5.0.0"
}
```

**Total size**: ~15 MB (including dependencies)

## Project Structure After Installation

```
PixelGen/
├── node_modules/          # Dependencies (auto-generated)
├── dist/                  # Compiled JavaScript (auto-generated)
│   ├── core/
│   ├── layers/
│   ├── model/
│   ├── training/
│   ├── data/
│   └── io/
├── src/                   # TypeScript source code
├── examples/              # Example usage files
├── Documentation/         # This documentation
├── package.json           # Project metadata
├── tsconfig.json          # TypeScript configuration
└── README.md              # Project overview
```

## Development Setup

If you want to modify PixelGen or contribute:

### Watch Mode

Automatically recompile on file changes:

```bash
npm run dev
```

This runs TypeScript in watch mode. Any changes to `.ts` files will automatically trigger recompilation.

### Clean Build

Remove previous build and rebuild:

```bash
npm run clean
npm run build
```

## Troubleshooting

### Issue: "npm install" fails

**Solution**:
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and package-lock.json
rm -rf node_modules package-lock.json

# Reinstall
npm install
```

### Issue: "tsc: command not found"

**Solution**:
```bash
# Install TypeScript globally
npm install -g typescript

# Or use npx
npx tsc
```

### Issue: Node version too old

**Solution**:
- Update Node.js to version 18+ from https://nodejs.org/
- Or use nvm (Node Version Manager):
```bash
# Install nvm first, then:
nvm install 18
nvm use 18
```

### Issue: Permission denied on Linux/Mac

**Solution**:
```bash
# Don't use sudo! Instead, fix npm permissions:
mkdir ~/.npm-global
npm config set prefix '~/.npm-global'

# Add to PATH (add to ~/.bashrc or ~/.zshrc)
export PATH=~/.npm-global/bin:$PATH

# Reload shell
source ~/.bashrc  # or source ~/.zshrc
```

### Issue: Build succeeds but examples fail

**Solution**:
```bash
# Make sure you're in the project root
cd PixelGen

# Rebuild
npm run build

# Check dist/ was created
ls dist/

# Try example again
npm run example:basic
```

## IDE Setup (Optional)

### Visual Studio Code (Recommended)

1. Install VS Code: https://code.visualstudio.com/
2. Open PixelGen folder in VS Code
3. Install recommended extensions:
   - TypeScript (built-in)
   - ESLint (optional)
   - Prettier (optional)

VS Code will automatically provide:
- TypeScript IntelliSense
- Error checking
- Auto-completion
- Go-to-definition

### Other IDEs

PixelGen works with any IDE that supports TypeScript:
- WebStorm
- Sublime Text + TypeScript plugin
- Vim/Neovim + CoC or LSP
- Emacs + tide

## Next Steps

Installation complete! Continue to:

- **[Quick Start Guide](03-QuickStart.md)**: Train your first model
- **[Core Concepts](04-CoreConcepts.md)**: Understand the architecture
- **[API Reference](05-API-Reference.md)**: Explore available functions

---

[← Introduction](01-Introduction.md) | [Quick Start →](03-QuickStart.md)
