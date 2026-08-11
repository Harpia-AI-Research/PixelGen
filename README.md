# PixelGen

> Machine Learning para PixelArt, construído do zero em TypeScript

PixelGen é uma arquitetura de machine learning especializada em criar e transformar PixelArt. Desenvolvida completamente do zero, sem dependências de frameworks ML externos.

## 🎯 Características

- ✨ **Arquitetura especializada** - Otimizada exclusivamente para PixelArt
- 🧠 **Implementação do zero** - Tensor engine, autograd e layers construídos from scratch
- 💻 **CPU-friendly** - Treinamento em CPU, sem necessidade de GPU
- 📦 **Formato próprio** - Modelos salvos em `.pgm` (Pixel Gen Model)
- 🎨 **Geração e transformação** - Crie PixelArt do zero ou transforme imagens existentes

## 🏗️ Arquitetura

PixelGen v1.0 implementa um autoencoder convolucional com:

- **Encoder**: Conv2D → Pool → Conv2D → Pool → Bottleneck
- **Decoder**: Upsample → Conv2D → Upsample → Conv2D → Output
- **Otimizadores**: SGD e Adam
- **Loss functions**: MSE, MAE, BCE, e PixelArt-specific loss

## 📦 Instalação

```bash
# Clone o repositório
git clone <repo-url>
cd PixelGen

# Instale dependências
npm install

# Build
npm run build
```

## 🚀 Uso Rápido

### Com Dados de Teste

```typescript
import {
  PixelGenModel,
  Trainer,
  createTestDataset,
  saveModel,
} from 'pixelgen';

// Criar modelo
const model = new PixelGenModel(3, 3); // RGB input/output

// Criar dataset de teste
const dataset = createTestDataset(20, 3, 32, 32);

// Treinar
const trainer = new Trainer(model, {
  epochs: 50,
  batchSize: 4,
  learningRate: 0.001,
  optimizer: 'adam',
  lossFunction: 'pixelart',
  verbose: true,
});

trainer.train(dataset);

// Salvar modelo
saveModel(model, 'meu-modelo.pgm', {
  learningRate: 0.001,
  batchSize: 4,
  epochs: 50,
  optimizer: 'adam',
});
```

### Com Imagens Reais (PNG/JPG)

```typescript
import {
  PixelGenModel,
  Trainer,
  ImageDataset,
  saveModel,
} from 'pixelgen';

// Criar modelo
const model = new PixelGenModel(3, 3);

// Carregar imagens de uma pasta
const dataset = new ImageDataset({
  targetSize: 32,
  normalize: true,
  channels: 3,
});

dataset.loadFromDirectory('./meu-dataset');

// Treinar
const trainer = new Trainer(model, {
  epochs: 50,
  batchSize: 4,
  learningRate: 0.001,
  optimizer: 'adam',
  lossFunction: 'pixelart',
  verbose: true,
});

trainer.train(dataset);

// Salvar
saveModel(model, 'modelo-treinado.pgm', {
  learningRate: 0.001,
  batchSize: 4,
  epochs: 50,
  optimizer: 'adam',
});
```

## 📚 Exemplos

Veja exemplos completos na pasta `examples/`:

```bash
# Exemplo básico (usa dados de teste)
npm run example:basic

# Exemplo avançado (fine-tuning, avaliação)
npm run example:advanced

# Treinar com imagens reais (PNG/JPG)
npm run example:images
```

## 🗂️ Estrutura do Projeto

```
pixelgen/
├── src/
│   ├── core/          # Tensor, autograd, activations
│   ├── layers/        # Conv2D, Dense, Pooling, Upsample
│   ├── model/         # PixelGenModel architecture
│   ├── training/      # Optimizers, loss functions, trainer
│   ├── data/          # Dataset loader e preprocessamento
│   └── io/            # Salvar/carregar modelos .pgm
├── examples/          # Exemplos de uso
├── VISION/            # Documento de visão do projeto
└── README.md
```

## 🎨 Formato .pgm

Os modelos são salvos no formato `.pgm` (Pixel Gen Model):

```
[JSON Header com metadados]
---WEIGHTS---
[Dados binários dos pesos em Float32Array]
```

O header contém:
- Versão da arquitetura
- Hiperparâmetros de treinamento
- Shape dos tensors
- Configuração do modelo

## 🔧 Desenvolvimento

```bash
# Compilar TypeScript
npm run build

# Watch mode
npm run dev

# Limpar build
npm run clean
```

## 📋 Requisitos

- Node.js >= 18.0.0
- TypeScript >= 5.0.0

##  Documentação

Para entender a visão completa do projeto, veja [VISION/README.md](VISION/README.md).

## 🤝 Contribuindo

Este é um projeto experimental em desenvolvimento. Sugestões e contribuições são bem-vindas!

## 📄 Licença

BSD-3-Clause

---

**PixelGen** - Machine Learning para PixelArt, feito do zero, para todos.
