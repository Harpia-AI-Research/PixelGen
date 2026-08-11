# Arquitetura Técnica do PixelGen

Este documento detalha a implementação técnica do PixelGen.

## 📐 Visão Geral

PixelGen é implementado em camadas, de baixo para cima:

```
┌─────────────────────────────────────┐
│         Model & Training            │  ← Alto nível (usuário)
├─────────────────────────────────────┤
│      Layers (Conv2D, Dense, etc)    │  ← Building blocks
├─────────────────────────────────────┤
│    Core (Tensor, Autograd, etc)     │  ← Fundação
└─────────────────────────────────────┘
```

## 🧱 Camada Core

### Tensor (`src/core/Tensor.ts`)

A classe fundamental que representa tensores n-dimensionais.

**Características:**
- Usa `Float32Array` para performance
- Suporte a autograd (diferenciação automática)
- Operações básicas: add, multiply, matmul, transpose
- Shape tracking e validação

**Exemplo:**
```typescript
const tensor = new Tensor([1, 2, 3, 4], [2, 2], true);
console.log(tensor.shape); // [2, 2]
```

**Operações Implementadas:**
- Aritméticas: `add()`, `multiply()`
- Matriciais: `matmul()`, `transpose()`
- Shape: `reshape()`, `clone()`
- Autograd: `backward()`, `zeroGrad()`

### Autograd

Sistema de diferenciação automática baseado em computational graph.

**Como funciona:**
1. Durante forward pass, cada operação registra seus "pais"
2. Cada operação define sua função `_backward()`
3. `backward()` percorre o grafo em ordem topológica reversa
4. Gradientes são acumulados nos tensores

**Exemplo de gradiente:**
```typescript
const a = new Tensor([2], [1], true);
const b = new Tensor([3], [1], true);
const c = a.multiply(b); // c = 6

c.backward(); // Calcula gradientes
console.log(a.grad); // dc/da = b = 3
console.log(b.grad); // dc/db = a = 2
```

### Activation Functions (`src/core/activations.ts`)

Funções de ativação com suporte a autograd:

- **ReLU**: `f(x) = max(0, x)`
- **Leaky ReLU**: `f(x) = x if x > 0 else α*x`
- **Sigmoid**: `f(x) = 1 / (1 + e^(-x))`
- **Tanh**: `f(x) = tanh(x)`
- **Softmax**: Normalização probabilística

## 🔨 Camada de Layers

### Dense Layer (`src/layers/Dense.ts`)

Camada totalmente conectada (fully connected).

**Matemática:**
```
output = input @ weights + bias
```

**Dimensões:**
- Input: `[batch_size, input_size]`
- Output: `[batch_size, output_size]`
- Weights: `[input_size, output_size]`
- Bias: `[output_size]`

**Inicialização:** Xavier/Glorot para estabilidade

### Conv2D Layer (`src/layers/Conv2D.ts`)

Camada convolucional 2D para processar imagens.

**Matemática:**
```
output[b,oc,h,w] = Σ(input[b,ic,h',w'] * kernel[oc,ic,kh,kw]) + bias[oc]
```

**Parâmetros:**
- `inChannels`: Número de canais de entrada
- `outChannels`: Número de filtros
- `kernelSize`: Tamanho do kernel (ex: 3x3)
- `stride`: Passo da convolução
- `padding`: Padding nas bordas

**Implementação:**
- Suporta padding
- Stride configurável
- Inicialização Xavier

### Pooling Layers (`src/layers/Pooling.ts`)

**MaxPool2D**: Reduz dimensão espacial pegando o máximo em cada janela

**AvgPool2D**: Reduz dimensão espacial pela média

**Upsample2D**: Aumenta dimensão espacial (nearest neighbor)

## 🏗️ Modelo PixelGen

### PixelGenModel (`src/model/PixelGenModel.ts`)

Autoencoder convolucional especializado para PixelArt.

**Arquitetura:**

```
Input (32x32x3)
    ↓
[Encoder]
Conv2D(3→32) + ReLU
MaxPool(2x) → 16x16
Conv2D(32→64) + ReLU
MaxPool(2x) → 8x8
Conv2D(64→128) + ReLU
    ↓
[Bottleneck] (8x8x128)
    ↓
[Decoder]
Upsample(2x) → 16x16
Conv2D(128→64) + ReLU
Upsample(2x) → 32x32
Conv2D(64→32) + ReLU
Conv2D(32→3)
    ↓
Output (32x32x3)
```

**Por que essa arquitetura?**
- Encoder comprime a imagem em representação latente
- Bottleneck força o modelo a aprender features importantes
- Decoder reconstrói a imagem a partir do latent space
- Perfeito para aprender padrões de PixelArt

## 🎓 Sistema de Treinamento

### Optimizers (`src/training/Optimizer.ts`)

**SGD (Stochastic Gradient Descent)**
```
θ = θ - lr * ∇θ
```

**SGD com Momentum**
```
v = β * v - lr * ∇θ
θ = θ + v
```

**Adam (Adaptive Moment Estimation)**
```
m = β₁ * m + (1-β₁) * ∇θ
v = β₂ * v + (1-β₂) * (∇θ)²
θ = θ - lr * m̂ / (√v̂ + ε)
```

### Loss Functions (`src/training/Loss.ts`)

**MSE (Mean Squared Error)**
```
loss = (1/n) * Σ(predicted - target)²
```

**PixelArt Loss**
```
loss = MSE + λ * EdgeLoss
```

EdgeLoss penaliza bordas borradas:
```
EdgeLoss = Σ|∇predicted - ∇target|
```

Isso incentiva transições nítidas entre cores, essencial para PixelArt.

### Trainer (`src/training/Trainer.ts`)

Loop de treinamento padrão:

```typescript
for epoch in epochs:
  shuffle(dataset)
  for batch in dataset:
    // Forward pass
    output = model.forward(batch)
    
    // Calculate loss
    loss = lossFunction(output, batch)
    
    // Backward pass
    model.zeroGrad()
    loss.backward()
    
    // Update weights
    optimizer.step(model.parameters())
```

## 💾 Sistema de I/O

### Formato .pgm (`src/io/ModelIO.ts`)

Estrutura binária com header JSON:

```
[Header JSON]\n
---WEIGHTS---\n
[Float32Array binário]
```

**Header contém:**
```json
{
  "metadata": {
    "version": "1.0.0",
    "architecture": "pixelgen-autoencoder-v1",
    "inputChannels": 3,
    "outputChannels": 3,
    "layers": [...]
  },
  "hyperparameters": {
    "learningRate": 0.001,
    "batchSize": 4,
    "epochs": 50,
    "optimizer": "adam"
  },
  "parameterShapes": [[32, 3, 3, 3], ...],
  "totalParameters": 123456
}
```

**Vantagens:**
- Header legível em JSON
- Pesos em formato binário eficiente
- Todas as informações para reconstruir o modelo
- Compacto e rápido de carregar

## 📊 Data Pipeline

### ImageDataset (`src/data/ImageLoader.ts`)

**Responsabilidades:**
- Carregar imagens de uma pasta
- Pré-processar (resize, normalização)
- Batchificação
- Shuffle para treinamento

**Pipeline:**
```
Imagens em disco
    ↓
Parse PNG/JPG (TODO em v1)
    ↓
Resize para 32x32
    ↓
Normalizar [0, 255] → [0, 1]
    ↓
Convert para Tensor
    ↓
Batch + Shuffle
    ↓
Training
```

## ⚡ Otimizações

### Uso de TypedArrays

`Float32Array` ao invés de arrays JavaScript normais:
- ~10x mais rápido em operações numéricas
- Uso de memória mais eficiente
- Melhor cache locality

### Operações In-Place

Quando possível, reusar buffers:
```typescript
// Evitar
result = new Float32Array(size);

// Preferir (quando seguro)
result.data[i] += value;
```

### Batch Processing

Processar múltiplas imagens de uma vez:
- Melhor utilização da CPU
- Amortiza overhead de operações

## 🔮 Limitações Conhecidas (v1.0)

1. **Autograd simplificado**: Não suporta todas as operações complexas
2. **Performance em CPU**: Mais lento que implementações GPU
3. **Tamanho fixo**: Apenas 32x32 pixels
4. **Parser de imagens**: Não implementado, usa dados de teste
5. **Arquitetura fixa**: Não customizável (será na v2.0)

## 🚀 Próximas Melhorias

1. **SIMD**: Usar WebAssembly SIMD para acelerar operações
2. **Worker Threads**: Paralelizar batch processing
3. **Autograd completo**: Suportar operações mais complexas
4. **Arquitetura flexível**: Permitir definir modelos customizados
5. **Image parser**: Suporte nativo a PNG/JPG

---

Esta é a base técnica do PixelGen v1.0. A arquitetura foi projetada para ser simples, compreensível e extensível.
