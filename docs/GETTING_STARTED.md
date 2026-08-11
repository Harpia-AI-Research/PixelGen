# Getting Started com PixelGen

Este guia vai te ajudar a começar a usar PixelGen para treinar seus próprios modelos de PixelArt.

## 📦 Instalação

```bash
# Clone o repositório
git clone <repo-url>
cd PixelGen

# Instale as dependências
npm install

# Build o projeto
npm run build
```

## 🎓 Seu Primeiro Modelo

### 1. Preparar o Dataset

Para treinar um modelo, você precisa de imagens de PixelArt. Por enquanto (v1.0), você pode:

**Opção A: Usar dataset de teste (para experimentar)**
```typescript
import { createTestDataset } from 'pixelgen';

const dataset = createTestDataset(
  50,  // número de imagens
  3,   // canais RGB
  32,  // altura
  32   // largura
);
```

**Opção B: Preparar suas próprias imagens**
Coloque suas imagens PNG/JPG em uma pasta:
```
meu-dataset/
├── sprite001.png
├── sprite002.png
├── sprite003.png
└── ...
```

> **Nota**: O parser de imagens PNG/JPG será implementado em versões futuras. Por enquanto, use a opção A para testar.

### 2. Criar o Modelo

```typescript
import { PixelGenModel } from 'pixelgen';

// Criar modelo com 3 canais de entrada e saída (RGB)
const model = new PixelGenModel(3, 3);

console.log(`Modelo criado com ${model.parameters().length} tensores de parâmetros`);
```

### 3. Configurar o Treinamento

```typescript
import { Trainer, DEFAULT_TRAINING_CONFIG } from 'pixelgen';

const trainingConfig = {
  ...DEFAULT_TRAINING_CONFIG,
  epochs: 50,           // número de épocas
  batchSize: 4,         // tamanho do batch
  learningRate: 0.001,  // taxa de aprendizado
  optimizer: 'adam',    // otimizador (adam ou sgd)
  lossFunction: 'pixelart', // função de loss
  verbose: true         // mostrar progresso
};

const trainer = new Trainer(model, trainingConfig);
```

### 4. Treinar o Modelo

```typescript
// Treinar
const stats = trainer.train(dataset);

// Ver estatísticas
console.log('Treinamento completo!');
console.log(`Loss final: ${stats[stats.length - 1].loss}`);
```

### 5. Salvar o Modelo

```typescript
import { saveModel } from 'pixelgen';

saveModel(model, 'meu-modelo.pgm', {
  learningRate: trainingConfig.learningRate,
  batchSize: trainingConfig.batchSize,
  epochs: trainingConfig.epochs,
  optimizer: trainingConfig.optimizer,
});
```

### 6. Carregar e Usar o Modelo

```typescript
import { loadModel } from 'pixelgen';

// Carregar modelo salvo
const { model: loadedModel, hyperparameters } = loadModel('meu-modelo.pgm');

// Usar o modelo para gerar/transformar
const output = loadedModel.forward(inputTensor);
```

## 🔍 Exemplo Completo

Veja o arquivo `examples/basic-training.ts` para um exemplo completo e funcional.

Execute o exemplo:
```bash
npm run example:basic
```

## ⚙️ Configurações Avançadas

### Hiperparâmetros

```typescript
const trainingConfig = {
  epochs: 50,              // Mais épocas = mais treinamento (pode overfittar)
  batchSize: 4,            // Maior batch = mais estável, mas usa mais memória
  learningRate: 0.001,     // Menor = mais preciso mas mais lento
  optimizer: 'adam',       // 'adam' geralmente funciona melhor
  lossFunction: 'pixelart', // 'pixelart' ou 'mse'
  verbose: true
};
```

### Funções de Loss

**MSE (Mean Squared Error)**
- Loss padrão, boa para começar
- Simples e rápida

**PixelArt Loss**
- Customizada para PixelArt
- Preserva bordas e transições nítidas
- Recomendada para melhores resultados

### Otimizadores

**Adam** (Recomendado)
- Adaptativo, funciona bem na maioria dos casos
- Bom ponto de partida

**SGD** (Stochastic Gradient Descent)
- Mais simples
- Pode precisar de ajuste fino do learning rate

## 📊 Monitorando o Treinamento

Durante o treinamento, você verá:
```
Epoch 1/50 - Loss: 0.123456 - Time: 1234ms
Epoch 2/50 - Loss: 0.098765 - Time: 1220ms
...
```

**O que observar:**
- ✅ Loss diminuindo = modelo aprendendo
- ⚠️ Loss estagnado = talvez precise ajustar learning rate
- ❌ Loss aumentando = learning rate muito alto ou problema no dataset

## 🐛 Troubleshooting

### "Shape mismatch error"
- Certifique-se que todas as imagens têm o mesmo tamanho
- PixelGen v1 espera imagens 32x32

### Treinamento muito lento
- Reduza o batch size
- Reduza o número de imagens no dataset
- Use menos épocas para testar primeiro

### Loss não diminui
- Tente learning rate menor (ex: 0.0001)
- Verifique se o dataset tem variação suficiente
- Aumente o número de épocas

## 🎯 Próximos Passos

1. ✅ Execute o exemplo básico
2. 📝 Experimente com diferentes hiperparâmetros
3. 🎨 Prepare seu próprio dataset de PixelArt
4. 🚀 Treine seu primeiro modelo real

## 📚 Recursos Adicionais

- [Visão do Projeto](../VISION/README.md)
- [Exemplos](../examples/)
- [API Reference](./API.md) _(em breve)_

---

Dúvidas ou problemas? Abra uma issue no repositório!
