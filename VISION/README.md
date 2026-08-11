# PixelGen - Visão do Projeto

## O que é PixelGen?

PixelGen é uma arquitetura de machine learning especializada em PixelArt, construída completamente do zero em TypeScript. A proposta é criar modelos que aprendam a gerar e transformar PixelArt através de exemplos, sem dependências externas de frameworks de ML.

## Filosofia

### Por que "baseado em regras"?
PixelArt não é arte livre — ela tem estrutura matemática clara:
- Pixels discretos em uma grade
- Paletas de cores limitadas
- Padrões repetitivos e simetrias
- Convenções visuais bem definidas (outlines, dithering, etc.)

PixelGen usa essas propriedades como **restrições arquiteturais** — o modelo não aprende a desenhar qualquer coisa, ele aprende a operar dentro do espaço específico de PixelArt.

### Por que TypeScript?
- Acessível para desenvolvedores web e game devs
- Código legível e tipado
- Possibilita integração futura em tooling JS

### Por que CPU?
Treinamento em CPU torna o projeto acessível para qualquer pessoa, sem necessidade de hardware especializado. O domínio restrito de PixelArt (imagens pequenas, regras claras) torna isso viável.

### Por que do zero?
Construir tudo do zero nos dá controle total sobre:
- Como o modelo aprende
- Como ele representa PixelArt internamente  
- Otimizações específicas para o domínio
- Zero dependências externas

## Objetivos

### v1.0 (MVP)
- ✨ Arquitetura fixa e opinada
- 📁 Dataset fornecido pelo usuário (pasta com PNGs)
- 🧠 Treinamento em CPU
- 💾 Formato próprio de modelo (.pgm - Pixel Gen Model)
- 🎨 Geração de PixelArt do zero após treinamento
- 🔄 Transformação de imagens normais em PixelArt

### Futuro
- 🔧 Framework flexível onde usuários definem arquiteturas customizadas
- ⚡ Otimizações de performance (multi-threading, SIMD)
- 🎯 Arquiteturas especializadas (sprites, tilesets, animações)
- 🌐 Bindings para outras linguagens

## Estrutura Técnica

### Componentes Core
- **Tensor Engine** — operações matriciais com TypedArrays
- **Autograd** — diferenciação automática para backpropagation
- **Layers** — Conv2D, Dense, Pooling, Activation
- **Optimizers** — SGD, Adam
- **Model** — arquitetura de rede encoder-decoder
- **Training Loop** — forward, backward, update
- **Data Loader** — preprocessamento de imagens
- **I/O** — salvar/carregar modelos .pgm

### Formato .pgm (Pixel Gen Model)
Binário com header JSON:
```
[JSON Header com metadados]
[Separador]
[Pesos binários em Float32Array]
```

O header contém:
- Versão da arquitetura
- Hiperparâmetros (tamanho input/output, paleta, learning rate)
- Shape dos tensores
- Metadados de treinamento

## Uso Esperado

```bash
# Treinar um modelo
pixelgen train --dataset ./meu-dataset --output modelo.pgm

# Gerar nova PixelArt
pixelgen generate --model modelo.pgm --output sprite.png

# Transformar imagem em PixelArt
pixelgen transform --model modelo.pgm --input foto.jpg --output pixel.png
```

## Princípios de Design

1. **Simplicidade primeiro** — código legível é mais importante que micro-otimizações
2. **Domínio restrito** — aproveitar as características únicas de PixelArt
3. **Acessibilidade** — qualquer pessoa com Node.js deve conseguir treinar
4. **Extensibilidade futura** — arquitetura preparada para evoluir para framework flexível

## Não-Objetivos (pelo menos na v1)

- ❌ Competir em performance com PyTorch/TensorFlow
- ❌ Ser um framework de ML de propósito geral
- ❌ Rodar em browser (foco é treinamento, não inferência web)
- ❌ Gerar arte fotorrealista ou domínios fora de PixelArt

---

**PixelGen** — Machine Learning para PixelArt, feito do zero, para todos.
