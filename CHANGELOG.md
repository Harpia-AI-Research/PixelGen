# Changelog

Todas as mudanças notáveis neste projeto serão documentadas neste arquivo.

O formato é baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/),
e este projeto adere ao [Semantic Versioning](https://semver.org/lang/pt-BR/).

## [Unreleased]

### Added
- Parser de imagens PNG e JPEG
- Suporte a carregamento de imagens de diretórios
- Redimensionamento automático de imagens para 32x32
- Exemplo de treinamento com imagens reais (`train-with-images.ts`)
- Conversão automática de formatos de imagem para tensores
- Dependências: `pngjs` e `jpeg-js` para processamento de imagens

---

## [0.1.0] - 2026-08-10

### Adicionado

#### Core
- Implementação da classe `Tensor` com suporte a operações n-dimensionais
- Sistema de autograd (diferenciação automática)
- Funções de ativação: ReLU, Leaky ReLU, Sigmoid, Tanh, Softmax
- Uso de `Float32Array` para performance

#### Layers
- `Dense` - Camada totalmente conectada
- `Conv2D` - Convolução 2D com padding e stride configuráveis
- `MaxPool2D` e `AvgPool2D` - Pooling layers
- `Upsample2D` - Upsampling para decoder

#### Model
- `PixelGenModel` - Autoencoder convolucional fixo para PixelArt
- Arquitetura encoder-decoder otimizada para imagens 32x32
- Suporte a metadados e configuração

#### Training
- Otimizadores: `SGD` (com momentum) e `Adam`
- Loss functions: MSE, MAE, BCE, e PixelArt-specific loss
- Classe `Trainer` com loop de treinamento completo
- Tracking de estatísticas de treinamento

#### Data
- `ImageDataset` para gerenciar datasets
- Utilitários de normalização e denormalização
- Função para criar datasets de teste
- Suporte a batchificação e shuffle

#### I/O
- Formato `.pgm` (Pixel Gen Model) - binário com header JSON
- Funções para salvar e carregar modelos
- Função para inspecionar modelos sem carregar pesos completos

#### Documentação
- README principal com guia de uso
- Documento de visão do projeto (VISION/)
- Getting Started guide
- Documentação de arquitetura técnica
- Exemplos: básico e avançado
- Licença MIT

#### Tooling
- Configuração TypeScript
- Scripts npm para build, dev, e exemplos
- .gitignore configurado
- package.json com metadados completos

### Notas

Esta é a primeira versão pública do PixelGen. A arquitetura é fixa e focada em demonstrar o conceito de ML para PixelArt construído do zero.

### Limitações Conhecidas

- Parser de imagens PNG/JPG não implementado (usa dados de teste)
- Arquitetura fixa em 32x32 pixels
- Apenas RGB (3 canais)
- Autograd simplificado
- Performance limitada por execução em CPU puro

### Próximos Passos

Veja o roadmap no README para features planejadas para versões futuras.

---

## [Unreleased]

### Planejado para v0.2.0
- [ ] Parser de imagens PNG
- [ ] CLI para treino e geração
- [ ] Suporte a diferentes tamanhos de entrada
- [ ] Otimizações de performance (multi-threading)

---

[0.1.0]: https://github.com/user/pixelgen/releases/tag/v0.1.0
