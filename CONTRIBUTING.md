# Contribuindo para o PixelGen

Obrigado por considerar contribuir para o PixelGen! Este documento fornece diretrizes para contribuições.

## 🎯 Como Contribuir

### Reportando Bugs

Se você encontrou um bug, abra uma issue com:

1. **Título claro** - Descreva o problema em poucas palavras
2. **Descrição detalhada** - Explique o que aconteceu
3. **Passos para reproduzir** - Como reproduzir o problema
4. **Comportamento esperado** - O que deveria acontecer
5. **Ambiente** - Versão do Node.js, sistema operacional, etc.
6. **Código exemplo** - Se possível, um snippet mínimo que reproduz o problema

### Sugerindo Features

Para sugerir uma nova feature:

1. **Verifique issues existentes** - Talvez já esteja planejado
2. **Descreva o caso de uso** - Por que essa feature é útil?
3. **Proponha uma solução** - Como você imagina que funcionaria?
4. **Considere alternativas** - Outras formas de resolver o problema?

### Pull Requests

1. **Fork o repositório**
2. **Crie uma branch** - `git checkout -b feature/minha-feature`
3. **Faça suas mudanças** - Siga as convenções de código
4. **Teste suas mudanças** - Certifique-se que funciona
5. **Commit suas mudanças** - Mensagens claras e descritivas
6. **Push para o fork** - `git push origin feature/minha-feature`
7. **Abra um Pull Request** - Descreva suas mudanças

## 📝 Convenções de Código

### TypeScript

- Use **TypeScript strict mode**
- Prefira **interfaces** para tipos públicos
- Use **tipos explícitos** em parâmetros de função
- Evite `any` - use tipos específicos ou `unknown`

### Nomenclatura

```typescript
// Classes: PascalCase
class PixelGenModel { }

// Funções e variáveis: camelCase
function trainModel() { }
const learningRate = 0.001;

// Constantes: UPPER_SNAKE_CASE
const DEFAULT_BATCH_SIZE = 4;

// Tipos e interfaces: PascalCase
interface ModelConfig { }
type TensorShape = number[];
```

### Comentários

```typescript
/**
 * Descrição da função
 * @param input - Descrição do parâmetro
 * @returns Descrição do retorno
 */
function forward(input: Tensor): Tensor {
  // Comentários inline quando necessário
  return output;
}
```

### Estrutura de Arquivos

```
src/
├── core/          # Fundação (Tensor, autograd)
├── layers/        # Building blocks (Conv2D, Dense)
├── model/         # Arquiteturas de modelos
├── training/      # Otimizadores, loss, trainer
├── data/          # Data loading e preprocessing
└── io/            # Salvar/carregar modelos
```

## 🧪 Testes

Por enquanto, o projeto não tem suite de testes automatizados. Contribuições para adicionar testes são muito bem-vindas!

### Testando Manualmente

```bash
# Build
npm run build

# Executar exemplos
npm run example:basic
npm run example:advanced
```

## 📚 Documentação

Se você adicionar uma nova feature, por favor:

1. Atualize o README se necessário
2. Adicione comentários no código
3. Crie ou atualize documentação em `docs/`
4. Adicione exemplos se apropriado

## 🎨 Áreas Que Precisam de Ajuda

### Alta Prioridade

- [ ] **Parser de imagens PNG/JPG** - Implementar carregamento nativo de imagens
- [ ] **Testes automatizados** - Criar suite de testes com Jest ou similar
- [ ] **CLI** - Interface de linha de comando para treino/geração
- [ ] **Otimizações de performance** - SIMD, multi-threading, etc.

### Média Prioridade

- [ ] **Validação de dados** - Melhor validação de inputs
- [ ] **Logging** - Sistema de logging estruturado
- [ ] **Visualização** - Ferramentas para visualizar treinamento
- [ ] **Documentação** - Mais exemplos e tutoriais

### Baixa Prioridade

- [ ] **Suporte a RGBA** - Transparência em imagens
- [ ] **Export para outros formatos** - ONNX, TensorFlow.js
- [ ] **Modelos pre-treinados** - Disponibilizar modelos base

## 🤔 Perguntas?

Se tiver dúvidas sobre como contribuir:

1. Abra uma issue com a tag `question`
2. Descreva sua dúvida claramente
3. Responderemos o mais rápido possível

## 📜 Código de Conduta

### Nossa Promessa

Estamos comprometidos em proporcionar uma experiência acolhedora e inclusiva para todos.

### Comportamento Esperado

- Use linguagem acolhedora e inclusiva
- Seja respeitoso com diferentes pontos de vista
- Aceite críticas construtivas
- Foque no que é melhor para a comunidade
- Mostre empatia com outros membros

### Comportamento Inaceitável

- Linguagem ou imagens sexualizadas
- Trolling, comentários insultuosos ou depreciativos
- Assédio público ou privado
- Publicar informações privadas de outros sem permissão
- Outra conduta que seria considerada inadequada em ambiente profissional

## 📄 Licença

Ao contribuir, você concorda que suas contribuições serão licenciadas sob a mesma licença BSD-3-Clause do projeto.

---

Obrigado por ajudar a tornar o PixelGen melhor! 🎨✨
