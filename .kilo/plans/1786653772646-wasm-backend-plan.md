# Plano: Backend WebAssembly para PixelGen

## Objetivo
Adicionar suporte a backend WebAssembly (WASM) para acelerar treinamento e inferência dos modelos PixelGen, mantendo 100% de compatibilidade com a API TypeScript existente.

## Abordagem
**Híbrida Rust + TypeScript**: apenas kernels computacionalmente intensivos (Conv2D, pooling, matmul) são reescritos em Rust e compilados para WASM. O autograd, otimizadores, Trainer e sistema de modelos permanecem em TypeScript sem alteração.

## Escopo
- **Incluído**: Conv2D forward/backward, MaxPool2D forward, AvgPool2D forward, Upsample2D forward, matmul
- **Fora do escopo**: Reescrita do autograd em Rust, suporte a GPU/CUDA, browser runtime (foco é Node.js)

## Decisões
1. **Backend selecionado**: Opção 1 (híbrida), escolhida pelo usuário.
2. **Interface de backend**: nova classe abstrata `Backend` em `src/core/backend/`.
3. **Seleção de backend**: factory `Backend.create('cpu' | 'wasm')` com default `'cpu'`.
4. **Compatibilidade**: API pública do PixelGen permanece inalterada; backend é injetado via construtor/config.
5. **Formato de pesos**: `.pgm` continua idêntico; WASM não altera serialização.

## Tarefas

### Fase 1 — Infraestrutura de Backend
1. Criar `src/core/backend/Backend.ts`: interface abstrata com operações `conv2dForward`, `conv2dBackward`, `poolForward`, `upsampleForward`, `matmul`.
2. Criar `src/core/backend/CPUBackend.ts`: implementação usando loops TypeScript existentes (wraps do código atual).
3. Criar `src/core/backend/WasmBackend.ts`: stub que carrega módulo WASM e delega operações.
4. Criar `src/core/backend/index.ts` e atualizar `src/core/index.ts` para exportar backend.
5. Atualizar `src/layers/Conv2D.ts`, `Pooling.ts`, `Dense.ts` para aceitar `Backend` opcional e usá-lo nas operações de forward/backward.

### Fase 2 — Módulo Rust/WASM
1. Criar estrutura Rust em `wasm/` com `Cargo.toml`, `src/lib.rs`.
2. Adicionar dependências: `wasm-bindgen`, `js-sys`, `web-sys`.
3. Implementar kernels Rust:
   - `conv2d_forward`: loops unroll, cálculo de output e gradientes de peso/bias
   - `conv2d_backward`: roteamento de gradiente para input
   - `maxpool_forward`: argmax routing
   - `avgpool_forward`: média uniforme
   - `upsample_forward`: nearest-neighbor
   - `matmul`: multiplicação de matriz 2D
4. Compilar para WASM com `wasm-pack build --target nodejs`.

### Fase 3 — Integração TypeScript
1. Criar `src/core/backend/WasmBindings.ts`: carrega `.wasm` via `@aspect-build/rules_js` ou `fs` no Node.
2. Modificar `Tensor` para manter referência opcional ao `Backend` (não obrigatório para preservar compatibilidade).
3. Atualizar `PixelGenModel` para aceitar `Backend` no construtor e propagá-lo para as camadas.
4. Atualizar `Trainer` para aceitar `Backend` opcional na config ou construtor.

### Fase 4 — Empacotamento e Build
1. Configurar `package.json` com scripts `wasm:build`, `wasm:clean`.
2. Garantir que `npm run build` inclua/ gere artefatos WASM corretamente.
3. Atualizar `.npmignore` para incluir diretórios WASM necessários.

### Fase 5 — Validação
1. Testes de paridade numérica: comparar saídas de CPU vs WASM para Conv2D, Pooling, Upsample com seeds fixas.
2. Teste de treinamento completo: rodar `examples/basic-training.ts` com ambos backends e verificar loss convergente.
3. Teste de inferência: `examples/advanced-usage.ts` com WASM backend.
4. Benchmark simples: medir tempo de epoch em CPU vs WASM para batch size 4.

## Riscos
- **Precisão numérica**: Rust `f32` vs JS `number` (float64). Varrer diferenças e garantir tolerância.
- **Tamanho do WASM**: módulo inicial ~50-100KB; aceitável para Node.js.
- **Complexidade do backward**: roteamento de gradientes via WASM exige passagem de arrays planos; manter invariantes de shape no TS.

## Arquivos Alterados/Criados
- `src/core/backend/Backend.ts` (novo)
- `src/core/backend/CPUBackend.ts` (novo)
- `src/core/backend/WasmBackend.ts` (novo)
- `src/core/backend/WasmBindings.ts` (novo)
- `src/core/backend/index.ts` (novo)
- `src/core/index.ts` (modificado)
- `src/core/Tensor.ts` (modificado: backend opcional)
- `src/layers/Conv2D.ts` (modificado: usa backend)
- `src/layers/Pooling.ts` (modificado: usa backend)
- `src/layers/Dense.ts` (modificado: usa backend)
- `src/model/PixelGenModel.ts` (modificado: aceita backend)
- `src/training/Trainer.ts` (modificado: aceita backend)
- `wasm/Cargo.toml` (novo)
- `wasm/src/lib.rs` (novo)
- `package.json` (modificado: scripts WASM)

## Ordem de Execução
Fase 1 → Fase 2 → Fase 3 → Fase 4 → Fase 5. Fases 1-2 podem ser paralelizadas após definição da interface `Backend`.
