"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const PixelGenModel_1 = require("../src/model/PixelGenModel");
const backend_1 = require("../src/core/backend");
const Tensor_1 = require("../src/core/Tensor");
async function main() {
    console.log('=== PixelGen WASM Backend Validation ===\n');
    const wasmBackend = new backend_1.WasmBackend();
    console.log('Initializing WASM backend...');
    try {
        await wasmBackend.init();
        console.log('WASM backend initialized successfully.\n');
    }
    catch (error) {
        console.error('Failed to initialize WASM backend:', error);
        process.exit(1);
    }
    console.log('Testing CPU vs WASM parity...\n');
    const seed = 42;
    const random = (s) => {
        const x = Math.sin(s) * 10000;
        return x - Math.floor(x);
    };
    let seedCounter = seed;
    const cpuModel = new PixelGenModel_1.PixelGenModel(3, 3);
    const wasmModel = new PixelGenModel_1.PixelGenModel(3, 3, wasmBackend);
    const inputData = new Float32Array(1 * 3 * 32 * 32);
    for (let i = 0; i < inputData.length; i++) {
        inputData[i] = random(seedCounter++) * 2 - 1;
    }
    const input = new Tensor_1.Tensor(inputData, [1, 3, 32, 32]);
    console.log('Running forward pass on CPU...');
    const cpuOutput = await cpuModel.forward(input);
    console.log(`CPU output shape: ${cpuOutput.shape}`);
    console.log(`CPU output[0][0][0][0]: ${cpuOutput.data[0].toFixed(6)}`);
    console.log('\nRunning forward pass on WASM...');
    const wasmOutput = await wasmModel.forward(input);
    console.log(`WASM output shape: ${wasmOutput.shape}`);
    console.log(`WASM output[0][0][0][0]: ${wasmOutput.data[0].toFixed(6)}`);
    let maxDiff = 0;
    for (let i = 0; i < cpuOutput.data.length; i++) {
        const diff = Math.abs(cpuOutput.data[i] - wasmOutput.data[i]);
        if (diff > maxDiff)
            maxDiff = diff;
    }
    console.log(`\nMax absolute difference: ${maxDiff.toFixed(6)}`);
    if (maxDiff < 1e-5) {
        console.log('Parity test PASSED: CPU and WASM outputs match.\n');
    }
    else {
        console.log('Parity test FAILED: CPU and WASM outputs differ.\n');
        process.exit(1);
    }
    console.log('Testing backward pass on CPU...');
    const targetData = new Float32Array(1 * 3 * 32 * 32);
    for (let i = 0; i < targetData.length; i++) {
        targetData[i] = random(seedCounter++) * 2 - 1;
    }
    const target = new Tensor_1.Tensor(targetData, [1, 3, 32, 32]);
    cpuOutput.grad = cpuOutput.clone();
    for (let i = 0; i < cpuOutput.grad.data.length; i++) {
        cpuOutput.grad.data[i] = 2 * (cpuOutput.data[i] - target.data[i]) / cpuOutput.data.length;
    }
    cpuOutput.backward();
    const cpuFilterGrad = cpuModel['conv1'].filters.grad;
    const cpuBiasGrad = cpuModel['conv1'].bias.grad;
    console.log('Testing backward pass on WASM...');
    wasmOutput.grad = wasmOutput.clone();
    for (let i = 0; i < wasmOutput.grad.data.length; i++) {
        wasmOutput.grad.data[i] = 2 * (wasmOutput.data[i] - target.data[i]) / wasmOutput.data.length;
    }
    wasmOutput.backward();
    const wasmFilterGrad = wasmModel['conv1'].filters.grad;
    const wasmBiasGrad = wasmModel['conv1'].bias.grad;
    let filterGradDiff = 0;
    for (let i = 0; i < cpuFilterGrad.data.length; i++) {
        const diff = Math.abs(cpuFilterGrad.data[i] - wasmFilterGrad.data[i]);
        if (diff > filterGradDiff)
            filterGradDiff = diff;
    }
    let biasGradDiff = 0;
    for (let i = 0; i < cpuBiasGrad.data.length; i++) {
        const diff = Math.abs(cpuBiasGrad.data[i] - wasmBiasGrad.data[i]);
        if (diff > biasGradDiff)
            biasGradDiff = diff;
    }
    console.log(`Filter grad max diff: ${filterGradDiff.toFixed(6)}`);
    console.log(`Bias grad max diff: ${biasGradDiff.toFixed(6)}`);
    if (filterGradDiff < 1e-4 && biasGradDiff < 1e-4) {
        console.log('Backward parity test PASSED.\n');
    }
    else {
        console.log('Backward parity test FAILED.\n');
        process.exit(1);
    }
    console.log('All validation tests completed successfully.');
}
main().catch(error => {
    console.error('Validation failed:', error);
    process.exit(1);
});
//# sourceMappingURL=wasm-backend-test.js.map