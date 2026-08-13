let wasmModule: any = null;

export async function loadWasmModule(): Promise<any> {
  if (wasmModule) {
    return wasmModule;
  }

  try {
    const wasmPath = require.resolve('pixelgen-wasm');
    const wasmDir = wasmPath.replace(/pixelgen_wasm\.js$/, '');
    
    const wasm = require(wasmPath);
    
    if (wasm.default && typeof wasm.default === 'function') {
      wasmModule = await wasm.default();
    } else if (typeof wasm === 'function') {
      wasmModule = await wasm();
    } else {
      wasmModule = wasm;
    }

    return wasmModule;
  } catch (error) {
    throw new Error(`Failed to load WASM module: ${error}`);
  }
}

export function isWasmAvailable(): boolean {
  try {
    require.resolve('pixelgen-wasm');
    return true;
  } catch {
    return false;
  }
}
