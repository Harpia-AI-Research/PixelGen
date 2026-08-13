import { Backend, Conv2DForwardResult, Conv2DBackwardResult, PoolForwardResult, UpsampleForwardResult, MatmulResult } from './Backend';
import { Tensor } from '../Tensor';

export class WasmBackend implements Backend {
  private wasm: any;
  private ready: boolean = false;

  constructor() {
    this.wasm = null;
  }

  async init(): Promise<void> {
    if (this.ready) return;

    try {
      const bindings = require('./WasmBindings');
      this.wasm = await bindings.loadWasmModule();
      this.ready = true;
    } catch (error) {
      throw new Error(`Failed to initialize WASM backend: ${error}`);
    }
  }

  async conv2dForward(
    input: Tensor,
    filters: Tensor,
    bias: Tensor,
    outChannels: number,
    kernelSize: number,
    stride: number,
    padding: number
  ): Promise<Conv2DForwardResult> {
    await this.ensureReady();

    const result = this.wasm.conv2d_forward(
      input.data,
      input.shape,
      filters.data,
      filters.shape,
      bias.data,
      bias.shape,
      outChannels,
      kernelSize,
      stride,
      padding
    );

    const output = new Tensor(
      new Float32Array(result.output),
      Array.from(result.outputShape),
      true
    );
    const paddedInput = new Tensor(
      new Float32Array(result.paddedInput),
      Array.from(result.paddedInputShape),
      input.requiresGrad
    );

    return { output, paddedInput };
  }

  async conv2dBackward(
    input: Tensor,
    paddedInput: Tensor,
    filters: Tensor,
    outputGrad: Tensor,
    outChannels: number,
    kernelSize: number,
    stride: number,
    padding: number,
    batch: number,
    channels: number,
    height: number,
    width: number,
    outHeight: number,
    outWidth: number
  ): Promise<Conv2DBackwardResult> {
    await this.ensureReady();

    const result = this.wasm.conv2d_backward(
      input.data,
      input.shape,
      paddedInput.data,
      paddedInput.shape,
      filters.data,
      filters.shape,
      outputGrad.data,
      outputGrad.shape,
      outChannels,
      kernelSize,
      stride,
      padding,
      batch,
      channels,
      height,
      width,
      outHeight,
      outWidth
    );

    return {
      inputGrad: new Tensor(new Float32Array(result.inputGrad), Array.from(result.inputGradShape), false),
      filterGrad: new Tensor(new Float32Array(result.filterGrad), Array.from(result.filterGradShape), false),
      biasGrad: new Tensor(new Float32Array(result.biasGrad), Array.from(result.biasGradShape), false),
    };
  }

  async poolForward(input: Tensor, poolSize: number, stride: number): Promise<PoolForwardResult> {
    await this.ensureReady();

    const result = this.wasm.maxpool_forward(input.data, input.shape, poolSize, stride);
    const output = new Tensor(new Float32Array(result.output), Array.from(result.outputShape), input.requiresGrad);

    return { output, argmax: new Int32Array(result.argmax) };
  }

  async avgPoolForward(input: Tensor, poolSize: number, stride: number): Promise<PoolForwardResult> {
    await this.ensureReady();

    const result = this.wasm.avgpool_forward(input.data, input.shape, poolSize, stride);
    const output = new Tensor(new Float32Array(result.output), Array.from(result.outputShape), input.requiresGrad);

    return { output };
  }

  async upsampleForward(input: Tensor, scale: number): Promise<UpsampleForwardResult> {
    await this.ensureReady();

    const result = this.wasm.upsample_forward(input.data, input.shape, scale);
    const output = new Tensor(new Float32Array(result.output), Array.from(result.outputShape), input.requiresGrad);

    return { output };
  }

  async matmul(a: Tensor, b: Tensor): Promise<MatmulResult> {
    await this.ensureReady();

    const result = this.wasm.matmul(a.data, a.shape, b.data, b.shape);
    const output = new Tensor(new Float32Array(result.output), Array.from(result.outputShape), a.requiresGrad || b.requiresGrad);

    return { output };
  }

  private async ensureReady(): Promise<void> {
    if (!this.ready) {
      await this.init();
    }
  }
}
