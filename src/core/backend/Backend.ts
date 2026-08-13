import { Tensor } from '../Tensor';

export interface Conv2DForwardResult {
  output: Tensor;
  paddedInput: Tensor;
}

export interface Conv2DBackwardResult {
  inputGrad: Tensor;
  filterGrad: Tensor;
  biasGrad: Tensor;
}

export interface PoolForwardResult {
  output: Tensor;
  argmax?: Int32Array;
}

export interface UpsampleForwardResult {
  output: Tensor;
}

export interface MatmulResult {
  output: Tensor;
}

export interface Backend {
  conv2dForward(
    input: Tensor,
    filters: Tensor,
    bias: Tensor,
    outChannels: number,
    kernelSize: number,
    stride: number,
    padding: number
  ): Promise<Conv2DForwardResult>;

  conv2dBackward(
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
  ): Promise<Conv2DBackwardResult>;

  poolForward(
    input: Tensor,
    poolSize: number,
    stride: number
  ): Promise<PoolForwardResult>;

  avgPoolForward(
    input: Tensor,
    poolSize: number,
    stride: number
  ): Promise<PoolForwardResult>;

  upsampleForward(
    input: Tensor,
    scale: number
  ): Promise<UpsampleForwardResult>;

  matmul(a: Tensor, b: Tensor): Promise<MatmulResult>;
}
