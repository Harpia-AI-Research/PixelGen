import { Tensor } from './Tensor';

/**
 * Activation functions for neural networks
 */

export function relu(input: Tensor): Tensor {
  const result = new Tensor(
    new Float32Array(input.data.length),
    [...input.shape],
    input.requiresGrad
  );

  for (let i = 0; i < input.data.length; i++) {
    result.data[i] = Math.max(0, input.data[i]);
  }

  if (input.requiresGrad) {
    result._backward = () => {
      if (!result.grad) return;
      if (!input.grad) input.grad = Tensor.zerosLike(input);
      
      for (let i = 0; i < input.data.length; i++) {
        input.grad.data[i] += input.data[i] > 0 ? result.grad.data[i] : 0;
      }
    };
  }

  return result;
}

export function leakyRelu(input: Tensor, alpha: number = 0.01): Tensor {
  const result = new Tensor(
    new Float32Array(input.data.length),
    [...input.shape],
    input.requiresGrad
  );

  for (let i = 0; i < input.data.length; i++) {
    result.data[i] = input.data[i] > 0 ? input.data[i] : alpha * input.data[i];
  }

  if (input.requiresGrad) {
    result._backward = () => {
      if (!result.grad) return;
      if (!input.grad) input.grad = Tensor.zerosLike(input);
      
      for (let i = 0; i < input.data.length; i++) {
        input.grad.data[i] += input.data[i] > 0 ? result.grad.data[i] : alpha * result.grad.data[i];
      }
    };
  }

  return result;
}

export function sigmoid(input: Tensor): Tensor {
  const result = new Tensor(
    new Float32Array(input.data.length),
    [...input.shape],
    input.requiresGrad
  );

  for (let i = 0; i < input.data.length; i++) {
    result.data[i] = 1 / (1 + Math.exp(-input.data[i]));
  }

  if (input.requiresGrad) {
    result._backward = () => {
      if (!result.grad) return;
      if (!input.grad) input.grad = Tensor.zerosLike(input);
      
      for (let i = 0; i < input.data.length; i++) {
        const s = result.data[i];
        input.grad.data[i] += result.grad.data[i] * s * (1 - s);
      }
    };
  }

  return result;
}

export function tanh(input: Tensor): Tensor {
  const result = new Tensor(
    new Float32Array(input.data.length),
    [...input.shape],
    input.requiresGrad
  );

  for (let i = 0; i < input.data.length; i++) {
    result.data[i] = Math.tanh(input.data[i]);
  }

  if (input.requiresGrad) {
    result._backward = () => {
      if (!result.grad) return;
      if (!input.grad) input.grad = Tensor.zerosLike(input);
      
      for (let i = 0; i < input.data.length; i++) {
        const t = result.data[i];
        input.grad.data[i] += result.grad.data[i] * (1 - t * t);
      }
    };
  }

  return result;
}

/**
 * Softmax activation (typically used in output layer)
 * Operates on the last dimension
 */
export function softmax(input: Tensor): Tensor {
  const result = new Tensor(
    new Float32Array(input.data.length),
    [...input.shape],
    input.requiresGrad
  );

  // Find max for numerical stability
  let max = -Infinity;
  for (let i = 0; i < input.data.length; i++) {
    if (input.data[i] > max) max = input.data[i];
  }

  // Compute exp and sum
  let sum = 0;
  for (let i = 0; i < input.data.length; i++) {
    result.data[i] = Math.exp(input.data[i] - max);
    sum += result.data[i];
  }

  // Normalize
  for (let i = 0; i < input.data.length; i++) {
    result.data[i] /= sum;
  }

  return result;
}
