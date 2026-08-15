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
    result._prev.add(input);  // ADD INPUT TO GRAPH!
    
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
    result._prev.add(input);
    
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
    result._prev.add(input);
    
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
    result._prev.add(input);
    
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
 * Operates on the last dimension of the input shape.
 */
export function softmax(input: Tensor): Tensor {
  const result = new Tensor(
    new Float32Array(input.data.length),
    [...input.shape],
    input.requiresGrad
  );

  const lastDim = input.shape[input.shape.length - 1];
  const numRows = input.data.length / lastDim;

  for (let r = 0; r < numRows; r++) {
    const offset = r * lastDim;

    // Find max for numerical stability
    let max = -Infinity;
    for (let i = 0; i < lastDim; i++) {
      const v = input.data[offset + i];
      if (v > max) max = v;
    }

    // Compute exp and sum
    let sum = 0;
    for (let i = 0; i < lastDim; i++) {
      result.data[offset + i] = Math.exp(input.data[offset + i] - max);
      sum += result.data[offset + i];
    }

    // Normalize
    for (let i = 0; i < lastDim; i++) {
      result.data[offset + i] /= sum;
    }
  }

  if (input.requiresGrad) {
    result._prev.add(input);

    result._backward = () => {
      if (!result.grad) return;
      if (!input.grad) input.grad = Tensor.zerosLike(input);

      // dL/dx_i = s_i * (dL/ds_i - sum_k(dL/ds_k * s_k))
      for (let r = 0; r < numRows; r++) {
        const offset = r * lastDim;

        let dot = 0;
        for (let k = 0; k < lastDim; k++) {
          dot += result.grad.data[offset + k] * result.data[offset + k];
        }

        for (let i = 0; i < lastDim; i++) {
          input.grad.data[offset + i] +=
            result.data[offset + i] * (result.grad.data[offset + i] - dot);
        }
      }
    };
  }

  return result;
}
