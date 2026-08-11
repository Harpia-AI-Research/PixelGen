import { Tensor } from '../core/Tensor';

/**
 * Base optimizer interface
 */
export interface Optimizer {
  step(parameters: Tensor[]): void;
  zeroGrad(parameters: Tensor[]): void;
}

/**
 * Stochastic Gradient Descent optimizer
 */
export class SGD implements Optimizer {
  private learningRate: number;
  private momentum: number;
  private velocities: Map<Tensor, Tensor>;

  constructor(learningRate: number = 0.01, momentum: number = 0) {
    this.learningRate = learningRate;
    this.momentum = momentum;
    this.velocities = new Map();
  }

  step(parameters: Tensor[]): void {
    for (const param of parameters) {
      if (!param.grad) continue;

      if (this.momentum > 0) {
        // Momentum SGD
        if (!this.velocities.has(param)) {
          this.velocities.set(param, Tensor.zerosLike(param));
        }

        const velocity = this.velocities.get(param)!;

        // v = momentum * v - lr * grad
        for (let i = 0; i < velocity.data.length; i++) {
          velocity.data[i] = this.momentum * velocity.data[i] - this.learningRate * param.grad.data[i];
        }

        // param = param + v
        for (let i = 0; i < param.data.length; i++) {
          param.data[i] += velocity.data[i];
        }
      } else {
        // Standard SGD
        for (let i = 0; i < param.data.length; i++) {
          param.data[i] -= this.learningRate * param.grad.data[i];
        }
      }
    }
  }

  zeroGrad(parameters: Tensor[]): void {
    for (const param of parameters) {
      param.zeroGrad();
    }
  }
}

/**
 * Adam optimizer (Adaptive Moment Estimation)
 */
export class Adam implements Optimizer {
  private learningRate: number;
  private beta1: number;
  private beta2: number;
  private epsilon: number;
  private t: number;
  private m: Map<Tensor, Tensor>; // First moment
  private v: Map<Tensor, Tensor>; // Second moment

  constructor(
    learningRate: number = 0.001,
    beta1: number = 0.9,
    beta2: number = 0.999,
    epsilon: number = 1e-8
  ) {
    this.learningRate = learningRate;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.epsilon = epsilon;
    this.t = 0;
    this.m = new Map();
    this.v = new Map();
  }

  step(parameters: Tensor[]): void {
    this.t += 1;

    for (const param of parameters) {
      if (!param.grad) continue;

      // Initialize moments if needed
      if (!this.m.has(param)) {
        this.m.set(param, Tensor.zerosLike(param));
        this.v.set(param, Tensor.zerosLike(param));
      }

      const m = this.m.get(param)!;
      const v = this.v.get(param)!;

      for (let i = 0; i < param.data.length; i++) {
        const grad = param.grad.data[i];

        // Update biased first moment estimate
        m.data[i] = this.beta1 * m.data[i] + (1 - this.beta1) * grad;

        // Update biased second raw moment estimate
        v.data[i] = this.beta2 * v.data[i] + (1 - this.beta2) * grad * grad;

        // Compute bias-corrected first moment estimate
        const mHat = m.data[i] / (1 - Math.pow(this.beta1, this.t));

        // Compute bias-corrected second raw moment estimate
        const vHat = v.data[i] / (1 - Math.pow(this.beta2, this.t));

        // Update parameters
        param.data[i] -= this.learningRate * mHat / (Math.sqrt(vHat) + this.epsilon);
      }
    }
  }

  zeroGrad(parameters: Tensor[]): void {
    for (const param of parameters) {
      param.zeroGrad();
    }
  }
}
