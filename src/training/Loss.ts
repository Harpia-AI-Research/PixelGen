import { Tensor } from '../core/Tensor';

/**
 * Mean Squared Error loss and gradient
 */
export function mseLoss(predicted: Tensor, target: Tensor): number {
  if (predicted.data.length !== target.data.length) {
    throw new Error('Predicted and target tensors must have the same size');
  }

  let sum = 0;
  for (let i = 0; i < predicted.data.length; i++) {
    const diff = predicted.data[i] - target.data[i];
    sum += diff * diff;
  }

  return sum / predicted.data.length;
}

/**
 * MSE Loss gradient: dL/dPredicted = 2 * (predicted - target) / N
 */
export function mseLossGrad(predicted: Tensor, target: Tensor): Tensor {
  const grad = Tensor.zerosLike(predicted);
  const scale = 2.0 / predicted.data.length;
  
  for (let i = 0; i < predicted.data.length; i++) {
    grad.data[i] = scale * (predicted.data[i] - target.data[i]);
  }
  
  return grad;
}

/**
 * Mean Absolute Error loss
 */
export function maeLoss(predicted: Tensor, target: Tensor): number {
  if (predicted.data.length !== target.data.length) {
    throw new Error('Predicted and target tensors must have the same size');
  }

  let sum = 0;
  for (let i = 0; i < predicted.data.length; i++) {
    sum += Math.abs(predicted.data[i] - target.data[i]);
  }

  return sum / predicted.data.length;
}

/**
 * MAE Loss gradient: dL/dPredicted = sign(predicted - target) / N
 * Uses a subgradient of 0 when the difference is exactly 0.
 */
export function maeLossGrad(predicted: Tensor, target: Tensor): Tensor {
  if (predicted.data.length !== target.data.length) {
    throw new Error('Predicted and target tensors must have the same size');
  }

  const grad = Tensor.zerosLike(predicted);
  const scale = 1.0 / predicted.data.length;

  for (let i = 0; i < predicted.data.length; i++) {
    const diff = predicted.data[i] - target.data[i];
    grad.data[i] = scale * (diff > 0 ? 1 : diff < 0 ? -1 : 0);
  }

  return grad;
}

/**
 * Binary Cross Entropy loss
 */
export function bceLoss(predicted: Tensor, target: Tensor): number {
  if (predicted.data.length !== target.data.length) {
    throw new Error('Predicted and target tensors must have the same size');
  }

  let sum = 0;
  const epsilon = 1e-7; // For numerical stability

  for (let i = 0; i < predicted.data.length; i++) {
    const p = Math.max(epsilon, Math.min(1 - epsilon, predicted.data[i]));
    sum += -(target.data[i] * Math.log(p) + (1 - target.data[i]) * Math.log(1 - p));
  }

  return sum / predicted.data.length;
}

/**
 * BCE Loss gradient: dL/dPredicted = -(target/p - (1-target)/(1-p)) / N
 * Uses the same clamping as the forward pass for numerical stability.
 */
export function bceLossGrad(predicted: Tensor, target: Tensor): Tensor {
  if (predicted.data.length !== target.data.length) {
    throw new Error('Predicted and target tensors must have the same size');
  }

  const grad = Tensor.zerosLike(predicted);
  const epsilon = 1e-7;
  const scale = 1.0 / predicted.data.length;

  for (let i = 0; i < predicted.data.length; i++) {
    const p = Math.max(epsilon, Math.min(1 - epsilon, predicted.data[i]));
    grad.data[i] = scale * -(target.data[i] / p - (1 - target.data[i]) / (1 - p));
  }

  return grad;
}

/**
 * Pixel-specific loss for PixelArt
 * Combines MSE with edge preservation
 */
export function pixelArtLoss(
  predicted: Tensor,
  target: Tensor,
  edgeWeight: number = 0.3
): number {
  if (predicted.shape.length !== 4 || target.shape.length !== 4) {
    throw new Error('PixelArt loss expects 4D tensors [batch, channels, height, width]');
  }

  const mseLossValue = mseLoss(predicted, target);

  // Calculate edge loss (penalize blurry edges)
  const edgeLoss = calculateEdgeLoss(predicted, target);

  return mseLossValue + edgeWeight * edgeLoss;
}

/**
 * PixelArt Loss gradient
 */
export function pixelArtLossGrad(
  predicted: Tensor,
  target: Tensor,
  edgeWeight: number = 0.3
): Tensor {
  // MSE gradient
  const mseGrad = mseLossGrad(predicted, target);
  
  // Edge loss gradient
  const edgeGrad = calculateEdgeLossGrad(predicted, target);
  
  // Combine gradients
  const totalGrad = Tensor.zerosLike(predicted);
  for (let i = 0; i < totalGrad.data.length; i++) {
    totalGrad.data[i] = mseGrad.data[i] + edgeWeight * edgeGrad.data[i];
  }
  
  return totalGrad;
}

/**
 * Calculate edge preservation loss
 * Encourages sharp transitions between colors
 */
function calculateEdgeLoss(predicted: Tensor, target: Tensor): number {
  const [batch, channels, height, width] = predicted.shape;
  let totalLoss = 0;
  let count = 0;

  for (let b = 0; b < batch; b++) {
    for (let c = 0; c < channels; c++) {
      for (let h = 0; h < height - 1; h++) {
        for (let w = 0; w < width - 1; w++) {
          const idx = b * channels * height * width + c * height * width + h * width + w;
          const idxRight = idx + 1;
          const idxDown = idx + width;

          // Horizontal gradient
          const predGradH = Math.abs(predicted.data[idx] - predicted.data[idxRight]);
          const targetGradH = Math.abs(target.data[idx] - target.data[idxRight]);
          totalLoss += Math.abs(predGradH - targetGradH);

          // Vertical gradient
          const predGradV = Math.abs(predicted.data[idx] - predicted.data[idxDown]);
          const targetGradV = Math.abs(target.data[idx] - target.data[idxDown]);
          totalLoss += Math.abs(predGradV - targetGradV);

          count += 2;
        }
      }
    }
  }

  return totalLoss / count;
}

/**
 * Calculate edge loss gradient
 */
function calculateEdgeLossGrad(predicted: Tensor, target: Tensor): Tensor {
  const [batch, channels, height, width] = predicted.shape;
  const grad = Tensor.zerosLike(predicted);
  const count = batch * channels * (height - 1) * (width - 1) * 2;
  const scale = 1.0 / count;

  // Subgradient of |x|: 0 when x is exactly 0
  const sign = (v: number): number => (v > 0 ? 1 : v < 0 ? -1 : 0);
  // Subgradient of |a - b|: 0 when the values are equal
  const compareSign = (a: number, b: number): number => (a > b ? 1 : a < b ? -1 : 0);

  for (let b = 0; b < batch; b++) {
    for (let c = 0; c < channels; c++) {
      for (let h = 0; h < height - 1; h++) {
        for (let w = 0; w < width - 1; w++) {
          const idx = b * channels * height * width + c * height * width + h * width + w;
          const idxRight = idx + 1;
          const idxDown = idx + width;

          // Horizontal gradient contribution
          const predDiffH = predicted.data[idx] - predicted.data[idxRight];
          const targetDiffH = target.data[idx] - target.data[idxRight];
          const predGradH = Math.abs(predDiffH);
          const targetGradH = Math.abs(targetDiffH);
          const signH = compareSign(predGradH, targetGradH);
          const gradSignH = sign(predDiffH);

          grad.data[idx] += scale * signH * gradSignH;
          grad.data[idxRight] += scale * signH * (-gradSignH);

          // Vertical gradient contribution
          const predDiffV = predicted.data[idx] - predicted.data[idxDown];
          const targetDiffV = target.data[idx] - target.data[idxDown];
          const predGradV = Math.abs(predDiffV);
          const targetGradV = Math.abs(targetDiffV);
          const signV = compareSign(predGradV, targetGradV);
          const gradSignV = sign(predDiffV);

          grad.data[idx] += scale * signV * gradSignV;
          grad.data[idxDown] += scale * signV * (-gradSignV);
        }
      }
    }
  }

  return grad;
}
