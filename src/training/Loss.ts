import { Tensor } from '../core/Tensor';

/**
 * Mean Squared Error loss
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
