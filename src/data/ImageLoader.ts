import * as fs from 'fs';
import * as path from 'path';
import { Tensor } from '../core/Tensor';

/**
 * Image data loader for PixelArt training
 * Note: For v1, we'll use a simple approach with Node.js built-ins
 * In future versions, we can add PNG/JPG parsing
 */

export interface ImageData {
  width: number;
  height: number;
  channels: number;
  data: Uint8Array;
}

export interface DatasetConfig {
  targetSize: number; // Square images (e.g., 32x32)
  normalize: boolean; // Normalize to [0, 1]
  channels: number; // RGB = 3, RGBA = 4
}

export class ImageDataset {
  private images: Tensor[] = [];
  private config: DatasetConfig;
  private imagePaths: string[] = [];

  constructor(config: DatasetConfig) {
    this.config = config;
  }

  /**
   * Load images from a directory
   * For v1: expects raw image data files or will be extended with image parsing
   */
  loadFromDirectory(directoryPath: string): void {
    if (!fs.existsSync(directoryPath)) {
      throw new Error(`Directory not found: ${directoryPath}`);
    }

    const files = fs.readdirSync(directoryPath);
    const imageFiles = files.filter(f => 
      f.toLowerCase().endsWith('.png') || 
      f.toLowerCase().endsWith('.jpg') || 
      f.toLowerCase().endsWith('.jpeg')
    );

    if (imageFiles.length === 0) {
      throw new Error(`No image files found in ${directoryPath}`);
    }

    console.log(`Found ${imageFiles.length} images in ${directoryPath}`);
    
    this.imagePaths = imageFiles.map(f => path.join(directoryPath, f));
    
    // Note: Actual image parsing would be implemented here
    // For now, this is a placeholder that shows the structure
    console.log('Note: Image parsing not yet implemented in v1');
    console.log('You will need to provide pre-processed tensor data or implement PNG/JPG parsing');
  }

  /**
   * Add pre-processed tensor directly (useful for testing)
   */
  addTensor(tensor: Tensor): void {
    if (tensor.shape.length !== 4) {
      throw new Error('Image tensors must be 4D: [batch, channels, height, width]');
    }
    this.images.push(tensor);
  }

  /**
   * Get a batch of images
   */
  getBatch(batchSize: number, startIdx: number = 0): Tensor | null {
    if (startIdx >= this.images.length) {
      return null;
    }

    const endIdx = Math.min(startIdx + batchSize, this.images.length);
    const batchImages = this.images.slice(startIdx, endIdx);

    if (batchImages.length === 0) {
      return null;
    }

    // Concatenate along batch dimension
    // For simplicity, assuming all images have the same shape
    const [, channels, height, width] = batchImages[0].shape;
    const actualBatchSize = batchImages.length;

    const batchData = new Float32Array(actualBatchSize * channels * height * width);
    
    for (let i = 0; i < actualBatchSize; i++) {
      const img = batchImages[i];
      batchData.set(img.data, i * img.data.length);
    }

    return new Tensor(
      batchData,
      [actualBatchSize, channels, height, width],
      false
    );
  }

  /**
   * Get number of images in dataset
   */
  size(): number {
    return this.images.length;
  }

  /**
   * Shuffle the dataset
   */
  shuffle(): void {
    for (let i = this.images.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [this.images[i], this.images[j]] = [this.images[j], this.images[i]];
    }
  }

  /**
   * Get image paths
   */
  getImagePaths(): string[] {
    return [...this.imagePaths];
  }
}

/**
 * Utility: Normalize tensor values from [0, 255] to [0, 1]
 */
export function normalizeTensor(tensor: Tensor): Tensor {
  const normalized = tensor.clone();
  for (let i = 0; i < normalized.data.length; i++) {
    normalized.data[i] = normalized.data[i] / 255.0;
  }
  return normalized;
}

/**
 * Utility: Denormalize tensor values from [0, 1] to [0, 255]
 */
export function denormalizeTensor(tensor: Tensor): Tensor {
  const denormalized = tensor.clone();
  for (let i = 0; i < denormalized.data.length; i++) {
    denormalized.data[i] = Math.round(Math.max(0, Math.min(255, denormalized.data[i] * 255)));
  }
  return denormalized;
}

/**
 * Utility: Create a simple test dataset with random pixel data
 */
export function createTestDataset(
  numImages: number,
  channels: number,
  height: number,
  width: number
): ImageDataset {
  const dataset = new ImageDataset({
    targetSize: height,
    normalize: true,
    channels,
  });

  for (let i = 0; i < numImages; i++) {
    const data = new Float32Array(channels * height * width);
    for (let j = 0; j < data.length; j++) {
      data[j] = Math.random(); // Random values [0, 1]
    }
    const tensor = new Tensor(data, [1, channels, height, width]);
    dataset.addTensor(tensor);
  }

  return dataset;
}
