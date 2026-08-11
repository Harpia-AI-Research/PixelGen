import * as fs from 'fs';
import * as path from 'path';
import { PNG } from 'pngjs';
import * as jpeg from 'jpeg-js';
import { Tensor } from '../core/Tensor';

/**
 * Image data loader for PixelArt training
 * Supports PNG and JPG/JPEG formats
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
    
    // Load and parse each image
    for (const imagePath of this.imagePaths) {
      try {
        const tensor = this.loadImage(imagePath);
        this.images.push(tensor);
      } catch (error) {
        console.warn(`Failed to load image ${imagePath}:`, error);
      }
    }
    
    console.log(`Successfully loaded ${this.images.length} images`);
  }

  /**
   * Load a single image file (PNG or JPG)
   */
  private loadImage(imagePath: string): Tensor {
    const ext = path.extname(imagePath).toLowerCase();
    
    let imageData: ImageData;
    
    if (ext === '.png') {
      imageData = this.loadPNG(imagePath);
    } else if (ext === '.jpg' || ext === '.jpeg') {
      imageData = this.loadJPEG(imagePath);
    } else {
      throw new Error(`Unsupported image format: ${ext}`);
    }

    // Resize to target size if needed
    if (imageData.width !== this.config.targetSize || imageData.height !== this.config.targetSize) {
      imageData = this.resizeImage(imageData, this.config.targetSize, this.config.targetSize);
    }

    // Convert to tensor [1, channels, height, width]
    const tensor = this.imageDataToTensor(imageData);
    
    // Normalize if configured
    if (this.config.normalize) {
      return normalizeTensor(tensor);
    }
    
    return tensor;
  }

  /**
   * Load PNG image
   */
  private loadPNG(imagePath: string): ImageData {
    const buffer = fs.readFileSync(imagePath);
    const png = PNG.sync.read(buffer);
    
    return {
      width: png.width,
      height: png.height,
      channels: png.data.length / (png.width * png.height),
      data: png.data,
    };
  }

  /**
   * Load JPEG image
   */
  private loadJPEG(imagePath: string): ImageData {
    const buffer = fs.readFileSync(imagePath);
    const jpg = jpeg.decode(buffer);
    
    return {
      width: jpg.width,
      height: jpg.height,
      channels: 4, // JPEG decoder returns RGBA
      data: jpg.data,
    };
  }

  /**
   * Resize image using nearest neighbor interpolation
   */
  private resizeImage(imageData: ImageData, newWidth: number, newHeight: number): ImageData {
    const { width, height, channels, data } = imageData;
    const newData = new Uint8Array(newWidth * newHeight * channels);

    const xRatio = width / newWidth;
    const yRatio = height / newHeight;

    for (let y = 0; y < newHeight; y++) {
      for (let x = 0; x < newWidth; x++) {
        const srcX = Math.floor(x * xRatio);
        const srcY = Math.floor(y * yRatio);
        
        const srcIdx = (srcY * width + srcX) * channels;
        const dstIdx = (y * newWidth + x) * channels;
        
        for (let c = 0; c < channels; c++) {
          newData[dstIdx + c] = data[srcIdx + c];
        }
      }
    }

    return {
      width: newWidth,
      height: newHeight,
      channels,
      data: newData,
    };
  }

  /**
   * Convert ImageData to Tensor [1, channels, height, width]
   */
  private imageDataToTensor(imageData: ImageData): Tensor {
    const { width, height, channels: srcChannels, data } = imageData;
    const targetChannels = this.config.channels;

    // Allocate tensor data
    const tensorData = new Float32Array(targetChannels * height * width);

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const srcIdx = (y * width + x) * srcChannels;
        
        for (let c = 0; c < Math.min(targetChannels, srcChannels); c++) {
          const tensorIdx = c * height * width + y * width + x;
          tensorData[tensorIdx] = data[srcIdx + c];
        }
      }
    }

    return new Tensor(tensorData, [1, targetChannels, height, width]);
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
