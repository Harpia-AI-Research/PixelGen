import * as fs from 'fs';
import { Tensor } from '../core/Tensor';
import { PixelGenModel, ModelMetadata } from '../model/PixelGenModel';

/**
 * Model file format (.pgm - Pixel Gen Model)
 * 
 * Structure:
 * [JSON Header with metadata]
 * [Separator: "---WEIGHTS---"]
 * [Binary weights data as Float32Array]
 */

const SEPARATOR = '---WEIGHTS---\n';

export interface PGMFile {
  metadata: ModelMetadata;
  hyperparameters: Hyperparameters;
  weightsBuffer: Buffer;
}

export interface Hyperparameters {
  learningRate: number;
  batchSize: number;
  epochs: number;
  optimizer: string;
}

/**
 * Save model to .pgm file
 */
export function saveModel(
  model: PixelGenModel,
  filepath: string,
  hyperparameters: Hyperparameters
): void {
  const metadata = model.getMetadata();
  const parameters = model.parameters();

  // Create header JSON
  const header = {
    metadata,
    hyperparameters,
    parameterShapes: parameters.map(p => p.shape),
    totalParameters: parameters.reduce((sum, p) => sum + p.data.length, 0),
  };

  const headerJson = JSON.stringify(header, null, 2);

  // Concatenate all parameter data into single Float32Array
  const totalSize = parameters.reduce((sum, p) => sum + p.data.length, 0);
  const allWeights = new Float32Array(totalSize);
  
  let offset = 0;
  for (const param of parameters) {
    allWeights.set(param.data, offset);
    offset += param.data.length;
  }

  // Convert Float32Array to Buffer
  const weightsBuffer = Buffer.from(allWeights.buffer);

  // Write to file
  const headerBuffer = Buffer.from(headerJson + '\n' + SEPARATOR);
  const finalBuffer = Buffer.concat([headerBuffer, weightsBuffer]);
  
  fs.writeFileSync(filepath, finalBuffer);
  
  console.log(`Model saved to ${filepath}`);
  console.log(`Total parameters: ${totalSize}`);
  console.log(`File size: ${(finalBuffer.length / 1024).toFixed(2)} KB`);
}

/**
 * Load model from .pgm file
 */
export function loadModel(filepath: string): {
  model: PixelGenModel;
  hyperparameters: Hyperparameters;
} {
  const fileBuffer = fs.readFileSync(filepath);
  
  // Find separator
  const separatorBuffer = Buffer.from(SEPARATOR);
  const separatorIndex = fileBuffer.indexOf(separatorBuffer);
  
  if (separatorIndex === -1) {
    throw new Error('Invalid .pgm file format: separator not found');
  }

  // Extract header and weights
  const headerBuffer = fileBuffer.slice(0, separatorIndex);
  const weightsBuffer = fileBuffer.slice(separatorIndex + separatorBuffer.length);

  // Parse header
  const headerJson = headerBuffer.toString('utf-8');
  const header = JSON.parse(headerJson);

  // Validate version
  if (!header.metadata || !header.metadata.version) {
    throw new Error('Invalid .pgm file: missing version information');
  }

  // Create model
  const model = new PixelGenModel(
    header.metadata.inputChannels,
    header.metadata.outputChannels
  );

  // Convert weights buffer back to Float32Array
  const allWeights = new Float32Array(
    weightsBuffer.buffer,
    weightsBuffer.byteOffset,
    weightsBuffer.byteLength / Float32Array.BYTES_PER_ELEMENT
  );

  // Load weights into model parameters
  const parameters = model.parameters();
  let offset = 0;
  
  for (let i = 0; i < parameters.length; i++) {
    const param = parameters[i];
    const paramSize = param.data.length;
    
    if (offset + paramSize > allWeights.length) {
      throw new Error('Invalid .pgm file: insufficient weight data');
    }
    
    param.data.set(allWeights.slice(offset, offset + paramSize));
    offset += paramSize;
  }

  console.log(`Model loaded from ${filepath}`);
  console.log(`Architecture: ${header.metadata.architecture}`);
  console.log(`Total parameters: ${header.totalParameters}`);

  return {
    model,
    hyperparameters: header.hyperparameters,
  };
}

/**
 * Get model info without loading weights (useful for inspection)
 */
export function inspectModel(filepath: string): {
  metadata: ModelMetadata;
  hyperparameters: Hyperparameters;
  fileSize: number;
  totalParameters: number;
} {
  const fileBuffer = fs.readFileSync(filepath);
  const separatorBuffer = Buffer.from(SEPARATOR);
  const separatorIndex = fileBuffer.indexOf(separatorBuffer);
  
  if (separatorIndex === -1) {
    throw new Error('Invalid .pgm file format');
  }

  const headerBuffer = fileBuffer.slice(0, separatorIndex);
  const headerJson = headerBuffer.toString('utf-8');
  const header = JSON.parse(headerJson);

  return {
    metadata: header.metadata,
    hyperparameters: header.hyperparameters,
    fileSize: fileBuffer.length,
    totalParameters: header.totalParameters,
  };
}
