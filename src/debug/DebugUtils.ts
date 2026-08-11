import { Tensor } from '../core/Tensor';
import { PixelGenModel } from '../model/PixelGenModel';

/**
 * Debug utilities for PixelGen training
 */
export class DebugUtils {
  /**
   * Check if gradients are being computed for parameters
   */
  static checkGradients(parameters: Tensor[]): GradientInfo {
    let totalParams = 0;
    let paramsWithGrad = 0;
    let totalGradMagnitude = 0;
    let maxGrad = 0;
    let minGrad = Infinity;

    for (const param of parameters) {
      totalParams++;
      
      if (param.grad) {
        paramsWithGrad++;
        
        for (let i = 0; i < param.grad.data.length; i++) {
          const grad = Math.abs(param.grad.data[i]);
          totalGradMagnitude += grad;
          maxGrad = Math.max(maxGrad, grad);
          minGrad = Math.min(minGrad, grad);
        }
      }
    }

    const avgGradMagnitude = totalGradMagnitude / parameters.reduce((sum, p) => sum + p.data.length, 0);

    return {
      totalParams,
      paramsWithGrad,
      avgGradMagnitude,
      maxGrad,
      minGrad: minGrad === Infinity ? 0 : minGrad,
      hasGradients: paramsWithGrad > 0,
    };
  }

  /**
   * Print gradient statistics
   */
  static printGradientStats(parameters: Tensor[], label: string = 'Gradients'): void {
    const info = this.checkGradients(parameters);
    
    console.log(`\n[DEBUG] ${label}:`);
    console.log(`  Parameters: ${info.paramsWithGrad}/${info.totalParams} have gradients`);
    console.log(`  Avg magnitude: ${info.avgGradMagnitude.toExponential(4)}`);
    console.log(`  Max gradient: ${info.maxGrad.toExponential(4)}`);
    console.log(`  Min gradient: ${info.minGrad.toExponential(4)}`);
  }

  /**
   * Check parameter statistics
   */
  static checkParameters(parameters: Tensor[]): ParameterInfo {
    let totalElements = 0;
    let totalMagnitude = 0;
    let maxParam = -Infinity;
    let minParam = Infinity;

    for (const param of parameters) {
      for (let i = 0; i < param.data.length; i++) {
        const val = param.data[i];
        totalElements++;
        totalMagnitude += Math.abs(val);
        maxParam = Math.max(maxParam, val);
        minParam = Math.min(minParam, val);
      }
    }

    return {
      totalElements,
      avgMagnitude: totalMagnitude / totalElements,
      maxParam,
      minParam,
      totalParams: parameters.length,
    };
  }

  /**
   * Print parameter statistics
   */
  static printParameterStats(parameters: Tensor[], label: string = 'Parameters'): void {
    const info = this.checkParameters(parameters);
    
    console.log(`\n[DEBUG] ${label}:`);
    console.log(`  Total tensors: ${info.totalParams}`);
    console.log(`  Total elements: ${info.totalElements}`);
    console.log(`  Avg magnitude: ${info.avgMagnitude.toFixed(6)}`);
    console.log(`  Range: [${info.minParam.toFixed(6)}, ${info.maxParam.toFixed(6)}]`);
  }

  /**
   * Check if tensor requires gradients and has backward function
   */
  static checkTensorGrad(tensor: Tensor, name: string = 'Tensor'): TensorGradInfo {
    return {
      name,
      requiresGrad: tensor.requiresGrad,
      hasGrad: tensor.grad !== undefined,
      hasBackward: tensor._backward !== undefined,
      prevCount: tensor._prev.size,
      shape: tensor.shape,
    };
  }

  /**
   * Print tensor gradient info
   */
  static printTensorGradInfo(tensor: Tensor, name: string = 'Tensor'): void {
    const info = this.checkTensorGrad(tensor, name);
    
    console.log(`\n[DEBUG] ${info.name}:`);
    console.log(`  Shape: [${info.shape.join(', ')}]`);
    console.log(`  requiresGrad: ${info.requiresGrad}`);
    console.log(`  hasGrad: ${info.hasGrad}`);
    console.log(`  hasBackward: ${info.hasBackward}`);
    console.log(`  Previous tensors: ${info.prevCount}`);
  }

  /**
   * Trace computational graph backwards
   */
  static traceGraph(tensor: Tensor): GraphNode[] {
    const visited = new Set<Tensor>();
    const nodes: GraphNode[] = [];

    const trace = (t: Tensor, depth: number = 0) => {
      if (visited.has(t)) return;
      visited.add(t);

      nodes.push({
        depth,
        shape: t.shape,
        requiresGrad: t.requiresGrad,
        hasGrad: t.grad !== undefined,
        hasBackward: t._backward !== undefined,
        prevCount: t._prev.size,
      });

      t._prev.forEach(prev => trace(prev, depth + 1));
    };

    trace(tensor);
    return nodes;
  }

  /**
   * Print computational graph
   */
  static printGraph(tensor: Tensor): void {
    const nodes = this.traceGraph(tensor);
    
    console.log(`\n[DEBUG] Computational Graph:`);
    console.log(`  Total nodes: ${nodes.length}`);
    console.log(`  Max depth: ${Math.max(...nodes.map(n => n.depth))}`);
    
    const byDepth = nodes.reduce((acc, node) => {
      if (!acc[node.depth]) acc[node.depth] = 0;
      acc[node.depth]++;
      return acc;
    }, {} as Record<number, number>);
    
    console.log(`  Nodes by depth:`, byDepth);
  }

  /**
   * Compare two tensors
   */
  static compareTensors(t1: Tensor, t2: Tensor, name1: string = 'T1', name2: string = 'T2'): void {
    if (t1.shape.length !== t2.shape.length || !t1.shape.every((dim, i) => dim === t2.shape[i])) {
      console.log(`\n[DEBUG] ${name1} vs ${name2}: Shape mismatch!`);
      console.log(`  ${name1}: [${t1.shape.join(', ')}]`);
      console.log(`  ${name2}: [${t2.shape.join(', ')}]`);
      return;
    }

    let totalDiff = 0;
    let maxDiff = 0;
    
    for (let i = 0; i < t1.data.length; i++) {
      const diff = Math.abs(t1.data[i] - t2.data[i]);
      totalDiff += diff;
      maxDiff = Math.max(maxDiff, diff);
    }

    const avgDiff = totalDiff / t1.data.length;

    console.log(`\n[DEBUG] ${name1} vs ${name2}:`);
    console.log(`  Avg difference: ${avgDiff.toExponential(4)}`);
    console.log(`  Max difference: ${maxDiff.toExponential(4)}`);
  }
}

export interface GradientInfo {
  totalParams: number;
  paramsWithGrad: number;
  avgGradMagnitude: number;
  maxGrad: number;
  minGrad: number;
  hasGradients: boolean;
}

export interface ParameterInfo {
  totalElements: number;
  avgMagnitude: number;
  maxParam: number;
  minParam: number;
  totalParams: number;
}

export interface TensorGradInfo {
  name: string;
  requiresGrad: boolean;
  hasGrad: boolean;
  hasBackward: boolean;
  prevCount: number;
  shape: number[];
}

export interface GraphNode {
  depth: number;
  shape: number[];
  requiresGrad: boolean;
  hasGrad: boolean;
  hasBackward: boolean;
  prevCount: number;
}
