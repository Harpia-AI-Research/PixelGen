/**
 * Tensor - N-dimensional array with automatic differentiation support
 * Uses Float32Array for performance
 */
export class Tensor {
  public data: Float32Array;
  public shape: number[];
  public grad?: Tensor;
  public requiresGrad: boolean;
  
  // For autograd
  public _backward?: () => void;
  public _prev: Set<Tensor>;

  constructor(
    data: number[] | Float32Array | number[][],
    shape?: number[],
    requiresGrad: boolean = false
  ) {
    // Flatten nested arrays if provided
    if (Array.isArray(data)) {
      const flattened = this.flattenArray(data);
      this.data = new Float32Array(flattened);
      this.shape = shape || this.inferShape(data);
    } else {
      this.data = data;
      this.shape = shape || [data.length];
    }

    this.requiresGrad = requiresGrad;
    this._prev = new Set();

    // Verify shape matches data length
    const expectedSize = this.shape.reduce((a, b) => a * b, 1);
    if (this.data.length !== expectedSize) {
      throw new Error(
        `Shape ${this.shape} does not match data length ${this.data.length}`
      );
    }
  }

  private flattenArray(arr: any[]): number[] {
    const result: number[] = [];
    const flatten = (a: any) => {
      if (Array.isArray(a)) {
        a.forEach(flatten);
      } else {
        result.push(a);
      }
    };
    flatten(arr);
    return result;
  }

  private inferShape(arr: any[]): number[] {
    const shape: number[] = [];
    let current = arr;
    while (Array.isArray(current)) {
      shape.push(current.length);
      current = current[0];
    }
    return shape;
  }

  /**
   * Get element at specific indices
   */
  get(...indices: number[]): number {
    if (indices.length !== this.shape.length) {
      throw new Error(`Expected ${this.shape.length} indices, got ${indices.length}`);
    }
    const index = this.indicesToFlatIndex(indices);
    return this.data[index];
  }

  /**
   * Set element at specific indices
   */
  set(value: number, ...indices: number[]): void {
    if (indices.length !== this.shape.length) {
      throw new Error(`Expected ${this.shape.length} indices, got ${indices.length}`);
    }
    const index = this.indicesToFlatIndex(indices);
    this.data[index] = value;
  }

  private indicesToFlatIndex(indices: number[]): number {
    let index = 0;
    let stride = 1;
    for (let i = this.shape.length - 1; i >= 0; i--) {
      index += indices[i] * stride;
      stride *= this.shape[i];
    }
    return index;
  }

  /**
   * Reshape tensor to new shape
   */
  reshape(newShape: number[]): Tensor {
    const newSize = newShape.reduce((a, b) => a * b, 1);
    if (newSize !== this.data.length) {
      throw new Error(
        `Cannot reshape tensor of size ${this.data.length} to shape ${newShape}`
      );
    }
    return new Tensor(this.data, newShape, this.requiresGrad);
  }

  /**
   * Transpose 2D tensor
   */
  transpose(): Tensor {
    if (this.shape.length !== 2) {
      throw new Error('Transpose only supports 2D tensors');
    }
    const [rows, cols] = this.shape;
    const result = new Float32Array(this.data.length);
    
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        result[j * rows + i] = this.data[i * cols + j];
      }
    }
    
    return new Tensor(result, [cols, rows], this.requiresGrad);
  }

  /**
   * Element-wise addition
   */
  add(other: Tensor | number): Tensor {
    const result = new Tensor(
      new Float32Array(this.data.length),
      [...this.shape],
      this.requiresGrad
    );

    if (typeof other === 'number') {
      for (let i = 0; i < this.data.length; i++) {
        result.data[i] = this.data[i] + other;
      }
    } else {
      if (!this.shapesMatch(this.shape, other.shape)) {
        throw new Error(`Shape mismatch: ${this.shape} vs ${other.shape}`);
      }
      for (let i = 0; i < this.data.length; i++) {
        result.data[i] = this.data[i] + other.data[i];
      }
    }

    if (this.requiresGrad) {
      result._prev.add(this);
      if (other instanceof Tensor) result._prev.add(other);
      
      result._backward = () => {
        if (!result.grad) return;
        if (!this.grad) this.grad = Tensor.zerosLike(this);
        
        for (let i = 0; i < this.data.length; i++) {
          this.grad.data[i] += result.grad.data[i];
        }
        
        if (other instanceof Tensor && other.requiresGrad) {
          if (!other.grad) other.grad = Tensor.zerosLike(other);
          for (let i = 0; i < other.data.length; i++) {
            other.grad.data[i] += result.grad.data[i];
          }
        }
      };
    }

    return result;
  }

  /**
   * Element-wise multiplication
   */
  multiply(other: Tensor | number): Tensor {
    const result = new Tensor(
      new Float32Array(this.data.length),
      [...this.shape],
      this.requiresGrad
    );

    if (typeof other === 'number') {
      for (let i = 0; i < this.data.length; i++) {
        result.data[i] = this.data[i] * other;
      }
    } else {
      if (!this.shapesMatch(this.shape, other.shape)) {
        throw new Error(`Shape mismatch: ${this.shape} vs ${other.shape}`);
      }
      for (let i = 0; i < this.data.length; i++) {
        result.data[i] = this.data[i] * other.data[i];
      }
    }

    if (this.requiresGrad) {
      result._prev.add(this);
      if (other instanceof Tensor) result._prev.add(other);
      
      result._backward = () => {
        if (!result.grad) return;
        if (!this.grad) this.grad = Tensor.zerosLike(this);
        
        if (typeof other === 'number') {
          for (let i = 0; i < this.data.length; i++) {
            this.grad.data[i] += result.grad.data[i] * other;
          }
        } else {
          for (let i = 0; i < this.data.length; i++) {
            this.grad.data[i] += result.grad.data[i] * other.data[i];
          }
          
          if (other.requiresGrad) {
            if (!other.grad) other.grad = Tensor.zerosLike(other);
            for (let i = 0; i < other.data.length; i++) {
              other.grad.data[i] += result.grad.data[i] * this.data[i];
            }
          }
        }
      };
    }

    return result;
  }

  /**
   * Matrix multiplication (2D tensors only for now)
   */
  matmul(other: Tensor): Tensor {
    if (this.shape.length !== 2 || other.shape.length !== 2) {
      throw new Error('Matmul requires 2D tensors');
    }
    
    const [m, n] = this.shape;
    const [n2, p] = other.shape;
    
    if (n !== n2) {
      throw new Error(`Cannot multiply matrices of shapes ${this.shape} and ${other.shape}`);
    }
    
    const result = new Float32Array(m * p);
    
    // Simple matrix multiplication (can be optimized later)
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < p; j++) {
        let sum = 0;
        for (let k = 0; k < n; k++) {
          sum += this.data[i * n + k] * other.data[k * p + j];
        }
        result[i * p + j] = sum;
      }
    }
    
    return new Tensor(result, [m, p], this.requiresGrad || other.requiresGrad);
  }

  /**
   * Backward pass for autograd
   */
  backward(): void {
    if (!this.requiresGrad) {
      throw new Error('This tensor does not require gradients');
    }

    // Initialize gradient if not set
    if (!this.grad) {
      this.grad = Tensor.onesLike(this);
    }

    // Topological sort
    const topo: Tensor[] = [];
    const visited = new Set<Tensor>();
    
    const buildTopo = (tensor: Tensor) => {
      if (!visited.has(tensor)) {
        visited.add(tensor);
        tensor._prev.forEach(buildTopo);
        topo.push(tensor);
      }
    };
    
    buildTopo(this);

    // Backward pass
    topo.reverse();
    topo.forEach(tensor => {
      if (tensor._backward) {
        tensor._backward();
      }
    });
  }

  /**
   * Zero gradients
   */
  zeroGrad(): void {
    if (this.grad) {
      this.grad.data.fill(0);
    }
  }

  private shapesMatch(shape1: number[], shape2: number[]): boolean {
    if (shape1.length !== shape2.length) return false;
    return shape1.every((dim, i) => dim === shape2[i]);
  }

  // Static factory methods
  static zeros(shape: number[], requiresGrad: boolean = false): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    return new Tensor(new Float32Array(size), shape, requiresGrad);
  }

  static ones(shape: number[], requiresGrad: boolean = false): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(size);
    data.fill(1);
    return new Tensor(data, shape, requiresGrad);
  }

  static zerosLike(tensor: Tensor): Tensor {
    return Tensor.zeros(tensor.shape, tensor.requiresGrad);
  }

  static onesLike(tensor: Tensor): Tensor {
    return Tensor.ones(tensor.shape, tensor.requiresGrad);
  }

  static random(shape: number[], requiresGrad: boolean = false): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      data[i] = Math.random() * 2 - 1; // Range [-1, 1]
    }
    return new Tensor(data, shape, requiresGrad);
  }

  /**
   * Xavier/Glorot initialization
   */
  static xavier(shape: number[], requiresGrad: boolean = false): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    const fanIn = shape[0];
    const fanOut = shape.length > 1 ? shape[1] : 1;
    const limit = Math.sqrt(6 / (fanIn + fanOut));
    
    const data = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      data[i] = (Math.random() * 2 - 1) * limit;
    }
    return new Tensor(data, shape, requiresGrad);
  }

  /**
   * Clone tensor
   */
  clone(): Tensor {
    return new Tensor(
      new Float32Array(this.data),
      [...this.shape],
      this.requiresGrad
    );
  }

  /**
   * Convert to regular array (for debugging)
   */
  toArray(): any {
    if (this.shape.length === 1) {
      return Array.from(this.data);
    }
    
    if (this.shape.length === 2) {
      const [rows, cols] = this.shape;
      const result = [];
      for (let i = 0; i < rows; i++) {
        const row = [];
        for (let j = 0; j < cols; j++) {
          row.push(this.data[i * cols + j]);
        }
        result.push(row);
      }
      return result;
    }
    
    // For higher dimensions, just return flattened array
    return Array.from(this.data);
  }
}
