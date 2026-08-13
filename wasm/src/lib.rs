use wasm_bindgen::prelude::*;
use js_sys::{Object, Float32Array, Int32Array, Uint32Array};

unsafe fn float32_array(data: Vec<f32>) -> Float32Array {
    let mut arr = Float32Array::new_with_length(data.len() as u32);
    arr.copy_from(&data);
    arr
}

unsafe fn int32_array(data: Vec<i32>) -> Int32Array {
    let mut arr = Int32Array::new_with_length(data.len() as u32);
    arr.copy_from(&data);
    arr
}

unsafe fn uint32_array(data: Vec<u32>) -> Uint32Array {
    let mut arr = Uint32Array::new_with_length(data.len() as u32);
    arr.copy_from(&data);
    arr
}

fn object_with_keys(keys: &[&str], values: &[JsValue]) -> Object {
    let obj = Object::new();
    for (key, value) in keys.iter().zip(values.iter()) {
        js_sys::Reflect::set(&obj, &JsValue::from_str(key), value).unwrap();
    }
    obj
}

#[wasm_bindgen]
pub fn conv2d_forward(
    input_data: &[f32],
    input_shape: &[usize],
    filters_data: &[f32],
    _filters_shape: &[usize],
    bias_data: &[f32],
    _bias_shape: &[usize],
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> Object {
    let batch = input_shape[0];
    let channels = input_shape[1];
    let height = input_shape[2];
    let width = input_shape[3];

    let out_height = (height + 2 * padding - kernel_size) / stride + 1;
    let out_width = (width + 2 * padding - kernel_size) / stride + 1;

    let padded_height = height + 2 * padding;
    let padded_width = width + 2 * padding;

    let mut padded_input = vec![0.0f32; batch * channels * padded_height * padded_width];

    if padding > 0 {
        for b in 0..batch {
            for c in 0..channels {
                for h in 0..height {
                    for w in 0..width {
                        let input_idx = b * channels * height * width + c * height * width + h * width + w;
                        let padded_idx = b * channels * padded_height * padded_width + c * padded_height * padded_width + (h + padding) * padded_width + (w + padding);
                        padded_input[padded_idx] = input_data[input_idx];
                    }
                }
            }
        }
    } else {
        padded_input = input_data.to_vec();
    }

    let mut output = vec![0.0f32; batch * out_channels * out_height * out_width];

    for b in 0..batch {
        for oc in 0..out_channels {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let mut sum = 0.0f32;

                    for ic in 0..channels {
                        for kh in 0..kernel_size {
                            for kw in 0..kernel_size {
                                let ih = oh * stride + kh;
                                let iw = ow * stride + kw;

                                let input_idx = b * channels * padded_height * padded_width + ic * padded_height * padded_width + ih * padded_width + iw;
                                let filter_idx = oc * channels * kernel_size * kernel_size + ic * kernel_size * kernel_size + kh * kernel_size + kw;

                                sum += padded_input[input_idx] * filters_data[filter_idx];
                            }
                        }
                    }

                    sum += bias_data[oc];

                    let output_idx = b * out_channels * out_height * out_width + oc * out_height * out_width + oh * out_width + ow;
                    output[output_idx] = sum;
                }
            }
        }
    }

    let output_shape = vec![batch as u32, out_channels as u32, out_height as u32, out_width as u32];
    let padded_input_shape = vec![batch as u32, channels as u32, padded_height as u32, padded_width as u32];

    object_with_keys(
        &["output", "paddedInput", "outputShape", "paddedInputShape"],
        &[
            unsafe { float32_array(output) }.into(),
            unsafe { float32_array(padded_input) }.into(),
            unsafe { uint32_array(output_shape) }.into(),
            unsafe { uint32_array(padded_input_shape) }.into(),
        ],
    )
}

#[wasm_bindgen]
pub fn conv2d_backward(
    input_data: &[f32],
    _input_shape: &[usize],
    padded_input_data: &[f32],
    padded_input_shape: &[usize],
    filters_data: &[f32],
    _filters_shape: &[usize],
    output_grad_data: &[f32],
    _output_grad_shape: &[usize],
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    out_height: usize,
    out_width: usize,
) -> Object {
    let padded_height = padded_input_shape[2];
    let padded_width = padded_input_shape[3];

    let mut filter_grad = vec![0.0f32; filters_data.len()];
    let mut bias_grad = vec![0.0f32; out_channels];
    let mut grad_padded = vec![0.0f32; padded_input_data.len()];

    for b in 0..batch {
        for oc in 0..out_channels {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let output_idx = b * out_channels * out_height * out_width + oc * out_height * out_width + oh * out_width + ow;
                    let g = output_grad_data[output_idx];

                    bias_grad[oc] += g;

                    for ic in 0..channels {
                        for kh in 0..kernel_size {
                            for kw in 0..kernel_size {
                                let ih = oh * stride + kh;
                                let iw = ow * stride + kw;

                                let filter_idx = oc * channels * kernel_size * kernel_size + ic * kernel_size * kernel_size + kh * kernel_size + kw;
                                let input_idx = b * channels * padded_height * padded_width + ic * padded_height * padded_width + ih * padded_width + iw;

                                filter_grad[filter_idx] += g * padded_input_data[input_idx];
                                grad_padded[input_idx] += g * filters_data[filter_idx];
                            }
                        }
                    }
                }
            }
        }
    }

    let mut input_grad = vec![0.0f32; input_data.len()];

    if padding > 0 {
        for b in 0..batch {
            for c in 0..channels {
                for h in 0..height {
                    for w in 0..width {
                        let input_idx = b * channels * height * width + c * height * width + h * width + w;
                        let padded_idx = b * channels * padded_height * padded_width + c * padded_height * padded_width + (h + padding) * padded_width + (w + padding);
                        input_grad[input_idx] = grad_padded[padded_idx];
                    }
                }
            }
        }
    } else {
        input_grad.copy_from_slice(&grad_padded);
    }

    let input_grad_shape = vec![batch as u32, channels as u32, height as u32, width as u32];
    let filter_grad_shape = vec![out_channels as u32, channels as u32, kernel_size as u32, kernel_size as u32];
    let bias_grad_shape = vec![out_channels as u32];

    object_with_keys(
        &["inputGrad", "filterGrad", "biasGrad", "inputGradShape", "filterGradShape", "biasGradShape"],
        &[
            unsafe { float32_array(input_grad) }.into(),
            unsafe { float32_array(filter_grad) }.into(),
            unsafe { float32_array(bias_grad) }.into(),
            unsafe { uint32_array(input_grad_shape) }.into(),
            unsafe { uint32_array(filter_grad_shape) }.into(),
            unsafe { uint32_array(bias_grad_shape) }.into(),
        ],
    )
}

#[wasm_bindgen]
pub fn maxpool_forward(
    input_data: &[f32],
    input_shape: &[usize],
    pool_size: usize,
    stride: usize,
) -> Object {
    let batch = input_shape[0];
    let channels = input_shape[1];
    let height = input_shape[2];
    let width = input_shape[3];

    let out_height = (height - pool_size) / stride + 1;
    let out_width = (width - pool_size) / stride + 1;

    let mut output = vec![0.0f32; batch * channels * out_height * out_width];
    let mut argmax = vec![0i32; batch * channels * out_height * out_width];

    for b in 0..batch {
        for c in 0..channels {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let mut max = -f32::INFINITY;
                    let mut max_idx = 0usize;

                    for ph in 0..pool_size {
                        for pw in 0..pool_size {
                            let ih = oh * stride + ph;
                            let iw = ow * stride + pw;

                            let input_idx = b * channels * height * width + c * height * width + ih * width + iw;

                            if input_data[input_idx] > max {
                                max = input_data[input_idx];
                                max_idx = input_idx;
                            }
                        }
                    }

                    let output_idx = b * channels * out_height * out_width + c * out_height * out_width + oh * out_width + ow;
                    output[output_idx] = max;
                    argmax[output_idx] = max_idx as i32;
                }
            }
        }
    }

    let output_shape = vec![batch as u32, channels as u32, out_height as u32, out_width as u32];

    object_with_keys(
        &["output", "argmax", "outputShape"],
        &[
            unsafe { float32_array(output) }.into(),
            unsafe { int32_array(argmax) }.into(),
            unsafe { uint32_array(output_shape) }.into(),
        ],
    )
}

#[wasm_bindgen]
pub fn avgpool_forward(
    input_data: &[f32],
    input_shape: &[usize],
    pool_size: usize,
    stride: usize,
) -> Object {
    let batch = input_shape[0];
    let channels = input_shape[1];
    let height = input_shape[2];
    let width = input_shape[3];

    let out_height = (height - pool_size) / stride + 1;
    let out_width = (width - pool_size) / stride + 1;

    let pool_area = pool_size * pool_size;
    let mut output = vec![0.0f32; batch * channels * out_height * out_width];

    for b in 0..batch {
        for c in 0..channels {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let mut sum = 0.0f32;

                    for ph in 0..pool_size {
                        for pw in 0..pool_size {
                            let ih = oh * stride + ph;
                            let iw = ow * stride + pw;

                            let input_idx = b * channels * height * width + c * height * width + ih * width + iw;
                            sum += input_data[input_idx];
                        }
                    }

                    let output_idx = b * channels * out_height * out_width + c * out_height * out_width + oh * out_width + ow;
                    output[output_idx] = sum / pool_area as f32;
                }
            }
        }
    }

    let output_shape = vec![batch as u32, channels as u32, out_height as u32, out_width as u32];

    object_with_keys(
        &["output", "argmax", "outputShape"],
        &[
            unsafe { float32_array(output) }.into(),
            unsafe { int32_array(vec![]) }.into(),
            unsafe { uint32_array(output_shape) }.into(),
        ],
    )
}

#[wasm_bindgen]
pub fn upsample_forward(
    input_data: &[f32],
    input_shape: &[usize],
    scale: usize,
) -> Object {
    let batch = input_shape[0];
    let channels = input_shape[1];
    let height = input_shape[2];
    let width = input_shape[3];

    let out_height = height * scale;
    let out_width = width * scale;

    let mut output = vec![0.0f32; batch * channels * out_height * out_width];

    for b in 0..batch {
        for c in 0..channels {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let ih = oh / scale;
                    let iw = ow / scale;

                    let input_idx = b * channels * height * width + c * height * width + ih * width + iw;
                    let output_idx = b * channels * out_height * out_width + c * out_height * out_width + oh * out_width + ow;

                    output[output_idx] = input_data[input_idx];
                }
            }
        }
    }

    let output_shape = vec![batch as u32, channels as u32, out_height as u32, out_width as u32];

    object_with_keys(
        &["output", "outputShape"],
        &[
            unsafe { float32_array(output) }.into(),
            unsafe { uint32_array(output_shape) }.into(),
        ],
    )
}

#[wasm_bindgen]
pub fn matmul(
    a_data: &[f32],
    a_shape: &[usize],
    b_data: &[f32],
    b_shape: &[usize],
) -> Object {
    let m = a_shape[0];
    let n = a_shape[1];
    let p = b_shape[1];

    let mut output = vec![0.0f32; m * p];

    for i in 0..m {
        for j in 0..p {
            let mut sum = 0.0f32;
            for k in 0..n {
                sum += a_data[i * n + k] * b_data[k * p + j];
            }
            output[i * p + j] = sum;
        }
    }

    let output_shape = vec![m as u32, p as u32];

    object_with_keys(
        &["output", "outputShape"],
        &[
            unsafe { float32_array(output) }.into(),
            unsafe { uint32_array(output_shape) }.into(),
        ],
    )
}
