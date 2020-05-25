use super::*;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::time::{Duration, Instant};

use crate::my_consts::*;
#[inline(always)]
fn idx_to_coords(idx: usize) -> (usize, usize) {
    (idx % WIDTH, idx / WIDTH)
}

#[inline(always)]
pub fn coords_to_idx(xidx: usize, yidx: usize) -> usize {
    xidx + yidx * WIDTH
}

#[inline(always)]
pub fn sum_p(buffer: &[MyComplex]) -> f32 {
    buffer.iter().map(|n| n.mag_2()).sum()
}

#[inline(always)]
pub fn max_p(buffer: &[MyComplex]) -> f32 {
    buffer
        .iter()
        .map(|n| n.mag_2())
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0.0)
}

#[inline(always)]
fn normalize(buffer: &mut [MyComplex]) {
    let p2sum: f32 = sum_p(buffer);
    let coeff = 1.0 / p2sum.sqrt();
    buffer.iter_mut().for_each(|n| *n = *n * coeff);
}

#[inline(always)]
fn on_border(xidx: usize, yidx: usize) -> bool {
    xidx == 0 || yidx == 0 || xidx >= WIDTH - 1 || yidx >= HEIGHT - 1
}

#[inline(always)]
fn in_slit(x: usize, y: usize) -> bool {
    let x = x as isize;
    let y = y as isize;
    y == SLIT_HEIGHT
        && ((x >= SLIT_1_START && x <= SLIT_1_END) || (x >= SLIT_2_START && x <= SLIT_2_END))
}

fn write_frame(frame: &[MyComplex]) -> Vec<u8> {
    let mut retvl = Vec::with_capacity(frame.len() * 8);
    for cur in frame {
        retvl.extend_from_slice(&cur.to_bytes());
    }
    retvl
}
#[inline(always)]
pub fn run() {
    let mut allocation = vec![MyComplex::zero(); TOTAL_BUFFER_SIZE];
    let mut frames_iter = allocation.chunks_exact_mut(FRAME_SIZE);
    let initial_frame = frames_iter.next().unwrap();
    initialize(initial_frame);
    normalize(initial_frame);

    let start = Instant::now();
    let mut current = &*initial_frame;
    println!("RMP: {}", max_p(current));
    for _ in 1..NUM_FRAMES {
        let next = frames_iter.next().unwrap();
        cpu_kernel(current, next);
        current = next;
    }
    let end = Instant::now();
    println!(
        "{} / {}",
        end.duration_since(start).as_millis(),
        (end.duration_since(start).as_millis() as f32) / (NUM_FRAMES as f32)
    );
    let mut output_file = OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open("cpu_out.data")
        .unwrap();
    let frames_iter = allocation.chunks_exact(FRAME_SIZE);
    for current_frame in frames_iter {
        let bbuffer = write_frame(current_frame);
        output_file.write_all(&bbuffer).unwrap();
    }
}

const cur_coeff : MyComplex = MyComplex::new(1.0, -2.0 * HBAR * DT/(M * DX * DX));
const neighbor_coeff : MyComplex = MyComplex::im( (HBAR * DT) / (2.0 * M * DX * DX));

#[inline(always)]
fn cpu_kernel(input: &[MyComplex], output: &mut [MyComplex]) {
    for y_idx in 0..HEIGHT {
        for x_idx in 0..WIDTH {
            let cur_idx = coords_to_idx(x_idx, y_idx);
            if on_border(x_idx, y_idx) || in_slit(x_idx, y_idx) {
                output[cur_idx] = MyComplex::zero();
                continue;
            }
            let cur_value = input[cur_idx];

            let left_idx = coords_to_idx(x_idx - 1, y_idx);
            let left_value = input[left_idx];

            let right_idx = coords_to_idx(x_idx + 1, y_idx);
            let right_value = input[right_idx];

            let top_idx = coords_to_idx(x_idx, y_idx + 1);
            let top_value = input[top_idx];

            let bottom_idx = coords_to_idx(x_idx, y_idx - 1);
            let bottom_value = input[bottom_idx];

            let neighbor_sum = left_value + right_value + top_value + bottom_value;
            let retvl = cur_coeff * cur_value + neighbor_coeff * neighbor_sum;
            output[cur_idx] = retvl;
        }
    }
    normalize(output);
}

pub fn initialize(buffer: &mut [MyComplex]) {
    for xidx in 0..WIDTH {
        buffer[coords_to_idx(xidx, 0)] = MyComplex::zero();
        buffer[coords_to_idx(xidx, HEIGHT - 1)] = MyComplex::zero();
    }
    for yidx in 0..HEIGHT {
        buffer[coords_to_idx(0, yidx)] = MyComplex::zero();
        buffer[coords_to_idx(WIDTH - 1, yidx)] = MyComplex::zero();
    }
    for yidx in 1..HEIGHT- 1{
        for xidx in 1..WIDTH -1 {
            let cur_idx = coords_to_idx(xidx, yidx);
            let tv = (cur_idx ) as f32;
            let tp = (((cur_idx as f32) % 100.0) / 50.0) * std::f32::consts::PI;
            let (ti, tr) = tp.sin_cos();
            buffer[coords_to_idx(xidx, yidx)] = MyComplex::new(tr * tv , ti * tv);
        }
    }
    normalize(buffer);
}

#[inline(always)]
fn gaussian(
    center_xidx: usize,
    center_yidx: usize,
    initial_px: f32,
    initial_py: f32,
    sigma: f32,
) -> impl Fn(usize, usize) -> MyComplex {
    #[inline(always)]
    move |xidx, yidx| {
        let y_dist = if yidx > center_yidx {
            yidx - center_yidx
        } else {
            center_yidx - yidx
        };
        let x_dist = if xidx > center_xidx {
            xidx - center_xidx
        } else {
            center_xidx - xidx
        };
        let ridx_squared = x_dist * x_dist + y_dist * y_dist;
        let pos_exp = -1.0 * DX * DX / (4.0 * sigma * sigma) * (ridx_squared as f32);
        let momentum_exp =
            -1.0 * (initial_px * DX * (xidx as f32) + initial_py * DX * (yidx as f32));
        let gen_exp = MyComplex::new(pos_exp, momentum_exp);
        gen_exp.exp()
    }
}
