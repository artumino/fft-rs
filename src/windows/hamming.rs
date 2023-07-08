use core::ops::Mul;

use crate::WindowFunction;

pub struct Hamming;
impl<T> WindowFunction<T> for Hamming
    where T : Copy + Mul<f32, Output = T>{
    type TMul = f32;

    #[inline(always)]
    fn calculate(i: f32, n: f32) -> f32 {
        0.54f32 - 0.46f32 * (2.0f32 * core::f32::consts::PI * i / n).cos()
    }
}