use core::ops::Mul;

use crate::WindowFunction;

pub mod hanning;
pub mod hamming;

pub struct Rect;
impl<T> WindowFunction<T> for Rect
    where T : Copy + Mul<f32, Output = T>{
    type TMul = f32;

    #[inline(always)]
    fn calculate(_: f32, _: f32) -> f32 {
        1.0f32
    }
}