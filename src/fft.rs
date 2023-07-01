use core::marker::PhantomData;

#[allow(unused_imports)]
use micromath::F32Ext;

#[allow(clippy::approx_constant)]
#[allow(clippy::excessive_precision)]
pub const PI: f32 = 3.141592653589793f32;

pub mod array;

#[cfg(feature = "alloc")]
pub mod arc_array;

pub struct Cooley;

struct FftEngine<Implementation> {
    marker: PhantomData<Implementation>,
}
