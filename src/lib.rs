#![cfg_attr(not(test), no_std)]
pub mod fft;

#[cfg(feature = "alloc")]
#[macro_use]
extern crate alloc;