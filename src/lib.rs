/*!
# embedded-kalman

**embedded-kalman** is a Kalman filtering library written for Rust targeting:

* Embedded or resource-constrained systems
* High-reliability-systems where runtime failure is not an option

Although this library was designed with embedded systems in mind, there's nothing wrong with using
it in other applications. On the contrary, it should be quite fast on unconstrained systems.
For now, **embedded-kalman** only features a standard optimal gain Kalman filter for
linear time-invariant systems. Future versions are planned to contain:

* Different variations such as the time-variant version of the regular Kalman Filter
* Kalman filter varieties for non-linear systems such as the Extended Kalman Filter (EKF) and the Unscented Kalman Filter (UKF)

## A quick example
```ignore
use embedded_kalman::KalmanFilter;
use nalgebra::{SMatrix, SVector, Vector2};

let kf: KalmanFilter<f64, 2, 2, 2, _> = KalmanFilter::new(
    ... // matrices and initial state go here
);

let kf = kf
    .predict(Vector2::new(1.0, 0.0)) // Give an input to the system
    .update(Vector2::new(1.0, 0.0)) // Update the state from a given measurement
    .expect("Update step failed!");

let (x_posterior, P_posterior) = kf.get_posteriors(); // Get the posteriors
```

## Features
* Built on top of the beautiful [nalgebra](https://github.com/dimforge/nalgebra) linear algebra library
* **embedded-kalman** does not rely on the standard library and can be run in a `#![no_std]` environment
* The filters do not require dynamic memory allocation, and can be run without an allocator
* Plays nicely with embedded platforms such as the microcontrollers running on the `ARM Cortex-M` architecture
* Uses Rust's type-state programming to ensure correct usage of the filters in compile-time
* Written in 100% safe Rust
* Few direct dependencies

## Non-features
* Thus far, little effort has been made to optimize the code in this library
* This library is pretty new and still not battle-tested. Please do not use this for anything important

## Using **embedded-kalman**
You will need the last stable build of the [rust compiler](https://www.rust-lang.org)
and the official package manager: [cargo](https://github.com/rust-lang/cargo).

Simply add the following to your `Cargo.toml` file:

```ignore
[dependencies]
embedded-kalman = "0.2.0"
```

## Implemented Filters
* [Kalman Filter](KalmanFilter)

*/


#![deny(
    missing_debug_implementations,
    missing_copy_implementations,
    trivial_casts,
    trivial_numeric_casts,
    unsafe_code,
    unstable_features,
    unused_import_braces,
    unused_qualifications
)]

#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![no_std]

mod covariance_matrix;
mod extended_kalman_filter;
mod kalman_filter;

pub use crate::covariance_matrix::CovarianceMatrix;
pub use crate::kalman_filter::KalmanFilter;
