//! Wolololol

/*!
# embedded-kalman

**embedded-kalman** is a Kalman filtering library written for Rust targeting:

* Embedded or resource-constrained systems
* High-reliability-systems where runtime failure is not an option

For now, **embedded-kalman** only features a standard optimal gain Kalman filter for
linear time-invariant systems. Future versions are planned to contain:

* Different variations such as the time-variant version of the regular Kalman Filter
* Kalman filter varieties for non-linear systems such as the Extended Kalman Filter (EKF) and the Unscented Kalman Filter (UKF)

## Using **embedded-kalman**
You will need the last stable build of the [rust compiler](https://www.rust-lang.org)
and the official package manager: [cargo](https://github.com/rust-lang/cargo).

Simply add the following to your `Cargo.toml` file:

```ignore
[dependencies]
embedded-kalman = "0.1.0"
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

## Implemented Filters
* [Kalman Filter](KalmanFilter)

*/


/*
#![deny(
    missing_docs,
    missing_debug_implementations,
    missing_copy_implementations,
    trivial_casts,
    trivial_numeric_casts,
    unsafe_code,
    unstable_features,
    unused_import_braces,
    unused_qualifications
)]
*/

#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use nalgebra::{RealField, SMatrix, SVector};


#[derive(Debug)]
struct Constants<T, const Nx: usize, const Nz: usize, const Nu: usize> {
    // State transition matrices
    A: SMatrix<T, Nx, Nx>,
    B: SMatrix<T, Nx, Nu>,

    // Measurement matrix
    H: SMatrix<T, Nz, Nx>,

    // State transition noise matrix
    Q: SMatrix<T, Nx, Nx>,

    // Measurement noise matrix
    R: SMatrix<T, Nz, Nz>,
}

#[derive(Debug)]
struct Predict<T, const Nx: usize, const Nz: usize>
    where
        T: RealField,
{
    // Prior state covariance matrix
    P_prior: SMatrix<T, Nx, Nx>,

    // State Vector
    x_prior: SVector<T, Nx>,
}

#[derive(Debug)]
struct Update<T, const Nx: usize, const Nz: usize>
    where
        T: RealField,
{
    // Prior state covariance matrix
    P_posterior: SMatrix<T, Nx, Nx>,

    // State Vector
    x_posterior: SVector<T, Nx>,

    // Kalman gain
    K: SMatrix<T, Nx, Nz>,
}

pub trait State {}
impl<T, const Nx: usize, const Nz: usize> State for Predict<T, Nx, Nz> where T: RealField {}
impl<T, const Nx: usize, const Nz: usize> State for Update<T, Nx, Nz> where T: RealField {}


/// # Kalman Filter
/// A standard linear time-invariant Kalman Filter.
///
/// ## Initialization
/// A new Kalman Filter is created using the [KalmanFilter::new()] function.
/// The Kalman Filter is generic around 5 parameters
/// * `T`: The underlying numeric type used in the calculations of the filter. Is usually `f32` or `f64` depending on your system
/// * `Nx`: The size of your state vector `x`
/// * `Nz`: The size of your measurement vector `z`
/// * `Nu`: The size of your input vector `u`
/// * `S`: The state of the Kalman Filter. This is either `Predict` or `Update`. The filter is always created in the `Update` state
///
/// To create an instance of the KalmanFilter you will need:
/// * A state transition Matrix `A` with dimensions `Nx`*`Nx`
/// * A state input Matrix `B` with dimensions `Nx`*`Nu`
/// * A state transition noise covariance Matrix `Q` with dimensions `Nx`*`Nx`
/// * A measurement noise covariance Matrix `R` with dimensions `Nz`*`Nz`
/// * A measurement Matrix `H` with dimensions `Nz`*`Nx`
/// * An initial state `x0` with dimensions `Nx`
/// * A initial state covariance Matrix `P0` with dimensions `Nx`*`Nx`
///
/// ## Usage
/// Let's make a (terribly uninteresting) instance of the Kalman Filter. We use mostly identity matrices here, for the sake of exemplification.
/// Replace these matrices with something more interesting
///
/// ```
/// use embedded_kalman::KalmanFilter;
/// let kf: KalmanFilter<f64, 2, 2, 2, _> = KalmanFilter::new(
///     SMatrix::identity(), // A
///     SMatrix::identity(), // B
///     SMatrix::identity(), // H
///     SMatrix::identity(), // Q
///     SMatrix::identity(), // R
///     SVector::zeros(),    // x0
///     SMatrix::identity(), // P0
/// );
/// ```


#[derive(Debug)]
pub struct KalmanFilter<T, const Nx: usize, const Nz: usize, const Nu: usize, S>
    where
        T: RealField,
        S: State,
{
    constants: Constants<T, Nx, Nz, Nu>,
    state: S,
}

impl<T, const Nx: usize, const Nz: usize, const Nu: usize>
KalmanFilter<T, Nx, Nz, Nu, Update<T, Nx, Nx>>
    where
        T: RealField,
{
    pub fn get_posteriors(&self) -> (&SVector<T, Nx>, &SMatrix<T, Nx, Nx>) {
        (&self.state.x_posterior, &self.state.P_posterior)
    }

    pub fn predict(self, u: SVector<T, Nu>) -> KalmanFilter<T, Nx, Nz, Nu, Predict<T, Nx, Nz>> {
        let Constants { A, B, Q, .. } = &self.constants;
        let Update {
            x_posterior,
            P_posterior,
            ..
        } = &self.state;
        let x_prior = A * x_posterior + B * u;
        let P_prior = A * P_posterior * A.transpose() + Q;

        KalmanFilter {
            constants: self.constants,
            state: Predict { P_prior, x_prior },
        }
    }

    pub fn new(
        A: SMatrix<T, Nx, Nx>,
        B: SMatrix<T, Nx, Nu>,
        H: SMatrix<T, Nz, Nx>,
        Q: SMatrix<T, Nx, Nx>,
        R: SMatrix<T, Nz, Nz>,
        x0: SVector<T, Nx>,
        P0: SMatrix<T, Nx, Nx>,
    ) -> Self {
        Self {
            constants: Constants { A, B, H, Q, R },
            state: Update {
                x_posterior: x0,
                P_posterior: P0,
                K: SMatrix::identity(),
            },
        }
    }
}

impl<T, const Nx: usize, const Nz: usize, const Nu: usize>
KalmanFilter<T, Nx, Nz, Nu, Predict<T, Nx, Nx>>
    where
        T: RealField,
{
    pub fn get_priors(&self) -> (&SVector<T, Nx>, &SMatrix<T, Nx, Nx>) {
        (&self.state.x_prior, &self.state.P_prior)
    }

    pub fn update(self, z: SVector<T, Nz>) -> Option<KalmanFilter<T, Nx, Nz, Nu, Update<T, Nx, Nz>>> {
        let Constants { H, R, .. } = &self.constants;
        let Predict {
            x_prior, P_prior, ..
        } = &self.state;

        // Innovation residual
        let y = z - H * x_prior;

        // Innovation covariance
        let S = H * P_prior * H.transpose() + R;

        // Kalman Gain
        let K = P_prior * H.transpose() * S.try_inverse()?;

        let x_posterior = x_prior + &K * y;
        let P_posterior = (SMatrix::identity() - &K * H) * P_prior;

        Some(KalmanFilter {
            constants: self.constants,
            state: Update {
                P_posterior,
                x_posterior,
                K,
            },
        })
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix2, Vector2};

    #[test]
    fn it_works() {
        let kf: KalmanFilter<f64, 2, 2, 2, _> = KalmanFilter::new(
            SMatrix::identity(),
            SMatrix::identity(),
            SMatrix::identity(),
            SMatrix::identity(),
            SMatrix::identity(),
            SVector::zeros(),
            SMatrix::identity(),
        );

        let (x, P) = kf
            .predict(Vector2::new(1.0, 0.0))
            .update(Vector2::new(1.0, 0.0))
            .expect("Innovation matrix cannot be inverted")
            .get_posteriors();
    }
}
