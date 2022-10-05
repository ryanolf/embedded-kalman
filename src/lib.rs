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
embedded-kalman = "0.1.0"
```

## Implemented Filters
* [Kalman Filter](KalmanFilter)

*/


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

#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![no_std]

use nalgebra::{RealField, SMatrix, SVector};

#[derive(Debug)]
struct Constants<T, const Nx: usize, const Nz: usize, const Nu: usize> {
    // State transition matrices
    F: SMatrix<T, Nx, Nx>,
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

/// A marker trait to indicate what state the Kalman filter is currently in
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
/// Replace these matrices with something more interesting. Note that the notation and names used for the Kalman Filter
/// is similar to the one in the [Wikipedia artice about the Kalman Filter](https://en.wikipedia.org/wiki/Kalman_filter) if you're in doubt which matrix is which
///
/// ```
/// use embedded_kalman::KalmanFilter;
/// use nalgebra::{SMatrix, SVector};
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
/// **embedded-kalman** can infer the dimensions of the matrices in the parameters in most cases.
/// If the dimensions of the matrices do not match, the Kalman Filter won't compile
///
/// Two main methods are used to iterate the KalmanFilter: [KalmanFilter::update] and [KalmanFilter::predict].
///
/// ```
/// use embedded_kalman::KalmanFilter;
/// use nalgebra::Vector2;
/// let kf = kf
///     .predict(Vector2::new(1.0, 0.0)) // Give an input to the system
///     .update(Vector2::new(1.0, 0.0)) // Update the state from a given measurement
///     .expect("Innovation matrix cannot be inverted");
/// ```
/// If the Kalman Filter is in the `Update` state, only the `predict` method is available.
/// If the filter is in the `Predict` state, only the `update` method is available, preserving
/// the 2-step recurrent nature of the Kalman Filter.
///
/// The update method is failable and might return [Option::None] if the innovation matrix is
/// non-invertible. This happens due to rounding errors when the process noise covariance `Q` is small,
/// and is rare.
/// The prior state and state covariance can be fetched after the `predict` stage
/// The posterior state and state covariance can be fetched after the `update` stage
///
/// ```
/// use embedded_kalman::KalmanFilter;
/// use nalgebra::Vector2;
///
/// let kf = kf.predict(Vector2::new(1.0, 0.0)); // Give an input to the system
/// let (x_prior, P_prior) = kf.get_priors(); // Get the priors
/// let kf = kf
///     .update(Vector2::new(1.0, 0.0)) // Update the state from a given measurement
///     .expect("Innovation matrix cannot be inverted");
/// let (x_posterior, P_posterior) = kf.get_posteriors(); // Get the posteriors
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

    /// Gets the posterior state `x` and posterior state covariance `P` as a tuple.
    /// This method is only available when the Kalman Filter is in the `Update` state
    pub fn get_posteriors(&self) -> (&SVector<T, Nx>, &SMatrix<T, Nx, Nx>) {
        (&self.state.x_posterior, &self.state.P_posterior)
    }

    /// Use this method to perform the predict step of a Kalman Filter iteration,
    /// given an input vector `u`
    /// This method is only available when the Kalman Filter is in the `Update` state
    pub fn predict(self, u: SVector<T, Nu>) -> KalmanFilter<T, Nx, Nz, Nu, Predict<T, Nx, Nz>> {
        let Constants { F, B, Q, .. } = &self.constants;
        let Update {
            x_posterior,
            P_posterior,
            ..
        } = &self.state;
        let x_prior = F * x_posterior + B * u;
        let P_prior = F * P_posterior * F.transpose() + Q;

        KalmanFilter {
            constants: self.constants,
            state: Predict { P_prior, x_prior },
        }
    }

    /// Creates a new Kalman Filter. Look at the top-level documentation for usage and examples
    pub fn new(
        F: SMatrix<T, Nx, Nx>,
        B: SMatrix<T, Nx, Nu>,
        H: SMatrix<T, Nz, Nx>,
        Q: SMatrix<T, Nx, Nx>,
        R: SMatrix<T, Nz, Nz>,
        x0: SVector<T, Nx>,
        P0: SMatrix<T, Nx, Nx>,
    ) -> Self {
        Self {
            constants: Constants { F, B, H, Q, R },
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
    /// Gets the prior state `x_prior` and prior state covariance `P_prior` as a tuple.
    /// This method is only available when the Kalman Filter is in the `Predict` state
    pub fn get_priors(&self) -> (&SVector<T, Nx>, &SMatrix<T, Nx, Nx>) {
        (&self.state.x_prior, &self.state.P_prior)
    }

    /// Use this method to perform the update step of a Kalman Filter iteration,
    /// given a measurement vector `z`. This method returns [None] when the innovation matrix
    /// can't be inverted.
    /// This method is only available when the Kalman Filter is in the `Update` state
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
    use nalgebra::Vector2;

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

        let (_x, _P) = kf
            .predict(Vector2::new(1.0, 0.0))
            .update(Vector2::new(1.0, 0.0))
            .expect("Innovation matrix cannot be inverted")
            .get_posteriors();
    }
}
