mod tests;

use nalgebra::{Matrix2, RealField, SMatrix, SVector, Vector2};
use crate::covariance_matrix::CovarianceMatrix;

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
/// use nalgebra::{SMatrix, SVector, Matrix2};
/// let kf: KalmanFilter<f64, 2, 2, 2, _> = KalmanFilter::new(
///     SMatrix::identity(), // A
///     SMatrix::identity(), // B
///     SMatrix::identity(), // H
///     Matrix2::identity().try_into().expect("Q is not a valid covariance Matrix"),
///     Matrix2::identity().try_into().expect("R is not a valid covariance Matrix"),
///     SVector::zeros(),
///     Matrix2::identity().try_into().expect("P0 is not a valid covariance Matrix"),
/// );
/// ```
/// **embedded-kalman** can infer the dimensions of the matrices in the parameters in most cases.
/// If the dimensions of the matrices do not match, the Kalman Filter won't compile. Note the use of the
/// `try_into` method. `Q`, `R` and `P0` are required to be valid covariance matrices.
/// **embedded-kalman** exposes its own [`CovarianceMatrix`] type, which implements [`TryFrom`] for
/// any square matrix.
///
/// Two main methods are used to iterate the KalmanFilter: [update](KalmanFilter::update) and [predict](KalmanFilter::predict).
///
/// ```
/// use embedded_kalman::KalmanFilter;
/// use nalgebra::Vector2;
/// let kf = kf
///     .predict(Vector2::new(1.0, 0.0)) // Give an input to the system
///     .update(Vector2::new(1.0, 0.0)) // Update the state from a given measurement
///     .expect("Innovation matrix cannot be inverted");
/// ```
/// If the Kalman Filter is in the `Update` state, only the [predict](KalmanFilter::predict) method is available.
/// If the filter is in the `Predict` state, only the [update](KalmanFilter::update) method is available, preserving
/// the 2-step recurrent nature of the Kalman Filter.
///
/// The update method is failable and might return [None](Option::None) if the innovation matrix is
/// non-invertible. This happens due to rounding errors when the process noise covariance `Q` is small,
/// and happens very rarely.
/// The prior state and state covariance can be fetched after the `Predict` stage.
/// The posterior state and state covariance can be fetched after the `Update` stage.
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

/*
#[derive(Debug)]
pub struct KalmanFilter
<
    T,                  // numerical type use for computation. Typically f32 or f64
    const Nx: usize,    // Size of the state vector x
    const Nz: usize,    // Size of the measurement vector z
    const Nu: usize,    // Size of the input vector u
    S                   // The state of the Kalman Filter
>
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
        Q: CovarianceMatrix<T, Nx>,
        R: CovarianceMatrix<T, Nz>,
        x0: SVector<T, Nx>,
        P0: CovarianceMatrix<T, Nx>,
    ) -> Self {
        Self {
            constants: Constants { F, B, H, Q: Q.0, R: R.0 },
            state: Update {
                x_posterior: x0,
                P_posterior: P0.0,
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
            },
        })
    }
}


*/


#[derive(Debug)]
struct Constants<T, const Nx: usize, const Nz: usize, const Nu: usize, Fs, Fj, Hs, Hj>
    where
    T: RealField,
    Fs: Fn(SVector<T, Nx>, SVector<T, Nu>) -> SVector<T, Nx>,
    Fj: Fn(SVector<T, Nx>, SVector<T, Nu>) -> SMatrix<T, Nx, Nx>,
    Hs: Fn(SVector<T, Nx>) -> SVector<T, Nz>,
    Hj: Fn(SVector<T, Nx>) -> SMatrix<T, Nz, Nz>,
{

    // State transfer function
    f: Fs,
    // Measurement function
    h: Hs,

    // Jacobians
    F: Fj,
    H: Hj,

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
}

/// A marker trait to indicate what state the Kalman filter is currently in
pub trait State {}
impl<T, const Nx: usize, const Nz: usize> State for Predict<T, Nx, Nz> where T: RealField {}
impl<T, const Nx: usize, const Nz: usize> State for Update<T, Nx, Nz> where T: RealField {}

fn wolol(){
    let v = Vector2::new(1.0, 1.0);
    let a: Constants<f64, 2, 2, 2, _, _, _, _> = Constants{
        f: |x, u| x + v,
        h: |x| x,
        F: |x, u| Matrix2::new(1.0, 1.0, 1.0, 1.0),
        H: |x| Matrix2::new(1.0, 1.0, 1.0, 1.0),
        Q: Matrix2::new(1.0, 0.0, 0.0, 1.0),
        R: Matrix2::new(1.0, 0.0, 0.0, 1.0),
    };
}
