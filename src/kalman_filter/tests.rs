
#[cfg(test)]
use super::*;
use nalgebra::{Matrix2, Vector2};

#[test]
fn kalman_filter_works() {
    let kf: KalmanFilter<f64, 2, 2, 2, _> = KalmanFilter::new(
        SMatrix::identity(),
        SMatrix::identity(),
        SMatrix::identity(),
        Matrix2::identity().try_into().expect("Q is not a valid covariance Matrix"),
        Matrix2::identity().try_into().expect("R is not a valid covariance Matrix"),
        SVector::zeros(),
        Matrix2::identity().try_into().expect("P0 is not a valid covariance Matrix"),
    );

    let (_x, _P) = kf
        .predict(Vector2::new(1.0, 0.0))
        .update(Vector2::new(1.0, 0.0))
        .expect("Innovation matrix cannot be inverted")
        .get_posteriors();
}