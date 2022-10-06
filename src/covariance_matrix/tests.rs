#[cfg(test)]
use super::*;
use nalgebra::Matrix2;


#[test]
fn valid_covariance_matrix() {
    let cov: Result<CovarianceMatrix<_, 2>, Error> =
        Matrix2::new(1.0, 0.0, 0.0, 1.0)
            .try_into();

    assert!(cov.is_ok())
}

#[test]
fn non_symmetric_covariance_matrix() {
    let cov: Result<CovarianceMatrix<_, 2>, Error> =
        Matrix2::new(1.0, -1.0, 0.0, 1.0)
            .try_into();

    assert_eq!(cov, Err(Error::NotSymmetric))
}

#[test]
fn non_positive_definite_covariance_matrix() {
    let cov: Result<CovarianceMatrix<_, 2>, Error> =
        Matrix2::new(1.0, 1.0, 1.0, 1.0)
            .try_into();

    assert_eq!(cov, Err(Error::NotPositiveDefinite))
}
