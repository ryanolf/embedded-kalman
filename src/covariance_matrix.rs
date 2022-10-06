/*!
A module that defines a covariance matrix type for use in Kalman Filters
*/

use nalgebra::{RealField, SMatrix};

/// A NewType Wrapper for a square matrix, that ensures that the wrapped matrix
/// always is a covariance matrix. A covariance matrix is always positive definite
/// and symmetric. [CovarianceMatrix] implements the [TryFrom] trait for any square matrix.
/// A simple way to make a covariance matrix is:
///
/// ```
/// use nalgebra::{Matrix2};
/// let matrix = Matrix2::new(1.0, 0.0, 0.0, 1.0);
/// let cov: CovarianceMatrix<_, 2> = matrix
///     .try_into()
///     .expect("Matrix is not a covariance matrix");
/// ```
#[derive(Debug, Eq, PartialEq)]
pub struct CovarianceMatrix<T: RealField, const N: usize>(pub SMatrix<T, N, N>);


#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub enum Error{
    NotSymmetric,
    NotPositiveDefinite
}

impl<T: RealField, const N: usize> TryFrom<SMatrix<T, N, N>> for CovarianceMatrix<T, N>{
    type Error = Error;

    fn try_from(value: SMatrix<T, N, N>) -> Result<Self, Self::Error> {

        // Check if matrix is symmetric
        if value != value.transpose() {
            return Err(Error::NotSymmetric);
        }
        // Check if matrix is positive definite
        if value.clone().cholesky().is_none() {
            return Err(Error::NotPositiveDefinite)
        }
        Ok(Self(value))
    }
}


#[cfg(test)]
mod tests {
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
}
