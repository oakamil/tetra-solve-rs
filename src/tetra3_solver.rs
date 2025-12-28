// Copyright (c) 2025 Omair Kamil oakamil@gmail.com
// See LICENSE file in root directory for license terms.

use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use canonical_error::{
    deadline_exceeded_error, invalid_argument_error, not_found_error, CanonicalError,
};
use ndarray::Array2;

use cedar_elements::cedar::{ImageCoord, PlateSolution};
use cedar_elements::cedar_common::CelestialCoord;
use cedar_elements::imu_trait::EquatorialCoordinates;
use cedar_elements::solver_trait::{SolveExtension, SolveParams, SolverTrait};

use crate::tetra3::Tetra3;

// Status constants matching Tetra3 implementation
const MATCH_FOUND: u8 = 1;
const NO_MATCH: u8 = 2;
const TIMEOUT: u8 = 3;
const CANCELLED: u8 = 4;
const TOO_FEW: u8 = 5;

pub struct Tetra3Solver {
    inner: tokio::sync::Mutex<Tetra3>,
    // Shared cancellation flag between the trait and the solver loop
    cancelled: Arc<AtomicBool>,
}

impl Tetra3Solver {
    pub fn new(tetra3: Tetra3) -> Self {
        Tetra3Solver {
            inner: tokio::sync::Mutex::new(tetra3),
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }
}

#[async_trait]
impl SolverTrait for Tetra3Solver {
    async fn solve_from_centroids(
        &self,
        star_centroids: &[ImageCoord],
        width: usize,
        height: usize,
        extension: &SolveExtension,
        params: &SolveParams,
        _imu_estimate: Option<EquatorialCoordinates>,
    ) -> Result<PlateSolution, CanonicalError> {
        // Reset cancellation state before starting a new solve
        self.cancelled.store(false, AtomicOrdering::SeqCst);

        let mut inner = self.inner.lock().await;
        
        // Map ImageCoord slice to ndarray Array2 (N x 2)
        // Tetra3 expects [[y, x], ...] as per its internal processing logic
        let mut centroids_arr = Array2::<f64>::zeros((star_centroids.len(), 2));
        for (i, coord) in star_centroids.iter().enumerate() {
            centroids_arr[[i, 0]] = coord.y;
            centroids_arr[[i, 1]] = coord.x;
        }

        // Map SolveParams to Tetra3 arguments
        let fov_estimate = params.fov_estimate.map(|(fov, _)| fov);
        let fov_max_error = params.fov_estimate.map(|(_, err)| err);
        let solve_timeout_ms = params.solve_timeout.map(|d| d.as_millis() as u64);

        // Synchronize trait cancellation flag with the Tetra3 instance
        // (Note: This assumes Tetra3 has been updated to check this flag)
        inner.set_cancelled(false); 

        let result = inner.solve_from_centroids(
            &centroids_arr,
            (height as u32, width as u32),
            fov_estimate,
            fov_max_error,
            params.match_radius,
            params.match_threshold,
            solve_timeout_ms,
            params.distortion,
        );

        match result.status {
            MATCH_FOUND => {
                let mut plate_solution = PlateSolution::default();
                
                // Populate core coordinates
                plate_solution.image_sky_coord = Some(CelestialCoord {
                    ra: result.ra.unwrap_or(0.0),
                    dec: result.dec.unwrap_or(0.0),
                });
                plate_solution.roll = result.roll.unwrap_or(0.0);
                plate_solution.fov = result.fov.unwrap_or(0.0);
                plate_solution.distortion = result.distortion;
                
                // Populate quality metrics
                plate_solution.rmse = result.rmse.unwrap_or(0.0);
                plate_solution.num_matches = result.matches.unwrap_or(0) as i32;
                plate_solution.prob = result.prob.unwrap_or(0.0);
                
                // Populate timing information (convert ms to proto Duration)
                plate_solution.solve_time = Some(prost_types::Duration {
                    seconds: (result.t_solve / 1000.0) as i64,
                    nanos: ((result.t_solve % 1000.0) * 1_000_000.0) as i32,
                });

                // Note: rotation_matrix, target_pixel, and target_sky_coord are omitted 
                // here as the provided SolveResult does not include the raw rotation matrix,
                // and coordinate transforms are typically handled by Cedar's astro_util.rs.

                Ok(plate_solution)
            }
            NO_MATCH => Err(not_found_error("Solver failed to find a match.")),
            TIMEOUT => Err(deadline_exceeded_error("Solver timed out.")),
            CANCELLED => Err(deadline_exceeded_error("Solve operation was cancelled.")),
            TOO_FEW => Err(invalid_argument_error("Too few centroids to attempt solve.")),
            _ => Err(not_found_error("Solver encountered an unknown error state.")),
        }
    }

    fn cancel(&self) {
        // Atomically set the flag to signal the solver loop to terminate
        self.cancelled.store(true, AtomicOrdering::SeqCst);
        
        // Attempt to set the flag on the inner instance if the lock is available
        if let Ok(mut inner) = self.inner.try_lock() {
            inner.set_cancelled(true);
        }
    }

    fn default_timeout(&self) -> Duration {
        // Return a default solve duration if not specified by params
        Duration::from_secs(1)
    }
}

