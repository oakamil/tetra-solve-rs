// Required Notice: Copyright (c) 2026 Omair Kamil
//
// This file is a derivative work - a Python interface to the optimized Rust port
// of the cedar-solve and esa/tetra3 projects. The original underlying code is
// licensed under the Apache License, Version 2.0.
// Original Copyright (c) European Space Agency, Steven Rosenthal, and contributors.
//
// This derivative work is licensed under the PolyForm Noncommercial License 1.0.0.
// You may not use this file except in compliance with the PolyForm Noncommercial
// License 1.0.0. A copy of the License is located in the LICENSE.md file in the
// root of this repository.
//
// Commercial use of this software is strictly prohibited without a separate
// commercial license.
//
//
// Cedar Solve license:
//    Copyright 2023 Steven Rosenthal smr@dt3.org
//
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//
//        https://www.apache.org/licenses/LICENSE-2.0
//
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.
//
//
// tetra3 license:
//    Copyright 2019 the European Space Agency
//
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//
//        https://www.apache.org/licenses/LICENSE-2.0
//
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.
//
//
// Original Tetra license notice:
//    Copyright (c) 2016 brownj4
//
//    Permission is hereby granted, free of charge, to any person obtaining a copy
//    of this software and associated documentation files (the "Software"), to deal
//    in the Software without restriction, including without limitation the rights
//    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//    copies of the Software, and to permit persons to whom the Software is
//    furnished to do so, subject to the following conditions:
//
//    The above copyright notice and this permission notice shall be included in all
//    copies or substantial portions of the Software.
//
//    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//    SOFTWARE.

use numpy::{IntoPyArray, PyArrayMethods, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::path::PathBuf;

use tetra3_core::Tetra3;
use tetra3_core::extractor::{BgSubMode, ExtractOptions, SigmaMode};
use tetra3_core::solver::SolveOptions;

/// Python wrapper for the highly optimized Tetra3 engine.
#[pyclass(name = "Tetra3", unsendable)]
pub struct PyTetra3 {
    inner: Tetra3,
}

#[pymethods]
impl PyTetra3 {
    /// Creates a new Tetra3 instance.
    /// The database is lazy-loaded, meaning it won't hit the disk until
    /// the first plate solving operation is executed.
    #[new]
    fn new(database_path: String) -> Self {
        Self {
            inner: Tetra3::new(PathBuf::from(database_path)),
        }
    }

    /// Extracts centroids from a 2D NumPy array.
    /// Uses PyO3's buffer protocol to read directly from Python's memory (zero-copy).
    ///
    /// Returns:
    ///     numpy.ndarray or tuple: If `return_moments=False` and `return_images=False` (the defaults)
    ///     an array of shape (N,2) is returned with centroid positions (y down, x right) of the
    ///     found spots in order of brightness. If `return_moments=True` a tuple of numpy arrays
    ///     is returned with: (N,2) centroid positions, N sum, N area, (N,3) xx yy and xy second
    ///     moments, N major over minor axis ratio. If `return_images=True` a tuple is returned
    ///     with the results as defined previously and a dictionary with images and data of partial
    ///     results. The keys are: `cropped_and_downsampled`, `removed_background`, `binary_mask`.
    #[pyo3(signature = (image, **kwargs))]
    fn get_centroids_from_image<'py>(
        &mut self,
        py: Python<'py>,
        image: PyReadonlyArray2<'py, f32>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // 1. Parse Python **kwargs into the native ExtractOptions struct
        let options = parse_extract_options(kwargs)?;

        // Check if Python specifically asked for moments or images
        let return_moments = kwargs
            .and_then(|d| d.get_item("return_moments").ok().flatten())
            .and_then(|v| v.extract::<bool>().ok())
            .unwrap_or(false);

        let return_images = kwargs
            .and_then(|d| d.get_item("return_images").ok().flatten())
            .and_then(|v| v.extract::<bool>().ok())
            .unwrap_or(false);

        // 2. Extract a zero-copy ndarray view directly from the Python array memory
        let img_view = image.as_array();

        // 3. Run the native Rust extraction pipeline
        let result = self.inner.get_centroids_from_image(&img_view, options);
        let num_centroids = result.centroids.len();

        // 4. Build the core results (ndarray or tuple of ndarrays)
        let core_result: Bound<'py, PyAny> = if return_moments {
            // Unpack everything for moments
            let mut cents = Vec::with_capacity(num_centroids * 2);
            let mut sums = Vec::with_capacity(num_centroids);
            let mut areas = Vec::with_capacity(num_centroids);
            let mut moments = Vec::with_capacity(num_centroids * 3);
            let mut ratios = Vec::with_capacity(num_centroids);

            for c in &result.centroids {
                cents.push(c.y);
                cents.push(c.x);
                sums.push(c.sum);
                areas.push(c.area as f64);
                moments.push(c.m2_xx);
                moments.push(c.m2_yy);
                moments.push(c.m2_xy);
                ratios.push(c.axis_ratio);
            }

            let py_cents = numpy::PyArray1::from_slice(py, &cents)
                .reshape([num_centroids, 2])
                .unwrap();
            let py_sums = numpy::PyArray1::from_slice(py, &sums);
            let py_areas = numpy::PyArray1::from_slice(py, &areas);
            let py_moments = numpy::PyArray1::from_slice(py, &moments)
                .reshape([num_centroids, 3])
                .unwrap();
            let py_ratios = numpy::PyArray1::from_slice(py, &ratios);

            let elements: Vec<Bound<'py, pyo3::types::PyAny>> = vec![
                py_cents.into_any(),
                py_sums.into_any(),
                py_areas.into_any(),
                py_moments.into_any(),
                py_ratios.into_any(),
            ];

            // PyTuple::new now safely unwrapped with ? before casting
            PyTuple::new(py, elements)?.into_any()
        } else {
            // Default: just [y, x] centroids
            let mut cents = Vec::with_capacity(num_centroids * 2);
            for c in &result.centroids {
                cents.push(c.y);
                cents.push(c.x);
            }
            let py_cents = numpy::PyArray1::from_slice(py, &cents)
                .reshape([num_centroids, 2])
                .unwrap();
            py_cents.into_any()
        };

        // 5. If return_images is requested, wrap the core_result in another tuple with the image dictionary
        if return_images {
            let images_dict = PyDict::new(py);
            if let Some(debug_images) = result.debug_images {
                let py_cropped = debug_images.cropped_and_downsampled.into_pyarray(py);
                images_dict.set_item("cropped_and_downsampled", py_cropped)?;

                let py_removed_bg = debug_images.removed_background.into_pyarray(py);
                images_dict.set_item("removed_background", py_removed_bg)?;

                let py_mask = debug_images.binary_mask.into_pyarray(py);
                images_dict.set_item("binary_mask", py_mask)?;
            }

            // Return: (core_results, dict_of_images)
            let elements: Vec<Bound<'py, pyo3::types::PyAny>> =
                vec![core_result, images_dict.into_any()];
            // PyTuple::new safely unwrapped with ? before casting
            Ok(PyTuple::new(py, elements)?.into_any())
        } else {
            // Return: core_results directly
            Ok(core_result)
        }
    }

    /// Runs plate solving from pre-extracted centroids.
    /// Returns a dictionary containing the solution and execution times.
    ///
    /// Returns:
    ///     dict: A dictionary with the following keys is returned:
    ///         - 'RA': Right ascension of centre of image in degrees.
    ///         - 'Dec': Declination of centre of image in degrees.
    ///         - 'Roll': Rotation in degrees of celestial north relative to image's "up"
    ///           direction (towards y=0). Zero when north and up coincide; a positive
    ///           roll angle means north is counter-clockwise from image "up".
    ///         - 'FOV': Calculated horizontal field of view of the provided image.
    ///         - 'distortion': Calculated distortion of the provided image. Omitted if
    ///           the caller's distortion estimate is None.
    ///         - 'RMSE': RMS residual of matched stars in arcseconds.
    ///         - 'P90E': 90 percentile matched star residual in arcseconds.
    ///         - 'MAXE': Maximum matched star residual in arcseconds.
    ///         - 'Matches': Number of stars in the image matched to the database.
    ///         - 'Prob': Probability that the solution is a false-positive.
    ///         - 'epoch_equinox': The celestial RA/Dec equinox reference epoch.
    ///         - 'epoch_proper_motion': The epoch the database proper motions were propageted to.
    ///         - 'T_solve': Time spent searching for a match in milliseconds.
    ///         - 'RA_target': Right ascension in degrees of the pixel positions passed in
    ///           target_pixel. Not included if target_pixel=None (the default). If a Kx2 array
    ///           of target_pixel was passed, this will be a length K list.
    ///         - 'Dec_target': Declination in degrees of the pixel positions in target_pixel.
    ///         - 'x_target': image x coordinates for the sky positions passed in target_sky_coord.
    ///         - 'y_target': image y coordinates for the sky positions passed in target_sky_coord.
    ///         - 'matched_stars': An Mx3 list with the (RA, Dec, magnitude) of the M matched stars
    ///           that were used in the solution. RA/Dec in degrees.
    ///         - 'matched_centroids': An Mx2 list with the (y, x) pixel coordinates in the image
    ///           corresponding to each matched star.
    ///         - 'matched_catID': The catalogue ID corresponding to each matched star.
    ///         - 'catalog_stars': A list of tuples (RA, Dec, magnitude, y, x). RA/Dec in degrees.
    ///         - 'rotation_matrix': 3x3 rotation matrix.
    ///         - 'status': One of MATCH_FOUND, NO_MATCH, TIMEOUT, CANCELLED, TOO_FEW.
    #[pyo3(signature = (centroids, size, **kwargs))]
    fn solve_from_centroids<'py>(
        &mut self,
        py: Python<'py>,
        centroids: PyReadonlyArray2<'py, f64>,
        size: (f64, f64),
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let solve_options = parse_solve_options(kwargs)?;

        // Convert the borrowed view into an owned array for the core solver
        let cent_owned = centroids.as_array().to_owned();

        // Run the native Rust solve pipeline, skipping extraction
        let solution = self
            .inner
            .solve_from_centroids(&cent_owned, size, solve_options)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let out_dict = PyDict::new(py);

        // Base coordinate properties
        out_dict.set_item("RA", solution.ra)?;
        out_dict.set_item("Dec", solution.dec)?;
        out_dict.set_item("Roll", solution.roll)?;
        out_dict.set_item("FOV", solution.fov)?;
        out_dict.set_item("distortion", solution.distortion)?;

        // Metrics and statistics
        out_dict.set_item("RMSE", solution.rmse)?;
        out_dict.set_item("P90E", solution.p90e)?;
        out_dict.set_item("MAXE", solution.maxe)?;
        out_dict.set_item("Matches", solution.matches)?;
        out_dict.set_item("Prob", solution.prob)?;
        out_dict.set_item("is_mirrored", solution.is_mirrored)?;

        // Epochs & Status
        out_dict.set_item("epoch_equinox", solution.epoch_equinox)?;
        out_dict.set_item("epoch_proper_motion", solution.epoch_proper_motion)?;
        out_dict.set_item("status", format!("{:?}", solution.status))?;

        // Timings
        out_dict.set_item("T_solve", solution.t_solve_ms)?;

        // Target mapping (Vecs map naturally to Python lists via PyO3)
        if let Some(target_ra) = solution.target_ra {
            out_dict.set_item("RA_target", target_ra)?;
        }
        if let Some(target_dec) = solution.target_dec {
            out_dict.set_item("Dec_target", target_dec)?;
        }
        if let Some(target_y) = solution.target_y {
            out_dict.set_item("y_target", target_y)?;
        }
        if let Some(target_x) = solution.target_x {
            out_dict.set_item("x_target", target_x)?;
        }

        // Star structures mapped to lists
        if let Some(matched_centroids) = solution.matched_centroids {
            out_dict.set_item("matched_centroids", matched_centroids)?;
        }
        if let Some(matched_stars) = solution.matched_stars {
            out_dict.set_item("matched_stars", matched_stars)?;
        }
        if let Some(matched_cat_id) = solution.matched_cat_id {
            out_dict.set_item("matched_catID", matched_cat_id)?;
        }
        if let Some(catalog_stars) = solution.catalog_stars {
            out_dict.set_item("catalog_stars", catalog_stars)?;
        }

        if let Some(rm) = solution.rotation_matrix {
            let flat_slice = rm.as_slice().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("Matrix not contiguous")
            })?;
            let py_matrix = numpy::PyArray1::from_slice(py, flat_slice)
                .reshape([3, 3])
                .unwrap();
            out_dict.set_item("rotation_matrix", py_matrix)?;
        }

        Ok(out_dict)
    }

    /// Runs extraction and plate solving in one uninterrupted pipeline.
    /// Returns a dictionary containing the solution and execution times.
    #[pyo3(signature = (image, **kwargs))]
    fn solve_from_image<'py>(
        &mut self,
        py: Python<'py>,
        image: PyReadonlyArray2<'py, f32>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let extract_options = parse_extract_options(kwargs)?;
        let solve_options = parse_solve_options(kwargs)?;

        let img_view = image.as_array();

        let (solution, ext_time) = self
            .inner
            .solve_from_image(&img_view, extract_options, solve_options)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let out_dict = PyDict::new(py);

        // Base coordinate properties
        out_dict.set_item("RA", solution.ra)?;
        out_dict.set_item("Dec", solution.dec)?;
        out_dict.set_item("Roll", solution.roll)?;
        out_dict.set_item("FOV", solution.fov)?;
        out_dict.set_item("distortion", solution.distortion)?;

        // Metrics and statistics
        out_dict.set_item("RMSE", solution.rmse)?;
        out_dict.set_item("P90E", solution.p90e)?;
        out_dict.set_item("MAXE", solution.maxe)?;
        out_dict.set_item("Matches", solution.matches)?;
        out_dict.set_item("Prob", solution.prob)?;
        out_dict.set_item("is_mirrored", solution.is_mirrored)?;

        // Epochs & Status
        out_dict.set_item("epoch_equinox", solution.epoch_equinox)?;
        out_dict.set_item("epoch_proper_motion", solution.epoch_proper_motion)?;
        out_dict.set_item("status", format!("{:?}", solution.status))?;

        // Timings
        out_dict.set_item("T_extract", ext_time)?;
        out_dict.set_item("T_solve", solution.t_solve_ms)?;

        // Target mapping (Vecs map naturally to Python lists via PyO3)
        if let Some(target_ra) = solution.target_ra {
            out_dict.set_item("RA_target", target_ra)?;
        }
        if let Some(target_dec) = solution.target_dec {
            out_dict.set_item("Dec_target", target_dec)?;
        }
        if let Some(target_y) = solution.target_y {
            out_dict.set_item("y_target", target_y)?;
        }
        if let Some(target_x) = solution.target_x {
            out_dict.set_item("x_target", target_x)?;
        }

        // Star structures mapped to lists
        if let Some(matched_centroids) = solution.matched_centroids {
            out_dict.set_item("matched_centroids", matched_centroids)?;
        }
        if let Some(matched_stars) = solution.matched_stars {
            out_dict.set_item("matched_stars", matched_stars)?;
        }
        if let Some(matched_cat_id) = solution.matched_cat_id {
            out_dict.set_item("matched_catID", matched_cat_id)?;
        }
        if let Some(catalog_stars) = solution.catalog_stars {
            out_dict.set_item("catalog_stars", catalog_stars)?;
        }

        if let Some(rm) = solution.rotation_matrix {
            let flat_slice = rm.as_slice().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("Matrix not contiguous")
            })?;
            let py_matrix = numpy::PyArray1::from_slice(py, flat_slice)
                .reshape([3, 3])
                .unwrap();
            out_dict.set_item("rotation_matrix", py_matrix)?;
        }

        Ok(out_dict)
    }
}

// --- Helper Functions to Map Python kwargs to Rust Structs ---

fn parse_extract_options(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<ExtractOptions> {
    let mut options = ExtractOptions::default();

    if let Some(dict) = kwargs {
        if let Some(val) = dict.get_item("sigma")? {
            options.sigma = val.extract()?;
        }
        if let Some(val) = dict.get_item("image_th")? {
            options.image_th = val.extract()?;
        }
        if let Some(val) = dict.get_item("downsample")? {
            options.downsample = val.extract()?;
        }
        if let Some(val) = dict.get_item("filtsize")? {
            options.filtsize = val.extract()?;
        }
        if let Some(val) = dict.get_item("binary_open")? {
            options.binary_open = val.extract()?;
        }
        if let Some(val) = dict.get_item("centroid_window")? {
            options.centroid_window = val.extract()?;
        }
        if let Some(val) = dict.get_item("min_area")? {
            options.min_area = val.extract()?;
        }
        if let Some(val) = dict.get_item("max_area")? {
            options.max_area = val.extract()?;
        }
        if let Some(val) = dict.get_item("min_sum")? {
            options.min_sum = val.extract()?;
        }
        if let Some(val) = dict.get_item("max_sum")? {
            options.max_sum = val.extract()?;
        }
        if let Some(val) = dict.get_item("max_axis_ratio")? {
            options.max_axis_ratio = val.extract()?;
        }
        if let Some(val) = dict.get_item("max_returned")? {
            options.max_returned = val.extract()?;
        }
        if let Some(val) = dict.get_item("return_images")? {
            options.return_images = val.extract()?;
        }

        // Background Subtraction Mode
        if let Some(val) = dict.get_item("bg_sub_mode")? {
            if val.is_none() {
                options.bg_sub_mode = None;
            } else {
                let mode_str: String = val.extract()?;
                options.bg_sub_mode = match mode_str.to_lowercase().as_str() {
                    "local_median" => Some(BgSubMode::LocalMedian),
                    "local_mean" => Some(BgSubMode::LocalMean),
                    "global_median" => Some(BgSubMode::GlobalMedian),
                    "global_mean" => Some(BgSubMode::GlobalMean),
                    "none" => None,
                    _ => {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "Invalid bg_sub_mode: {}",
                            mode_str
                        )));
                    }
                };
            }
        }

        // Sigma Threshold Mode
        if let Some(val) = dict.get_item("sigma_mode")? {
            let mode_str: String = val.extract()?;
            options.sigma_mode = match mode_str.to_lowercase().as_str() {
                "local_median_abs" => SigmaMode::LocalMedianAbs,
                "local_root_square" => SigmaMode::LocalRootSquare,
                "global_median_abs" => SigmaMode::GlobalMedianAbs,
                "global_root_square" => SigmaMode::GlobalRootSquare,
                _ => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Invalid sigma_mode: {}",
                        mode_str
                    )));
                }
            };
        }
    }
    Ok(options)
}

fn parse_solve_options(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<SolveOptions> {
    let mut options = SolveOptions::default();

    if let Some(dict) = kwargs {
        if let Some(val) = dict.get_item("fov_estimate")? {
            options.fov_estimate = val.extract()?;
        }
        if let Some(val) = dict.get_item("fov_max_error")? {
            options.fov_max_error = val.extract()?;
        }
        if let Some(val) = dict.get_item("match_radius")? {
            options.match_radius = val.extract()?;
        }
        if let Some(val) = dict.get_item("match_threshold")? {
            options.match_threshold = val.extract()?;
        }
        if let Some(val) = dict.get_item("solve_timeout")? {
            options.solve_timeout_ms = val.extract()?;
        }
        if let Some(val) = dict.get_item("distortion")? {
            options.distortion = val.extract()?;
        }
        if let Some(val) = dict.get_item("match_max_error")? {
            options.match_max_error = val.extract()?;
        }
        if let Some(val) = dict.get_item("return_matches")? {
            options.return_matches = val.extract()?;
        }
        if let Some(val) = dict.get_item("return_catalog")? {
            options.return_catalog = val.extract()?;
        }
        if let Some(val) = dict.get_item("return_rotation_matrix")? {
            options.return_rotation_matrix = val.extract()?;
        }

        // Target configurations (Parsing 2D arrays directly into ndarrays)
        if let Some(val) = dict.get_item("target_pixel")? {
            if !val.is_none() {
                let py_arr: numpy::PyReadonlyArray2<f64> = val.extract()?;
                options.target_pixel = Some(py_arr.as_array().to_owned());
            }
        }
        if let Some(val) = dict.get_item("target_sky_coord")? {
            if !val.is_none() {
                let py_arr: numpy::PyReadonlyArray2<f64> = val.extract()?;
                options.target_sky_coord = Some(py_arr.as_array().to_owned());
            }
        }
    }
    Ok(options)
}

// --- Module Initialization ---

/// The module initialization function. This matches the name defined in Cargo.toml.
/// This exposes the PyTetra3 class to Python under the name `Tetra3`.
#[pymodule]
#[pyo3(name = "tetra3_py")]
fn tetra3(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTetra3>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyModule;
    use std::path::Path;

    const PYTHON_EXAMPLE_CODE: &std::ffi::CStr = cr#"
import os
import time
import glob
import numpy as np
from PIL import Image

import tetra3 

def run_benchmark(db_path, img_dir):
    if not os.path.exists(db_path):
        return
    if not os.path.exists(img_dir):
        return

    print(f"\n{'Filename':<40} | {'RA':<8} | {'Dec':<8} | {'Extract (ms)':<12} | {'Solve (ms)':<12} | {'Total Inv (ms)':<14}")
    print("-" * 105)

    t3 = tetra3.Tetra3(db_path)
    search_pattern = os.path.join(img_dir, "*.jpg")
    image_files = sorted(glob.glob(search_pattern))

    for img_path in image_files:
        filename = os.path.basename(img_path)
        img = Image.open(img_path).convert('L')
        img_arr = np.asarray(img, dtype=np.float32)
        
        t0 = time.perf_counter()
        result = t3.solve_from_image(img_arr)
        inv_time_ms = (time.perf_counter() - t0) * 1000.0

        ra = result.get("RA")
        dec = result.get("Dec")
        ext_ms = result.get("T_extract", 0.0)
        solve_ms = result.get("T_solve", 0.0)

        ra_str = f"{ra:.3f}" if ra is not None else "N/A"
        dec_str = f"{dec:.3f}" if dec is not None else "N/A"

        print(f"{filename:<40.40} | {ra_str:<8} | {dec_str:<8} | {ext_ms:<12.2f} | {solve_ms:<12.2f} | {inv_time_ms:<14.2f}")

    print("-" * 105 + "\n")
"#;

    const PYTHON_SPLIT_EXAMPLE_CODE: &std::ffi::CStr = cr#"
import os
import time
import glob
import numpy as np
from PIL import Image

import tetra3 

def run_benchmark_split(db_path, img_dir):
    if not os.path.exists(db_path):
        return
    if not os.path.exists(img_dir):
        return

    print(f"\n{'Filename':<40} | {'RA':<8} | {'Dec':<8} | {'Extract (ms)':<12} | {'Solve (ms)':<12} | {'Total Inv (ms)':<14}")
    print("-" * 105)

    t3 = tetra3.Tetra3(db_path)
    search_pattern = os.path.join(img_dir, "*.jpg")
    image_files = sorted(glob.glob(search_pattern))

    for img_path in image_files:
        filename = os.path.basename(img_path)
        img = Image.open(img_path).convert('L')
        img_arr = np.asarray(img, dtype=np.float32)
        img_size = (img_arr.shape[0], img_arr.shape[1])
        
        t0 = time.perf_counter()
        
        # Step 1: Extract centroids
        # Using the corrected get_centroids_from_image which returns the ndarray directly
        centroids = t3.get_centroids_from_image(img_arr)
        
        t1 = time.perf_counter()
        ext_ms = (t1 - t0) * 1000.0
        
        # Step 2: Solve from extracted centroids
        # Create a 2D numpy float64 array for a non-central target_pixel 
        # (e.g., top-left quadrant of the image)
        target_y, target_x = img_size[0] / 4.0, img_size[1] / 4.0
        target_px = np.array([[target_y, target_x]], dtype=np.float64)
        
        result = t3.solve_from_centroids(centroids, img_size, target_pixel=target_px)
        
        inv_time_ms = (time.perf_counter() - t0) * 1000.0

        ra = result.get("RA")
        dec = result.get("Dec")
        solve_ms = result.get("T_solve", 0.0)

        ra_str = f"{ra:.3f}" if ra is not None else "N/A"
        dec_str = f"{dec:.3f}" if dec is not None else "N/A"

        print(f"{filename:<40.40} | {ra_str:<8} | {dec_str:<8} | {ext_ms:<12.2f} | {solve_ms:<12.2f} | {inv_time_ms:<14.2f}")

        # Ensure the vector lists correctly mapped across the PyO3 boundary
        ra_target = result.get("RA_target")
        dec_target = result.get("Dec_target")
        if ra_target is not None and dec_target is not None:
             print(f"  -> Target Pixel ({target_y}, {target_x}) solved to RA: {ra_target[0]:.3f}, Dec: {dec_target[0]:.3f}")

    print("-" * 105 + "\n")
"#;

    #[test]
    #[ignore]
    fn test_python_wrapper_from_python_script() {
        let db_path = "../tetra3/tests/fixtures/default_database.npz";
        let img_dir = "../tetra3/tests/fixtures/sample_images";

        if !Path::new(db_path).exists() || !Path::new(img_dir).exists() {
            return; // Skip gracefully if data isn't present
        }

        Python::initialize();
        Python::attach(|py| {
            // 1. Manually build the `tetra3` module
            let tetra3_mod = PyModule::new(py, "tetra3").unwrap();
            tetra3_mod.add_class::<PyTetra3>().unwrap();

            // 2. Inject the module into Python's `sys.modules`
            let sys = py.import("sys").unwrap();
            sys.getattr("modules")
                .unwrap()
                .set_item("tetra3", tetra3_mod)
                .unwrap();

            // 3. Compile and load the embedded script
            let main_mod =
                PyModule::from_code(py, PYTHON_EXAMPLE_CODE, c"example.py", c"example").unwrap();

            // 4. Run it
            let run_benchmark = main_mod.getattr("run_benchmark").unwrap();
            run_benchmark.call1((db_path, img_dir)).unwrap();
        });
    }

    #[test]
    #[ignore]
    fn test_python_wrapper_split_pipeline() {
        let db_path = "../tetra3/tests/fixtures/default_database.npz";
        let img_dir = "../tetra3/tests/fixtures/sample_images";

        if !Path::new(db_path).exists() || !Path::new(img_dir).exists() {
            return; // Skip gracefully if data isn't present
        }

        Python::initialize();
        Python::attach(|py| {
            // 1. Manually build the `tetra3` module
            let tetra3_mod = PyModule::new(py, "tetra3").unwrap();
            tetra3_mod.add_class::<PyTetra3>().unwrap();

            // 2. Inject the module into Python's `sys.modules`
            let sys = py.import("sys").unwrap();
            sys.getattr("modules")
                .unwrap()
                .set_item("tetra3", tetra3_mod)
                .unwrap();

            // 3. Compile and load the newly split embedded script
            let main_mod = PyModule::from_code(
                py,
                PYTHON_SPLIT_EXAMPLE_CODE,
                c"example_split.py",
                c"example_split",
            )
            .unwrap();

            // 4. Run the newly added benchmark
            let run_benchmark = main_mod.getattr("run_benchmark_split").unwrap();
            run_benchmark.call1((db_path, img_dir)).unwrap();
        });
    }
}
