use core::fmt::Debug;
use nalgebra::geometry::{Isometry3, Point3, Quaternion, Translation3, UnitQuaternion};
use nalgebra::{Matrix3, RowVector3, Vector3};
use ndarray::ArrayBase;
use ndarray::Ix3;
use ndarray::ViewRepr;
use ndarray::{ArrayView3, ArrayViewMut3};
use ndarray::{Axis, Data, DataMut, Zip};
use numpy::PyArray3;
use pyo3::prelude::*;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::ops::{Index, IndexMut};

pub fn matrices_from_array(
    in_matrix_ijk_to_lps: [f32; 9],
    origin_lps: [f32; 3],
) -> (Isometry3<f32>, Isometry3<f32>) {
    let rotation_matrix = Matrix3::from_rows(&[
        RowVector3::new(
            in_matrix_ijk_to_lps[0],
            in_matrix_ijk_to_lps[1],
            in_matrix_ijk_to_lps[2],
        ),
        RowVector3::new(
            in_matrix_ijk_to_lps[3],
            in_matrix_ijk_to_lps[4],
            in_matrix_ijk_to_lps[5],
        ),
        RowVector3::new(
            in_matrix_ijk_to_lps[6],
            in_matrix_ijk_to_lps[7],
            in_matrix_ijk_to_lps[8],
        ),
    ]);

    let max_iter = 10000;
    let eps = 1e-6;
    let unit_quaternion_ijk_to_lps = UnitQuaternion::from_matrix_eps(
        &rotation_matrix,
        eps,
        max_iter,
        UnitQuaternion::identity(),
    );

    let translation = Translation3::<f32>::new(origin_lps[0], origin_lps[1], origin_lps[2]);
    let matrix_ijk_to_lps = Isometry3::<f32>::from_parts(translation, unit_quaternion_ijk_to_lps);
    let matrix_lps_to_ijk = matrix_ijk_to_lps.inverse();
    (matrix_ijk_to_lps, matrix_lps_to_ijk)
}
#[derive(Copy, Clone, Debug)]
pub struct VolumeGeometryTTT {
    pub origin_lps: [f32; 3],
    pub spacing: [f32; 3],
    pub matrix_ijk_to_lps: Isometry3<f32>,
    matrix_lps_to_ijk: Isometry3<f32>,
}

impl VolumeGeometryTTT {
    pub fn new(
        origin_lps: [f32; 3],
        spacing: [f32; 3],
        in_matrix_lps_to_ijk: [f32; 9],
    ) -> VolumeGeometryTTT {
        let (matrix_ijk_to_lps, matrix_lps_to_ijk) =
            matrices_from_array(in_matrix_lps_to_ijk, origin_lps);
        VolumeGeometryTTT {
            origin_lps,
            spacing,
            matrix_ijk_to_lps,
            matrix_lps_to_ijk,
        }
    }

    pub fn ijk_2_lps(&self, p_ijk: [f32; 3]) -> [f32; 3] {
        let p_lps = self.matrix_ijk_to_lps.transform_point(&Point3::new(
            p_ijk[0] * self.spacing[0],
            p_ijk[1] * self.spacing[1],
            p_ijk[2] * self.spacing[2],
        ));
        [p_lps.x, p_lps.y, p_lps.z]
    }

    pub fn lps_2_ijk(&self, p_lps: [f32; 3]) -> [f32; 3] {
        let p_vox = self
            .matrix_lps_to_ijk
            .transform_point(&Point3::new(p_lps[0], p_lps[1], p_lps[2]));
        [
            (p_vox.x / self.spacing[0]),
            (p_vox.y / self.spacing[1]),
            (p_vox.z / self.spacing[2]),
        ]
    }
}

pub struct VolumeBaseTTT<S, U>
where
    S: Data<Elem = U>,
{
    pub data: ArrayBase<S, Ix3>,
    pub geom: VolumeGeometryTTT,
}

impl<U, S: Data<Elem = U>> VolumeBaseTTT<S, U> {
    pub fn new(
        data: ArrayBase<S, Ix3>,
        origin_lps: [f32; 3],
        spacing: [f32; 3],
        in_matrix_lps_to_ijk: [f32; 9],
    ) -> VolumeBaseTTT<S, U> {
        VolumeBaseTTT {
            data,
            geom: VolumeGeometryTTT::new(origin_lps, spacing, in_matrix_lps_to_ijk),
        }
    }

    #[inline(always)]
    pub fn get_voxel(&self, i: usize, j: usize, k: usize) -> Option<&U> {
        self.data.get((k, j, i))
    }
}

impl<U, S: Data<Elem = U>> Index<[usize; 3]> for VolumeBaseTTT<S, U> {
    type Output = U;
    #[inline(always)]
    fn index(&self, index_ijk: [usize; 3]) -> &U {
        unsafe { self.data.uget((index_ijk[2], index_ijk[1], index_ijk[0])) }
    }
}

impl<U, S: DataMut<Elem = U>> IndexMut<[usize; 3]> for VolumeBaseTTT<S, U> {
    #[inline(always)]
    fn index_mut(&mut self, index: [usize; 3]) -> &mut U {
        unsafe { self.data.uget_mut((index[2], index[1], index[0])) }
    }
}

pub type VolumeViewF32<'a> = VolumeBaseTTT<ViewRepr<&'a f32>, f32>;
pub type VolumeViewMutF32<'a> = VolumeBaseTTT<ViewRepr<&'a mut f32>, f32>;


pub type VolumeViewU8<'a> = VolumeBaseTTT<ViewRepr<&'a u8>,u8>;
pub type VolumeViewMutU8<'a> = VolumeBaseTTT<ViewRepr<&'a mut u8>, u8>;

fn neighbor_data<T: Copy>(grid_i: [i32; 3], far_l: [i32; 3], grid_ui: [usize; 3], far_r: [usize; 3], grid_f: [f32; 3], in_vol: &VolumeBaseTTT<ViewRepr<&T>, T>, out_val: T)-> [T;8] {
    let mut data: [T; 8] = [out_val; 8];
    if grid_i[0] == far_l[0]
        || grid_i[1] == far_l[1]
        || grid_i[2] == far_l[2]
        || grid_ui[0] == far_r[0]
        || grid_ui[1] == far_r[1]
        || grid_ui[2] == far_r[2]
    {
        let mut p = [grid_ui[0], grid_ui[1], grid_ui[2]];
        let mut pp = [
            (grid_f[0] + 1.0) as usize,
            (grid_f[1] + 1.0) as usize,
            (grid_f[2] + 1.0) as usize,
        ];
        if grid_i[0] < far_l[0] + 1 {
            p[0] = std::usize::MAX;
            pp[0] = 0;
        }
        if grid_i[1] < far_l[1] + 1 {
            p[1] = std::usize::MAX;
            pp[1] = 0;
        }
        if grid_i[2] < far_l[2] + 1 {
            p[2] = std::usize::MAX;
            pp[2] = 0;
        }
        data[0] = *in_vol.get_voxel(p[0], p[1], p[2]).unwrap_or(&out_val);
        data[1] = *in_vol.get_voxel(p[0], p[1], pp[2]).unwrap_or(&out_val);
        data[2] = *in_vol.get_voxel(p[0], pp[1], p[2]).unwrap_or(&out_val);
        data[3] = *in_vol.get_voxel(p[0], pp[1], pp[2]).unwrap_or(&out_val);
        data[4] = *in_vol.get_voxel(pp[0], p[1], p[2]).unwrap_or(&out_val);
        data[5] = *in_vol.get_voxel(pp[0], p[1], pp[2]).unwrap_or(&out_val);
        data[6] = *in_vol.get_voxel(pp[0], pp[1], p[2]).unwrap_or(&out_val);
        data[7] = *in_vol.get_voxel(pp[0], pp[1], pp[2]).unwrap_or(&out_val);
    } else {
        data[0] = in_vol[[grid_ui[0], grid_ui[1], grid_ui[2]]];
        data[1] = in_vol[[grid_ui[0], grid_ui[1], grid_ui[2] + 1]];
        data[2] = in_vol[[grid_ui[0], grid_ui[1] + 1, grid_ui[2]]];
        data[3] = in_vol[[grid_ui[0], grid_ui[1] + 1, grid_ui[2] + 1]];
        data[4] = in_vol[[grid_ui[0] + 1, grid_ui[1], grid_ui[2]]];
        data[5] = in_vol[[grid_ui[0] + 1, grid_ui[1], grid_ui[2] + 1]];
        data[6] = in_vol[[grid_ui[0] + 1, grid_ui[1] + 1, grid_ui[2]]];
        data[7] = in_vol[[grid_ui[0] + 1, grid_ui[1] + 1, grid_ui[2] + 1]];
    }
    data
}

pub fn trinilear_interpolation_(in_vol: &VolumeViewF32, out_vol: &mut VolumeViewMutF32, out_val: f32) {
    // Get dimensions of input volume
    let in_dims_kji = in_vol.data.shape();
    let in_dims_ijk = Point3::<usize>::new(in_dims_kji[2], in_dims_kji[1], in_dims_kji[0]);

    // Define indices for far left and far right corners
    let far_l = [-1, -1, -1];
    let far_r = [in_dims_ijk.x - 1, in_dims_ijk.y - 1, in_dims_ijk.z - 1];

    let out_geom = out_vol.geom;
    // Perform trilinear interpolation in parallel using Rayon
    Zip::indexed(&mut out_vol.data).par_for_each(|index_kji, value| {
        // Transform output point to LPS space
        let p_lps =
            out_geom.ijk_2_lps([index_kji.2 as f32, index_kji.1 as f32, index_kji.0 as f32]);

        // Transform output point to voxel space of input volume
        let p_vox_ijk = in_vol.geom.lps_2_ijk(p_lps);

        // Compute grid indices and coordinates
        let grid_f = [
            p_vox_ijk[0].floor(),
            p_vox_ijk[1].floor(),
            p_vox_ijk[2].floor(),
        ];
        let grid_i: [i32; 3] = [grid_f[0] as i32, grid_f[1] as i32, grid_f[2] as i32];
        let grid_ui: [usize; 3] = [grid_f[0] as usize, grid_f[1] as usize, grid_f[2] as usize];
        

        // Handle out-of-bounds cases
        if grid_i[0] < far_l[0]
            || grid_i[1] < far_l[1]
            || grid_i[2] < far_l[2]
            || grid_ui[0] > far_r[0]
            || grid_ui[1] > far_r[1]
            || grid_ui[2] > far_r[2]
        {
            *value = out_val;
        } else {
            let data =neighbor_data(grid_i, far_l, grid_ui, far_r, grid_f, in_vol, out_val);
            let delta: [f32; 3] = [
                p_vox_ijk[0] - grid_f[0],
                p_vox_ijk[1] - grid_f[1],
                p_vox_ijk[2] - grid_f[2],
            ];

            let w: [f32; 8] = [
                (1.0 - delta[0]) * (1.0 - delta[1]) * (1.0 - delta[2]),
                (1.0 - delta[0]) * (1.0 - delta[1]) * delta[2],
                (1.0 - delta[0]) * delta[1] * (1.0 - delta[2]),
                (1.0 - delta[0]) * delta[1] * delta[2],
                delta[0] * (1.0 - delta[1]) * (1.0 - delta[2]),
                delta[0] * (1.0 - delta[1]) * delta[2],
                delta[0] * delta[1] * (1.0 - delta[2]),
                delta[0] * delta[1] * delta[2],
            ];

            *value = w.iter().zip(data.iter()).map(|(w_i, data_i)| w_i * data_i).sum()
        }
    });
}



pub fn nearest_neighbor_interpolation_(in_vol: &VolumeViewU8, out_vol: &mut VolumeViewMutU8, out_val: u8) {
    // Get dimensions of input volume
    let in_dims_kji = in_vol.data.shape();
    let in_dims_ijk = Point3::<usize>::new(in_dims_kji[2], in_dims_kji[1], in_dims_kji[0]);

    // Define indices for far left and far right corners
    let far_l = [-1, -1, -1];
    let far_r = [in_dims_ijk.x - 1, in_dims_ijk.y - 1, in_dims_ijk.z - 1];

    let out_geom = out_vol.geom;
    // Perform trilinear interpolation in parallel using Rayon
    Zip::indexed(&mut out_vol.data).par_for_each(|index_kji, value| {
        // Transform output point to LPS space
        let p_lps =
            out_geom.ijk_2_lps([index_kji.2 as f32, index_kji.1 as f32, index_kji.0 as f32]);

        // Transform output point to voxel space of input volume
        let p_vox_ijk = in_vol.geom.lps_2_ijk(p_lps);

        // Compute grid indices and coordinates
        let grid_f = [
            p_vox_ijk[0].floor(),
            p_vox_ijk[1].floor(),
            p_vox_ijk[2].floor(),
        ];
        let grid_i: [i32; 3] = [grid_f[0] as i32, grid_f[1] as i32, grid_f[2] as i32];
        let grid_ui: [usize; 3] = [grid_f[0] as usize, grid_f[1] as usize, grid_f[2] as usize];
        

        // Handle out-of-bounds cases
        if grid_i[0] < far_l[0]
            || grid_i[1] < far_l[1]
            || grid_i[2] < far_l[2]
            || grid_ui[0] > far_r[0]
            || grid_ui[1] > far_r[1]
            || grid_ui[2] > far_r[2]
        {
            *value = out_val;
        } else {
            let data = neighbor_data(grid_i, far_l, grid_ui, far_r, grid_f, in_vol, out_val);
            let delta: [f32; 3] = [
                p_vox_ijk[0] - grid_f[0],
                p_vox_ijk[1] - grid_f[1],
                p_vox_ijk[2] - grid_f[2],
            ];

            let w: [f32; 8] = [
                (1.0 - delta[0]) * (1.0 - delta[1]) * (1.0 - delta[2]),
                (1.0 - delta[0]) * (1.0 - delta[1]) * delta[2],
                (1.0 - delta[0]) * delta[1] * (1.0 - delta[2]),
                (1.0 - delta[0]) * delta[1] * delta[2],
                delta[0] * (1.0 - delta[1]) * (1.0 - delta[2]),
                delta[0] * (1.0 - delta[1]) * delta[2],
                delta[0] * delta[1] * (1.0 - delta[2]),
                delta[0] * delta[1] * delta[2],
            ];
            let max_weight_index = w.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(index, _)| index).unwrap();

            *value = data[max_weight_index];
        }
    });
}

/// Python function wrapper for trilinear_interpolation_ function.
#[pyfunction]
fn trinilear_interpolation(
    in_vol_data: &PyArray3<f32>,
    in_spacing: [f32; 3],
    in_origin_lps: [f32; 3],
    in_matrix_ijk_to_lps: [f32; 9],
    out_vol_data: &PyArray3<f32>,
    out_spacing: [f32; 3],
    out_origin_lps: [f32; 3],
    out_matrix_ijk_to_lps: [f32; 9],
    out_val: f32,
) -> PyResult<()> {
    // Convert PyArray3 to ArrayView3
    let in_vol_data = unsafe { in_vol_data.as_array() };

    // Convert PyArray3 to mutable ArrayViewMut3
    let out_vol_data = unsafe { out_vol_data.as_array_mut() };

    let in_vol = VolumeViewF32::new(in_vol_data, in_origin_lps, in_spacing, in_matrix_ijk_to_lps);

    let mut out_vol = VolumeViewMutF32::new(
        out_vol_data,
        out_origin_lps,
        out_spacing,
        out_matrix_ijk_to_lps,
    );

    trinilear_interpolation_(&in_vol, &mut out_vol, out_val);

    Ok(())
}
/// Python function wrapper for neearest_neighbor_interpolation_ function.
#[pyfunction]
fn neareast_neighbor_interpolation(
    in_vol_data: &PyArray3<u8>,
    in_spacing: [f32; 3],
    in_origin_lps: [f32; 3],
    in_matrix_ijk_to_lps: [f32; 9],
    out_vol_data: &PyArray3<u8>,
    out_spacing: [f32; 3],
    out_origin_lps: [f32; 3],
    out_matrix_ijk_to_lps: [f32; 9],
    out_val: u8,
) -> PyResult<()> {
    // Convert PyArray3 to ArrayView3
    let in_vol_data = unsafe { in_vol_data.as_array() };

    // Convert PyArray3 to mutable ArrayViewMut3
    let out_vol_data = unsafe { out_vol_data.as_array_mut() };

    let in_vol = VolumeViewU8::new(in_vol_data, in_origin_lps, in_spacing, in_matrix_ijk_to_lps);

    let mut out_vol = VolumeViewMutU8::new(
        out_vol_data,
        out_origin_lps,
        out_spacing,
        out_matrix_ijk_to_lps,
    );

    nearest_neighbor_interpolation_(&in_vol, &mut out_vol, out_val);

    Ok(())
}


/// A Python module implemented in Rust.
#[pymodule]
fn ttt_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    // Add trinilear_interpolation function to the module
    m.add_function(wrap_pyfunction!(trinilear_interpolation, m)?)?;
    m.add_function(wrap_pyfunction!(neareast_neighbor_interpolation, m)?)?;

    Ok(())
}
