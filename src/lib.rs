// Copyright 2023 Greptime Team
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

pub mod error;
mod ffi;

use std::os::raw::c_void;

use ffi::dump_index;

use crate::error::Result;
use crate::ffi::{
    build_index, create_index, free_index, from_c_error, from_c_f32_vector, from_c_i64_vector,
    knn_search_index, to_c_string,
};

/// `VsagIndex` is a wrapper around the C++ index object.
///
/// When the `VsagIndex` is dropped, the C++ index object is freed.
pub struct VsagIndex {
    /// Pointer to the C++ index object.
    ptr: *const c_void,
}

/// The index in c doesn't contains any thread-locals variables, so it's sendable.
unsafe impl Send for VsagIndex {}

impl VsagIndex {
    /// Creates a new vsag index.
    ///
    /// `index_type` is the type of index to create. Currently supported values are:
    /// - `hnsw`
    /// - `diskann`
    ///
    /// HNSW.params in JSON format:
    ///    - dtype: string, required, one of [float32]
    ///    - metric_type: string, required, one of [l2, ip]
    ///    - dim: integer, required
    ///    - hnsw.max_degree: integer, required
    ///    - hnsw.ef_construction: integer, required
    ///      e.g.,
    ///      {
    ///         "dtype": "float32",
    ///         "metric_type": "l2",
    ///         "dim": 128,
    ///         "hnsw": {
    ///             "max_degree": 16,
    ///             "ef_construction": 200
    ///         }
    ///      }
    ///
    ///  DiskANN.params in JSON format:
    ///    - dtype: string, required, one of [float32]
    ///    - metric_type: string, required, one of [l2, ip]
    ///    - dim: integer, required
    ///    - diskann.max_degree: integer, required
    ///    - diskann.ef_construction: integer, required
    ///    - diskann.pq_dims: integer, required
    ///    - diskann.pq_sample_rate: floating number, required, in range (0.0, 1.0]
    ///      e.g.,
    ///      {
    ///         "dtype": "float32",
    ///         "metric_type": "l2",
    ///         "dim": 128,
    ///         "diskann": {
    ///             "max_degree": 16,
    ///             "ef_construction": 200,
    ///             "pq_dims": 64,
    ///             "pq_sample_rate": 0.5
    ///         }
    ///      }
    pub fn new(index_type: &str, params: &str) -> Result<Self> {
        let index_type_c = to_c_string(index_type);
        let parameters_c = to_c_string(params);

        unsafe {
            let out_index_ptr = &mut std::ptr::null();
            let err = create_index(index_type_c.as_ptr(), parameters_c.as_ptr(), out_index_ptr);

            if !err.is_null() {
                Err(from_c_error(err))
            } else {
                Ok(VsagIndex {
                    ptr: *out_index_ptr,
                })
            }
        }
    }

    /// Builds index with all vectors
    ///
    /// All vectors are passed as a single slice of f32. If you have `num_vectors` vectors of dimension `dim`,
    /// you should pass a `vectors` slice of length `num_vectors * dim` and `ids` slice of length `num_vectors`.
    ///
    /// Returns IDs of vectors that failed to be added to the index.
    pub fn build(
        &self,
        num_vectors: usize,
        dim: usize,
        ids: &[i64],
        vectors: &[f32],
    ) -> Result<Vec<i64>> {
        unsafe {
            let out_failed_ids: *mut *const i64 = &mut std::ptr::null();
            let out_num_failed: *mut usize = &mut 0;
            let err = build_index(
                self.ptr,
                num_vectors,
                dim,
                ids.as_ptr(),
                vectors.as_ptr(),
                out_failed_ids,
                out_num_failed,
            );

            if !err.is_null() {
                Err(from_c_error(err))
            } else {
                Ok(from_c_i64_vector(*out_failed_ids, *out_num_failed))
            }
        }
    }

    /// Searches for the `k` nearest neighbors of the `query_vector`.
    ///
    /// `search_params` is a JSON string that specifies the search parameters.
    ///
    /// HNSW.search_params in JSON format:
    ///   - hnsw.ef_search: integer, required
    ///   - hnsw.use_conjugate_graph_search: boolean, optional, default is true
    ///     e.g.,
    ///     {
    ///        "hnsw": {
    ///           "ef_search": 100,
    ///          "use_conjugate_graph_search": true
    ///       }
    ///     }
    ///
    /// DiskANN.search_params in JSON format:
    ///  - diskann.ef_search: integer, required
    ///  - diskann.beam_search: integer, required
    ///  - diskann.io_limit: integer, required
    ///  - diskann.use_reorder: boolean, optional, default is false
    ///     e.g.,
    ///     {
    ///       "diskann": {
    ///        "ef_search": 100,
    ///        "beam_search": 4,
    ///        "io_limit": 200,
    ///        "use_reorder": false
    ///       }
    ///     }
    pub fn knn_search(
        &self,
        query_vector: &[f32],
        k: usize,
        search_params: &str,
    ) -> Result<KnnSearchOutput> {
        let search_params = to_c_string(search_params);

        unsafe {
            let out_ids: *mut *const i64 = &mut std::ptr::null();
            let out_distances: *mut *const f32 = &mut std::ptr::null();
            let out_num_results: *mut usize = &mut 0;
            let err = knn_search_index(
                self.ptr,
                query_vector.len(),
                query_vector.as_ptr(),
                k,
                search_params.as_ptr(),
                out_ids,
                out_distances,
                out_num_results,
            );

            if !err.is_null() {
                Err(from_c_error(err))
            } else {
                Ok(KnnSearchOutput {
                    ids: from_c_i64_vector(*out_ids, *out_num_results),
                    distances: from_c_f32_vector(*out_distances, *out_num_results),
                })
            }
        }
    }

    /// Dumps the index to the file at `path`.
    pub fn dump(self, path: &str) -> Result<()> {
        let path = to_c_string(path);

        unsafe {
            let err = dump_index(self.ptr, path.as_ptr());
            if !err.is_null() {
                Err(from_c_error(err))
            } else {
                Ok(())
            }
        }
    }

    /// Loads an index from the file at `path`.
    ///
    /// `index_type` and `params` should be the same as the ones used to create the index.
    pub fn load(path: &str, index_type: &str, params: &str) -> Result<Self> {
        let path = to_c_string(path);
        let index_type = to_c_string(index_type);
        let params = to_c_string(params);

        unsafe {
            let out_index_ptr: *mut *const c_void = &mut std::ptr::null();
            let err = ffi::load_index(
                path.as_ptr(),
                index_type.as_ptr(),
                params.as_ptr(),
                out_index_ptr,
            );

            if !err.is_null() {
                Err(from_c_error(err))
            } else {
                Ok(VsagIndex {
                    ptr: *out_index_ptr,
                })
            }
        }
    }
}

impl Drop for VsagIndex {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                free_index(self.ptr);
            }
        }
    }
}

/// Output of a k-NN search.
pub struct KnnSearchOutput {
    /// IDs of the k-NNs.
    pub ids: Vec<i64>,
    /// Distances of the k-NNs.
    pub distances: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use simsimd::SpatialSimilarity;

    use super::*;

    #[test]
    fn test_create_build_search_index_hnsw_l2() {
        let index_type = "hnsw";
        let con_params = r#"{
            "dtype": "float32",
            "metric_type": "l2",
            "dim": 128,
            "hnsw": {
                "max_degree": 16,
                "ef_construction": 100
            }
        }"#;
        let search_params = r#"{
          "hnsw": {
            "ef_search": 100
          }
        }"#;

        let index = VsagIndex::new(index_type, con_params).unwrap();

        let num_vectors: usize = 1000;
        let dim: usize = 128;

        let ids: Vec<i64> = (0..num_vectors as i64).collect();
        let vectors = (0..num_vectors)
            .map(|_| {
                (0..dim)
                    .map(|_| rand::random::<f32>())
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<_>>();
        let vectors_for_index: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();

        let failed_ids = index
            .build(num_vectors, dim, &ids, &vectors_for_index)
            .unwrap();
        assert_eq!(failed_ids.len(), 0);

        let query_vector: Vec<f32> = (0..dim).map(|_| rand::random()).collect();
        let k = 10;
        let output = index.knn_search(&query_vector, k, search_params).unwrap();
        assert_eq!(output.ids.len(), k.min(num_vectors));
        assert_eq!(output.distances.len(), k.min(num_vectors));

        let mut distances = vectors
            .iter()
            .zip(ids.iter())
            .map(|(v, id)| {
                let d: f32 = f32::l2sq(&query_vector, &v).unwrap() as _;
                (d, *id)
            })
            .collect::<Vec<_>>();
        distances.sort_by(|(a, _), (b, _)| a.total_cmp(b));
        distances.truncate(k.min(num_vectors));

        // dump
        let dir = tempdir::TempDir::new("test_create_build_search_index_l2_").unwrap();
        let path = dir.path().join("index");
        index.dump(path.to_str().unwrap()).unwrap();

        // load
        let index = VsagIndex::load(path.to_str().unwrap(), index_type, con_params).unwrap();
        let output2 = index.knn_search(&query_vector, k, search_params).unwrap();
        assert_eq!(output.ids, output2.ids);
        assert_eq!(output.distances, output2.distances);
    }

    #[test]
    fn test_create_build_search_index_cos() {
        let index_type = "hnsw";
        let con_params = r#"{
            "dtype": "float32",
            "metric_type": "cosine",
            "dim": 128,
            "hnsw": {
                "max_degree": 16,
                "ef_construction": 100
            }
        }"#;
        let search_params = r#"{
            "hnsw": {
              "ef_search": 100
            }
          }"#;
        let index = VsagIndex::new(index_type, con_params).unwrap();

        let num_vectors: usize = 1000;
        let dim: usize = 128;

        let ids: Vec<i64> = (0..num_vectors as i64).collect();
        let vectors = (0..num_vectors)
            .map(|_| {
                (0..dim)
                    .map(|_| rand::random::<f32>())
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<_>>();
        let vectors_for_index: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();

        let failed_ids = index
            .build(num_vectors, dim, &ids, &vectors_for_index)
            .unwrap();
        assert_eq!(failed_ids.len(), 0);

        let query_vector: Vec<f32> = (0..dim).map(|_| rand::random()).collect();
        let k = 10;

        let output = index.knn_search(&query_vector, k, search_params).unwrap();
        assert_eq!(output.ids.len(), k.min(num_vectors));
        assert_eq!(output.distances.len(), k.min(num_vectors));

        let mut distances = vectors
            .iter()
            .zip(ids.iter())
            .map(|(v, id)| {
                let d: f32 = <f32 as SpatialSimilarity>::cos(&query_vector, &v).unwrap() as _;
                (d, *id)
            })
            .collect::<Vec<_>>();
        distances.sort_by(|(a, _), (b, _)| a.total_cmp(b));
        distances.truncate(k.min(num_vectors));

        // dump
        let dir = tempdir::TempDir::new("test_create_build_search_index_cos").unwrap();
        let path = dir.path().join("index");
        index.dump(path.to_str().unwrap()).unwrap();

        // load
        let index = VsagIndex::load(path.to_str().unwrap(), index_type, con_params).unwrap();
        let output2 = index.knn_search(&query_vector, k, search_params).unwrap();
        assert_eq!(output.ids, output2.ids);
        assert_eq!(output.distances, output2.distances);
    }
}
