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

use std::os::raw::{c_char, c_int, c_void};

extern "C" {
    pub fn create_index(
        in_index_type: *const c_char,
        in_parameters: *const c_char,

        out_index_ptr: *mut *const c_void,
    ) -> *const CError;

    pub fn build_index(
        in_index_ptr: *const c_void,
        in_num_vectors: usize,
        in_dim: usize,
        in_ids: *const i64,
        in_vectors: *const f32,

        out_failed_ids: *mut *const i64,
        out_num_failed: *mut usize,
    ) -> *const CError;

    pub fn knn_search_index(
        in_index_ptr: *const c_void,
        in_dim: usize,
        in_query_vector: *const f32,
        in_k: usize,
        in_search_parameters: *const c_char,

        out_ids: *mut *const i64,
        out_distances: *mut *const f32,
        out_num_results: *mut usize,
    ) -> *const CError;

    pub fn dump_index(in_index_ptr: *const c_void, in_file_path: *const c_char) -> *const CError;

    pub fn load_index(
        in_file_path: *const c_char,
        in_index_type: *const c_char,
        in_parameters: *const c_char,

        out_index_ptr: *mut *const c_void,
    ) -> *const CError;

    pub fn free_index(index_ptr: *const c_void);
    pub fn free_error(error: *const CError);
    pub fn free_i64_vector(vector: *const i64);
    pub fn free_f32_vector(vector: *const f32);
}

#[repr(C)]
pub struct CError {
    pub type_: c_int,
    // should be same length with CError defined in wrapper.h
    pub message: [u8; 256],
}

pub fn from_c_error(err: *const CError) -> crate::error::Error {
    let error = crate::error::Error {
        error_type: unsafe { std::mem::transmute::<i32, crate::error::ErrorType>((*err).type_) },
        message: unsafe {
            let null_pos = (*err)
                .message
                .iter()
                .position(|&x| x == 0)
                .unwrap_or((*err).message.len());
            String::from_utf8_lossy(&(*err).message[..null_pos]).into_owned()
        },
    };
    unsafe {
        free_error(err);
    }
    error
}

pub fn from_c_i64_vector(vector: *const i64, len: usize) -> Vec<i64> {
    let slice = unsafe { std::slice::from_raw_parts(vector, len) };
    let vec = slice.to_vec();
    unsafe {
        free_i64_vector(vector);
    }
    vec
}

pub fn from_c_f32_vector(vector: *const f32, len: usize) -> Vec<f32> {
    let slice = unsafe { std::slice::from_raw_parts(vector, len) };
    let vec = slice.to_vec();
    unsafe {
        free_f32_vector(vector);
    }
    vec
}

pub fn to_c_string(s: &str) -> std::ffi::CString {
    std::ffi::CString::new(s).expect("0 byte in string")
}
