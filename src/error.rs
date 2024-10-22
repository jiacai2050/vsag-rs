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

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub struct Error {
    pub error_type: ErrorType,
    pub message: String,
}

#[derive(Debug)]
#[repr(C)]
pub enum ErrorType {
    // [common errors]
    /// unknown error
    UnknownError = 1,
    /// some internal errors occupied in algorithm
    InternalError,
    /// invalid argument
    InvalidArgument,

    // [behavior errors]
    /// index has been build, cannot build again
    BuildTwice,
    /// index object is NOT empty so that should not deserialize on it
    IndexNotEmpty,
    /// trying to create an unsupported index
    UnsupportedIndex,
    /// the index does not support this function
    UnsupportedIndexOperation,
    /// the dimension of add/build/search request is NOT equal to index
    DimensionNotEqual,
    /// index is empty, cannot search or serialize
    IndexEmpty,

    // [runtime errors]
    /// failed to alloc memory
    NoEnoughMemory,
    /// cannot read from binary
    ReadError,
    /// some file missing in index diskann deserialization
    MissingFile,
    /// the content of binary is invalid
    InvalidBinary,
}
