# VSAG Rust Binding

[![Crates.io](https://img.shields.io/crates/v/vsag.svg)](https://crates.io/crates/vsag)
[![docs.rs](https://img.shields.io/docsrs/vsag/latest)](https://docs.rs/vsag)
![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)
[![CI](https://github.com/jiacai2050/vsag-rs/actions/workflows/CI.yml/badge.svg)](https://github.com/jiacai2050/vsag-rs/actions/workflows/CI.yml)

A Rust binding for the [VSAG](https://github.com/alipay/vsag).

## Usage

```bash
cargo add vsag
```

Then try the example:

```rust
use vsag_sys::VsagIndex;

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
```
