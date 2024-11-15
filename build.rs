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

fn main() {
    println!("cargo:rerun-if-env-changed=VSAG_LIB_PATH");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-link-lib=dylib=vsag");

    if let Some(lib_path) = vsag_lib_path() {
        println!("cargo:rustc-link-search=native={lib_path}",);
    }
}

/// `some-feature` becomes `SOME_FEATURE` options in cmake.
#[cfg(feature = "vendored")]
macro_rules! define_config_based_on_features {
    ($config:ident, $($feature:expr),*) => {
        $(
            $config.define(
                $feature.to_uppercase().replace("-", "_"),
                if cfg!(feature = $feature) {
                    "ON"
                } else {
                    "OFF"
                },
            );
        )*
    };
}

fn vsag_lib_path() -> Option<String> {
    #[cfg(feature = "vendored")]
    {
        let mut config = cmake::Config::new("vsag-sys");

        // Cargo sets TARGET to the target triple
        // but building openblas via cmake will fail if it's set
        // ```plaintext
        // The TARGET specified on the command line or in Makefile.rule is not supported. Please choose a target from TargetList.txt
        // ```
        config.env("TARGET", "");

        define_config_based_on_features!(
            config,
            "enable-intel-mkl",
            "enable-libcxx",
            "enable-cxx11-abi"
        );

        let dst = config.build();

        // centos use `lib64`, ubuntu use `lib` convention.
        for path in ["lib64", "lib"] {
            let lib = dst.join(path);
            if lib.join("libvsag.so").exists() {
                return Some(lib.display().to_string());
            }
        }
    }

    std::env::var("VSAG_LIB_PATH").map_or(None, |v| Some(v.to_string()))
}
