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
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-cdylib-link-arg=-fPIC");

    let dst = cmake::Config::new("vsag-sys")
        // Cargo sets TARGET to the target triple
        // but building openblas via cmake will fail if it's set
        // ```plaintext
        // The TARGET specified on the command line or in Makefile.rule is not supported. Please choose a target from TargetList.txt
        // ```
        .env("TARGET", "")
        .build();

    println!("cargo:rustc-link-lib=dylib=vsag");
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
}
