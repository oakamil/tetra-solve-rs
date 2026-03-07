#!/bin/bash
source ../cedar-solve/.cedar_venv/bin/activate
cargo test --release --test tetra3_server_py_test -- --nocapture
