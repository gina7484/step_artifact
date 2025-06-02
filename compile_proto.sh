cd step_perf_ir/proto
protoc --python_out=../../src/proto/ graph.proto ops.proto datatype.proto func.proto
cd ../../

cd step-perf
cargo build
maturin develop
cd ../