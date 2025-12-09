export PYTHONPATH=$(pwd)/src:$(pwd)/src/step_py:$(pwd)/src/sim:$(pwd)/src/proto
# source step-perf/scripts/python_path.sh
# rustup override set 1.83.0
rustup default 1.83.0

cd step_perf_ir/proto
mkdir -p ../../src/proto
protoc --python_out=../../src/proto/ graph.proto ops.proto datatype.proto func.proto
# You might need to add the --experimental_allow_proto3_optional flag if the command above fails to generate .proto files under --python_out
# protoc --experimental_allow_proto3_optional --python_out=../../src/proto/ graph.proto ops.proto datatype.proto func.proto
cd ../../

cd step-perf
cargo build
maturin develop
cd ../