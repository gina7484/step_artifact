export PYTHONPATH=$(pwd)/src:$(pwd)/src/step_py:$(pwd)/src/sim:$(pwd)/src/proto
source step-perf/scripts/python_path.sh
# rustup override set 1.83.0
rustup default 1.83.0