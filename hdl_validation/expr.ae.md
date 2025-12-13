# Figure 5

* Command:

    ```
    cd step_artifact
    mv step-perf/src/memory/mod.rs step-perf/src/memory/bw_64_mod.rs
    mv step-perf/src/memory/hdl_mod.rs step-perf/src/memory/mod.rs
    source setup.sh
    pytest hdl_validation/expert_tiling_sweep.py::test_expert_tiling_sweep_single_schedule -s

    mv step-perf/src/memory/mod.rs step-perf/src/memory/hdl_mod.rs
    mv step-perf/src/memory/bw_64_mod.rs step-perf/src/memory/mod.rs
    ```

* Time: 30s (c4)

* Results will be saved in `step_artifact/hdl_validation/fig5.csv`.
