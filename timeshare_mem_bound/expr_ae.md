# Figure 8 & 9

## Fig 8(a) & Fig 9(a)(b): Static Tiling

* Command:

    ```
    cd step_artifact
    pytest timeshare_mem_bound/test_membound_qwen_sweep_revet.py::test_static_tile -s
    ```

* Time: small_c4(1m11), full_c4(45m)

* Note:
    The "performance_overhead(%)" column denotes the performance overhead over the baseline (128 parallel regions, 1 expert per region), which is the leftmost datapoint in Figure 8(a).

## Fig 8 (b) Dynamic Tiling

* Command:

    ```
    cd step_artifact
    pytest timeshare_mem_bound/test_membound_qwen_sweep_dyn_tile.py::test_dyn_tile -s
    ```

* Time: full_c4(51m40s)

* Note:
    The "performance_overhead(%)" column denotes the performance overhead over the baseline (128 parallel regions, 1 expert per region), which is the leftmost datapoint in Figure 8(b).
