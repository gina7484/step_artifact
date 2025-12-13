# Figure 6

## Mixtral8x7B

* Command:

    ```
    cd step_artifact
    pytest dyn_tiling/test_mixtral_sweep.py::test_mixtral_b64 -s
    ```

* Produces:
    `./dyn_tiling/figure_6_mixtral_b64.csv`

* Time to run: Small_c4(32s), full_c4(30m)

## Qwen3-30B

* Command:

    ```
    cd step_artifact
    pytest dyn_tiling/test_qwen_sweep.py::test_qwen_b64 -s
    ```

* Produces:
    `./dyn_tiling/figure_6_qwen_b64.csv`

* Time to run: small_c4 (46s), full_c4(45m25s)

# Figure 7

## Mixtral8x7B

* Command:

    ```
    cd step_artifact
    pytest dyn_tiling/test_mixtral_sweep_prefill.py::test_mixtral_b1024 -s
    ```

* Produces:
    `./dyn_tiling/figure_7_mixtral_b1024.csv`

* Time to run: Small_c4(1m41)

## Qwen3-30B

* Command:

    ```
    cd step_artifact
    pytest dyn_tiling/test_qwen_sweep_prefill.py::test_qwen_b1024 -s
    ```

* Produces:
    `./dyn_tiling/figure_7_qwen_b1024.csv`
* Time to run:
