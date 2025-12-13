# How to generate figure 12

## B=16

* Experiment:

    ```
    cd step_artifact/
    pytest dynamic_par/sweep_ae.py::test_b16_sweep -s
    ```

* Estimated time: 6m46s (lagos), 1m38s (c4)
* Note: The experiments ran in the paper can be found in `batch16_sweep_simple.csv`.
The cycle counts in each row are normalized against the dynamic parallelization cycles.
Then, for each KV cache length variation group, the geometric mean of each parallelization method is calculated. This is the number found for each parallelization method in figure 12.

## B=64

* Experiment:

    ```
    cd step_artifact/
    pytest dynamic_par/sweep_ae.py::test_b64_sweep -s
    ```

* Estimated time: 18m47s
* Note: The experiments ran in the paper can be found in `batch64_sweep_combined.csv`.
The cycle counts in each row are normalized against the dynamic parallelization cycles.
Then, for each KV cache length variation group, the geometric mean of each parallelization method is calculated. This is the number found for each parallelization method in figure 12.

## B=64+16

* Experiment:

    ```
    cd step_artifact/
    pytest dynamic_par/sweep_ae.py::test_b64_b16_sweep -s
    ```

* Estimated time: 25m37s
* Note: The experiments ran in the paper can be found in `batch80_sweep_combined.csv`.
The cycle counts in each row are normalized against the dynamic parallelization cycles.
Then, for each KV cache length variation group, the geometric mean of each parallelization method is calculated. This is the number found for each parallelization method in figure 12.
