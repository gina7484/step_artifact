source setup.sh
pytest dyn_tiling/test_qwen_sweep_prefill.py::test_gemm_dyn_tile -s
pytest dyn_tiling/test_qwen_sweep_prefill.py::test_gemm_revet_sweep -s
