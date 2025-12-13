cd step_artifact
# figure 6
pytest dyn_tiling/test_mixtral_sweep.py::test_mixtral_b64 -s
pytest dyn_tiling/test_qwen_sweep.py::test_qwen_b64 -s

# figure 7
pytest dyn_tiling/test_mixtral_sweep_prefill.py::test_mixtral_b1024 -s
pytest dyn_tiling/test_qwen_sweep_prefill.py::test_qwen_b1024 -s