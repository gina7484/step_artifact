Done

* Mixtral 64 (mixtral_b64)
  * static (=revet)
  * dynamic
* Qwen 64 (qwen_b64)
  * static
  * revet
  * dynamic
* Qwen 1024 (qwen_b1024)
  * static (But this is ran with the old data)

Running

TODO

* Qwen 1024
  * static: Need to rerun with the new data
  * revet: `pytest dyn_tiling/test_qwen_sweep_prefill.py::test_gemm_revet_sweep -s`
  * dynamic: `pytest dyn_tiling/test_qwen_sweep_prefill.py::test_gemm_dyn_tile -s`
* Mixtral 1024: `pytest dyn_tiling/test_mixtral_sweep_prefill.py`
  * static (=revet)
  * dynamic

todo

1. test prefill with small mixtral
2. test b64 qwen with small qwen
3. test b1024 with small qwen
4. test how long the print statement is if you run w/o -s
5. Run the script with time (w/o -s)
