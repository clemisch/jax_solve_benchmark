Benchmark 100000 batched 3x3 solves on CPU with JAX using float32 over many jax,jaxlib versions.

See results_clean.txt: 

* 0.4.31 -> 0.4.33: x2 runtime increase, probably from [switch to thunked runtime on CPU](https://github.com/jax-ml/jax/issues/36799)
* 0.9.2 -> 0.10.0: x2 runtime increase, probably from [solve-specific batching changes](https://github.com/jax-ml/jax/issues/36927)

---

Run on Intel i7-8550U, Fedora 42, and

```python
>>> jax.print_environment_info()
jax:    ...
jaxlib: ...
numpy:  2.4.4
python: 3.12.13 (main, Apr 16 2026, 00:00:00) [GCC 15.2.1 20260123 (Red Hat 15.2.1-7)]
jax.devices (1 total, 1 local): [CpuDevice(id=0)]
process_count: 1
platform: uname_result(system='Linux', node='xxx', release='6.19.13-100.fc42.x86_64', version='#1 SMP PREEMPT_DYNAMIC Sat Apr 18 21:32:46 UTC 2026', machine='x86_64')
```