Benchmark 100000 batched 3x3 solves on CPU with JAX using float32 over many jax,jaxlib versions.

See results_clean.txt: 

* 0.4.31 -> 0.4.33: x2 runtime increase, probably from [switch to thunked runtime on CPU](https://github.com/jax-ml/jax/issues/36799)
* 0.9.2 -> 0.10.0: x2 runtime increase, probably from [solve-specific batching changes](https://github.com/jax-ml/jax/issues/36927)
