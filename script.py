import jax
import jax.numpy as jnp

print(f"jax={jax.__version__}")
print(f"jaxlib={jax.lib.__version__}")

batch = 100_000
key_A = jax.random.key(0)
key_b = jax.random.key(1)

A = jax.random.normal(key_A, (batch, 3, 3), dtype=jnp.float32)
A = A @ jnp.swapaxes(A, -1, -2) + 1e-2 * jnp.eye(3, dtype=jnp.float32)
b = jax.random.normal(key_b, (batch, 3, 1), dtype=jnp.float32)

solve = jax.jit(jnp.linalg.solve)
grad = jax.jit(
    jax.grad(lambda A, b: jnp.sum(jnp.linalg.solve(A, b)), argnums=(0, 1))
)

jax.block_until_ready(solve(A, b))
jax.block_until_ready(grad(A, b))

%timeit jax.block_until_ready(solve(A, b))
%timeit jax.block_until_ready(grad(A, b))
