# Kernel Recipes

Reusable patterns go here only after a local kernel proves them useful.

## Tiled Elementwise

Use when each element is loaded once and stored once. The roofline target is
global-memory bandwidth.

Triton shape:

```python
pid = tl.program_id(0)
offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offs < n
x = tl.load(x_ptr + offs, mask=mask)
tl.store(o_ptr + offs, f(x), mask=mask)
```

Sweep `BLOCK_SIZE`, then `num_warps` if the body has enough work to matter.
