# EvE

A four-member, steady-state differential-evolution optimizer with a gradient (Adam) fallback, implemented as a `torch.optim.Optimizer` subclass.

This is the reference implementation of the paper **"EvE: An Alternate Optimizer to Adam"**, currently under review at ICLR 2027.

```python
from eve import EvE

opt = EvE(model.parameters(), lr=1e-3)
for step in range(n_steps):
    def closure():
        return loss_fn(model(x), y)
    loss = opt.step(closure)
```

`step` takes a closure that returns the loss for the model's current weights, as `torch.optim.LBFGS` does. The closure must not call `.backward()`; EvE does that itself, and only inside the fallback. `step` returns the lowest loss in the population, and leaves the model holding that member's weights. EvE is not a drop-in replacement for `Adam.step()`, because of the closure and the extra forward passes per step.

## Algorithm

EvE keeps a population of four candidates. Each candidate is a full copy of the parameters, stored normalized to [0, 1] between per-coordinate bounds.

Each call to `step` handles one candidate (the target slot), in round-robin order:

1. The target's loss is re-evaluated on the current batch.
2. A child is built with DE `rand-to-pbest/1`: a randomly chosen member, plus `F` times the difference of two others, plus `F` times its offset from the current best.
3. The child is evaluated. If its loss is lower than the target's, it is accepted.
4. If it is not, the child is refined by `n_adam_steps` steps of Adam. Each target slot has its own persistent Adam state. The refined point is kept if it beats the child.
5. Greedy selection: the target slot is replaced only if the new point has a lower loss.

Cost per step: two forward passes (target and child), or three with `use_pm`. When step 3 fails, add `n_adam_steps` forward-and-backward passes and one more forward. Backward passes happen only in that case. The first call also scores all four initial members, so it costs about five forward passes.

Memory: four population copies, the bounds, and four Adam optimizers' state. Expect several times the memory of plain Adam.

## Constructor options

| argument | default | meaning |
|---|---|---|
| `lr` | `0.05` | learning rate of the built-in fallback Adam |
| `weight_decay` | `1e-4` | weight decay of the built-in fallback Adam |
| `n_adam_steps` | `3` | Adam steps per fallback burst |
| `F` | `0.5` | DE scale factor |
| `sigma` | `0.1` | spread of the initial population around its anchor, in normalized units |
| `use_pm` | `False` | apply polynomial mutation (index `pm_eta`, default 20) to the child |
| `sampling` | `"proximity"` | initial population: `"proximity"` (random anchor, three Gaussian neighbours), `"lhs"` (Latin hypercube), or a callable `(n_pop, n_var, generator) -> tensor` |
| `init_from_model` | `False` | anchor the population at the model's current weights. With this set, `sampling` is ignored |
| `bounds` | Kaiming | explicit `(lower_list, upper_list)` tensors, one per parameter |
| `seed` | `None` | seeds all internal randomness |
| `device` | model's | device for the sampling generator |
| `fallback_factory` | built-in Adam | `lambda params: optimizer`, to replace the fallback optimizer |

Properties after training: `opt.n_fallback` (number of fallback bursts so far) and `opt.best_loss`.

## Things to know

**Bounds.** By default each tensor is searched in `[-b, b]` with `b = sqrt(6 / fan_in)`. One-dimensional tensors (biases, normalization gains) use the fan-in of the preceding matrix. Candidates are clamped to these bounds, including after Adam steps. This is wrong for parameters whose natural scale is larger, such as a LayerNorm gain initialised to 1.0, which the default would clamp to a few hundredths. Pass `bounds=` for such models.

**One batch per step.** To compare the target and the child fairly, they must be evaluated on the same data. If your closure draws a new batch on every call, the comparison is made on different batches and is noisy. `step(closure, new_batch=fn)` calls `fn()` once at the start of the step; make the closure read a batch cached by that function, so every evaluation in the step, including every Adam step of a burst, sees the same batch. A side effect is that data loading is paid once per step instead of once per evaluation.

**Cost accounting.** If the closure has an attribute `charge_grad`, EvE calls `closure.charge_grad(seconds)` after each fallback backward pass with its synchronized duration. This lets a caller count backward passes in the same budget as forward passes.

**Checkpointing.** `state_dict()` and `load_state_dict()` cover only the standard `torch.optim.Optimizer` bookkeeping. The population, its fitness values and each slot's Adam state are not saved, so resuming through the usual PyTorch pattern restarts with a fresh population.

## Installation

```bash
pip install -e .
```

The only dependency is `torch` (2.0 or later).

## Self-test

```bash
python -m eve.optimizer
```

runs short checks of the default optimizer, `init_from_model`, polynomial mutation, a custom sampling function, and a custom fallback optimizer.
