import math
import time
from collections.abc import Callable

import torch
from torch.nn.init import _calculate_fan_in_and_fan_out
from torch.optim import Optimizer

__all__ = ["EvE"]


SamplingFn = Callable[[int, int, "torch.Generator | None"], torch.Tensor]


def _kaiming_bound(p: torch.Tensor, prev_fan_in: int | None) -> tuple[float, int | None]:
    if p.dim() >= 2:
        fan_in, _ = _calculate_fan_in_and_fan_out(p)
    else:
        fan_in = prev_fan_in if prev_fan_in is not None else p.numel()
    return (6.0 / fan_in) ** 0.5, (fan_in if p.dim() >= 2 else prev_fan_in)


def _latin_hypercube(n_samples: int, n_var: int, generator: torch.Generator | None) -> torch.Tensor:
    device = generator.device if generator is not None else "cpu"
    u = torch.rand(n_samples, n_var, generator=generator, device=device)
    strata = torch.stack([torch.randperm(n_samples, generator=generator, device=device) for _ in range(n_var)], dim=1)
    return (strata + u) / n_samples


def _polynomial_mutation(x: list[torch.Tensor], eta: float, generator: torch.Generator | None) -> list[torch.Tensor]:
    sizes = [t.numel() for t in x]
    n_var = sum(sizes)
    flat_idx = int(torch.randint(0, n_var, (), generator=generator).item())
    which = next(i for i, s in enumerate(torch.cumsum(torch.tensor(sizes), 0)) if flat_idx < s)
    local_idx = flat_idx - (sum(sizes[:which]))

    out = [t.clone() for t in x]
    flat = out[which].reshape(-1)
    val = flat[local_idx].item()
    r = torch.rand((), generator=generator).item()
    d1, d2 = val, 1.0 - val
    if r <= 0.5:
        dq = (2 * r + (1 - 2 * r) * (1 - d1) ** (eta + 1)) ** (1 / (eta + 1)) - 1
    else:
        dq = 1 - (2 * (1 - r) + 2 * (r - 0.5) * (1 - d2) ** (eta + 1)) ** (1 / (eta + 1))
    flat[local_idx] = min(1.0, max(0.0, val + dq))
    return out


class EvE(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.05,
        weight_decay: float = 1e-4,
        n_adam_steps: int = 3,
        F: float = 0.5,
        sigma: float = 0.1,
        use_pm: bool = False,
        pm_eta: float = 20.0,
        sampling: str | SamplingFn = "proximity",
        init_from_model: bool = False,
        bounds: tuple[list[torch.Tensor], list[torch.Tensor]] | None = None,
        seed: int | None = None,
        device: torch.device | str | None = None,
        fallback_factory: Callable[[list[torch.Tensor]], Optimizer] | None = None,
    ):
        params = list(params)
        if not params:
            raise ValueError("EvE needs at least one parameter")
        super().__init__(params, defaults=dict(lr=lr))
        if isinstance(sampling, str) and sampling not in ("proximity", "lhs"):
            raise ValueError(f"sampling must be 'proximity', 'lhs', or a callable, got {sampling!r}")

        self.n_pop = 4
        self.F, self.n_adam_steps = F, n_adam_steps
        self.use_pm, self.pm_eta = use_pm, pm_eta

        self._params = [p for g in self.param_groups for p in g["params"]]
        device = torch.device(device) if device is not None else self._params[0].device
        self._sizes = [p.numel() for p in self._params]
        self.n_var = sum(self._sizes)

        if bounds is not None:
            self._xl, self._xu = [l.clone() for l in bounds[0]], [u.clone() for u in bounds[1]]
        else:
            bound, prev_fan_in = [], None
            for p in self._params:
                b, prev_fan_in = _kaiming_bound(p, prev_fan_in)
                bound.append(b)
            self._xu = [torch.full_like(p, b) for p, b in zip(self._params, bound)]
            self._xl = [-u for u in self._xu]
        self._range = torch._foreach_sub(self._xu, self._xl)
        if any(torch.any(r == 0) for r in self._range):
            raise ValueError(
                "EvE got a zero-width search range (xl == xu for at least one coordinate) -- "
                "every parameter needs xu > xl, since candidates are stored normalized by "
                "(x - xl) / (xu - xl). Check your explicit bounds= (a common cause: an "
                "all-zero param combined with init_from_model and matching zero bounds)."
            )

        gen = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
        self._pop = self._init_population(sampling, sigma, init_from_model, gen)


        if fallback_factory is None:
            fallback_factory = lambda params: torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, foreach=True)
        self._fallback_opts = [fallback_factory(self._params) for _ in range(self.n_pop)]


        self._cpu_gen = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None
        self._fitness: list[float] | None = None
        self._slot = 0
        self.n_fallback = 0
        self.best_loss = math.inf

    def _init_population(self, sampling, sigma, init_from_model, gen) -> list[list[torch.Tensor]]:
        if init_from_model:
            anchor = torch._foreach_div(torch._foreach_sub([p.detach() for p in self._params], self._xl), self._range)
            torch._foreach_clamp_min_(anchor, 0.0)
            torch._foreach_clamp_max_(anchor, 1.0)
        elif sampling == "lhs" or callable(sampling):
            sample_fn = _latin_hypercube if sampling == "lhs" else sampling
            flat = sample_fn(self.n_pop, self.n_var, gen)
            return [[c.reshape(p.shape) for c, p in zip(torch.split(row, self._sizes), self._params)] for row in flat]
        else:
            anchor = [torch.rand_like(p, generator=gen) for p in self._params]

        pop = [anchor]
        for _ in range(self.n_pop - 1):
            jitter = [torch.randn_like(p, generator=gen) * sigma for p in self._params]
            cand = torch._foreach_add(anchor, jitter)
            torch._foreach_clamp_min_(cand, 0.0)
            torch._foreach_clamp_max_(cand, 1.0)
            pop.append(cand)
        return pop


    def _load(self, candidate: list[torch.Tensor]) -> None:
        actual = torch._foreach_add(self._xl, torch._foreach_mul(candidate, self._range))
        with torch.no_grad():
            torch._foreach_copy_(self._params, actual)

    def _eval(self, candidate: list[torch.Tensor], closure) -> float:
        self._load(candidate)
        with torch.no_grad():
            return float(closure())


    @torch.no_grad()
    def step(self, closure, new_batch=None) -> float:
        if new_batch is not None:
            new_batch()
        if self._fitness is None:
            self._fitness = [self._eval(c, closure) for c in self._pop]

        i = self._slot
        self._slot = (self._slot + 1) % self.n_pop
        others = [j for j in range(self.n_pop) if j != i]
        a, b, c = (others[k] for k in torch.randperm(3, generator=self._cpu_gen).tolist())
        best_j = min(range(self.n_pop), key=self._fitness.__getitem__)


        jitter = self.F * (1 + 1e-4 * (torch.rand((), generator=self._cpu_gen).item() - 0.5))
        diff = torch._foreach_sub(self._pop[b], self._pop[c])
        child = torch._foreach_add(self._pop[a], torch._foreach_mul(diff, jitter))
        pull = torch._foreach_mul(torch._foreach_sub(self._pop[best_j], self._pop[a]), self.F)
        torch._foreach_add_(child, pull)
        torch._foreach_clamp_min_(child, 0.0)
        torch._foreach_clamp_max_(child, 1.0)


        f_current = self._eval(self._pop[i], closure)
        self._fitness[i] = f_current

        if self.use_pm:
            f_child_clean = self._eval(child, closure)
            mutant = _polynomial_mutation(child, self.pm_eta, self._cpu_gen)
            f_mutant = self._eval(mutant, closure)
            child, f_child = (mutant, f_mutant) if f_mutant < f_child_clean else (child, f_child_clean)
        else:
            f_child = self._eval(child, closure)

        if f_child < f_current:
            best_cand, f_best = child, f_child
        else:
            self.n_fallback += 1
            self._load(child)
            opt = self._fallback_opts[i]


            charge_grad = getattr(closure, "charge_grad", None)
            is_cuda = self._params[0].is_cuda
            for _ in range(self.n_adam_steps):
                opt.zero_grad()
                with torch.enable_grad():
                    loss = closure()
                    if charge_grad is not None:
                        if is_cuda:
                            torch.cuda.synchronize()
                        t_bwd = time.perf_counter()
                    loss.backward()
                    if charge_grad is not None:
                        if is_cuda:
                            torch.cuda.synchronize()
                        charge_grad(time.perf_counter() - t_bwd)
                opt.step()
                with torch.no_grad():
                    torch._foreach_clamp_min_(self._params, self._xl)
                    torch._foreach_clamp_max_(self._params, self._xu)
            refined = [p.detach().clone() for p in self._params]
            f_refined = float(closure())
            norm_refined = torch._foreach_div(torch._foreach_sub(refined, self._xl), self._range)
            best_cand, f_best = (norm_refined, f_refined) if f_refined < f_child else (child, f_child)

        if f_best < self._fitness[i]:
            self._pop[i], self._fitness[i] = best_cand, f_best

        best_j = min(range(self.n_pop), key=self._fitness.__getitem__)
        self._load(self._pop[best_j])
        self.best_loss = self._fitness[best_j]
        return self.best_loss


if __name__ == "__main__":
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1))
    X = torch.randn(32, 4)
    y = (X @ torch.tensor([1.0, -2.0, 0.5, 0.3])).unsqueeze(1)

    def closure():
        return torch.nn.functional.mse_loss(model(X), y)

    opt = EvE(model.parameters(), lr=0.05, seed=0)
    losses = [opt.step(closure) for _ in range(300)]
    assert losses[-1] < losses[0], f"EvE should reduce loss: {losses[0]:.4f} -> {losses[-1]:.4f}"
    assert losses[-1] < 1.0, f"expected a reasonably fit model, got final loss {losses[-1]:.4f}"
    print("EvE optimizer ok -- loss:", losses[0], "->", losses[-1])


    torch.manual_seed(1)
    model2 = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1))
    with torch.no_grad():
        for p in model2.parameters():
            p.zero_()
    X2, y2 = torch.randn(32, 4), torch.randn(32, 1)

    def closure2():
        return torch.nn.functional.mse_loss(model2(X2), y2)

    true_zero_loss = closure2().item()
    EvE(model2.parameters(), lr=0.05, seed=2, init_from_model=True)
    with torch.no_grad():
        anchor_loss = float(closure2())
    assert abs(anchor_loss - true_zero_loss) < 1e-5, "init_from_model anchor should match the model's real starting loss"
    print("EvE init_from_model ok -- anchor loss matches model's real starting loss:", anchor_loss)


    torch.manual_seed(2)
    x = torch.nn.Parameter(torch.randn(5))

    def sphere():
        return (x**2).sum()

    opt_pm = EvE([x], lr=0.05, seed=3, use_pm=True, bounds=([torch.full((5,), -5.0)], [torch.full((5,), 5.0)]))
    losses_pm = [opt_pm.step(sphere) for _ in range(200)]
    assert losses_pm[-1] < losses_pm[0]
    print("EvE use_pm ok -- loss:", losses_pm[0], "->", losses_pm[-1])


    def all_high(n_pop, n_var, gen):
        return torch.full((n_pop, n_var), 0.9)

    torch.manual_seed(3)
    x2 = torch.nn.Parameter(torch.randn(5))

    def sphere2():
        return (x2**2).sum()

    opt_custom_sampling = EvE(
        [x2], lr=0.05, seed=4, sampling=all_high, bounds=([torch.full((5,), -5.0)], [torch.full((5,), 5.0)])
    )
    assert torch.all(opt_custom_sampling._pop[0][0] == 0.9), "custom sampling fn should have been used verbatim"
    losses_cs = [opt_custom_sampling.step(sphere2) for _ in range(100)]
    assert losses_cs[-1] < losses_cs[0]
    print("EvE custom sampling ok -- loss:", losses_cs[0], "->", losses_cs[-1])


    torch.manual_seed(4)
    x3 = torch.nn.Parameter(torch.randn(5))

    def sphere3():
        return (x3**2).sum()

    opt_custom_fallback = EvE(
        [x3],
        lr=0.1,
        seed=5,
        fallback_factory=lambda params: torch.optim.SGD(params, lr=0.1, momentum=0.9),
        bounds=([torch.full((5,), -5.0)], [torch.full((5,), 5.0)]),
    )
    assert isinstance(opt_custom_fallback._fallback_opts[0], torch.optim.SGD)
    losses_cf = [opt_custom_fallback.step(sphere3) for _ in range(200)]
    assert losses_cf[-1] < losses_cf[0]
    print("EvE custom fallback ok -- loss:", losses_cf[0], "->", losses_cf[-1])
