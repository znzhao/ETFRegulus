# GPU and Environment Setup

Resolves the compute half of [Stage 0](stages.md). D5 assumed "single machine with a GPU"; this file records
what that machine actually is, what to install, and how Stage 0 verifies it every run.

**Status: verified end-to-end on this machine 2026-08-31.** The GPU works. **It also measured 2.7x *slower*
than CPU for our actual workload** — see §3 and §5. `training.device: cpu` is the measured default.

---

## 1. The machine

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 Ti, 8 GB GDDR6 |
| Architecture | Ampere, compute capability **sm_86** |
| Driver | 596.49 (supports up to CUDA 13.2) |
| Driver model | WDDM (Windows display driver — see §4) |
| Python | 3.11.11, in `.venv/` |
| OS | Windows 11 Pro 26200 |

Two facts from this that shape the plan:

- **8 GB of VRAM, shared with the desktop.** `nvidia-smi` showed ~1.1 GB already consumed by Chrome, VS Code,
  Discord and the rest of a normal desktop session. Budget **~6.5 GB usable**, not 8. This is ample for the
  small MLP policies in [rl-training.md](rl-training.md) and would not be for anything larger.
- **The driver is newer than any CUDA runtime we will install**, so the wheel's CUDA version is a free choice —
  NVIDIA drivers are backward compatible with older CUDA runtimes.

---

## 2. What to install

```bash
.venv/Scripts/python.exe -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
.venv/Scripts/python.exe -m pip install stable-baselines3[extra] gymnasium
```

**Why cu126 rather than cu128/cu129/cu130** (all four are available for this driver):

- sm_86 is supported by every one of them, so compatibility is not the deciding factor.
- cu126 carries the newest stable torch (2.13.0) *and* is the most widely exercised CUDA 12.x line, which
  matters because the failure mode we are avoiding is an obscure kernel bug surfacing 40 hours into a
  curriculum run.
- cu129 is stuck at torch 2.9.0 and would pin us backward for no benefit.
- cu130 also offers 2.13.0 and is a legitimate alternative; cu126 is chosen for maturity, not necessity. If a
  future dependency requires CUDA 13, switching is a one-line index change.

Pin both `torch` and its CUDA build in `requirements.txt` — `torch==2.13.0+cu126` with the extra index URL
recorded. An unpinned `pip install torch` on Windows resolves to a different wheel and silently changes the
numerics of a run.

---

## 3. Measured: **the GPU is slower than the CPU for this workload**

This is the most important thing in this file, it runs against intuition, and it is **measured, not predicted**.

### The result

| Benchmark | CPU | CUDA | |
|---|---|---|---|
| Raw matmul, 200x `(4096x256) @ (256x256)` | 0.1347s | **0.0156s** | GPU 8.6x faster |
| **End-to-end PPO, 5120 steps, `net_arch=[256,256]`** | **2.35s** | 6.44s | **CPU 2.7x faster** |

The matmul says the GPU is 8.6x faster at the arithmetic. The end-to-end run says the GPU is 2.7x slower at
the actual job. Both are true, and the second is the one that decides the config.

### Why — decomposed, same 300→256→256→26 network

| Phase | CPU | CUDA | |
|---|---|---|---|
| Rollout step: 1 obs forward + host↔device transfers | **43 µs** | 449 µs | CPU 10.4x |
| …compute only, tensor already resident on device | **36 µs** | 288 µs | CPU 8.1x |
| Update: fwd + bwd + Adam @ batch 4096 | 19081 µs | **2043 µs** | CUDA 9.3x |

The middle row is the informative one. **Transfer is not the main cost** — with the tensor already on the
device, CUDA is still 288 µs against the CPU's 36 µs. Transfers add ~160 µs on top of a 288 µs floor.

The floor is **fixed per-operation overhead**: each of the ~6 layer ops costs a kernel launch plus Python
dispatch, on the order of 40-50 µs, independent of how much arithmetic it performs. A batch-1 forward pass
through this network is ~5,000 FLOPs — there is nothing to hide the overhead behind. The GPU spends
essentially all of its time on paperwork.

At batch 4096 the identical overhead is amortized over 4096x the arithmetic, and real throughput appears —
9.3x, the other way.

PPO runs both phases, in this ratio (2048 rollout steps + 40 update minibatches per iteration):

```
        rollout          update           total
cpu      88.6 ms  +     763.2 ms   =     851.8 ms
cuda    919.1 ms  +      81.7 ms   =    1000.8 ms
```

The GPU wins the phase that happens 40 times and loses the phase that happens 2048 times.

### The crossover condition — and why this margin is thin

The environment's own step cost `E` appears in **both** columns and cancels. What decides the device is only:

```
2048 x (cuda_rollout - cpu_rollout)    vs    (cpu_update - cuda_update)
            831 ms                                    681 ms
```

CPU wins by 150 ms — a **1.2x margin**, not the 2.7x the Pendulum end-to-end test suggested (Pendulum's
near-free env step inflates rollout's share of the total).

**This is close, and it tilts with architecture.** The left side scales with layer *count* — more sequential
ops means more kernel launches at batch 1. The right side scales with parameter *count*. A wider per-asset
encoder, and especially an attention block over assets, plausibly flips it. Hence the standing obligation in
§7 to re-measure rather than inherit the answer.

SB3 warns about this class of problem unprompted:

> You are trying to run PPO on the GPU, but it is primarily intended to run on the CPU when not using a CNN
> policy... The model will train, but the GPU utilization will be poor and the training might take longer than
> on CPU.

The warning is correct, and on this machine it is quantified: 2.7x.

### The underlying structure

A PPO iteration is two phases:

| Phase | Where it runs | Cost here |
|---|---|---|
| **Rollout** — stepping the environment | **CPU**, single-threaded per worker | Dominant |
| **Update** — forward/backward on the policy | GPU | Trivial |

Our environment step does: a feature lookup, a simplex projection (`O(K log K)` sort), a risk-envelope
evaluation (~10 matrix-vector products through the `alpha` bisection), and a ledger update. That is *pure
Python and NumPy on the CPU*, and it happens 2048 times per worker per iteration. The policy update is a
handful of gradient steps on a two-layer 256-unit MLP with a batch of 4096 — microseconds of GPU time.

**Consequences:**

1. **`training.device: cpu` is the default**, on measured evidence. Not a hedge — a benchmark.
2. **Scale rollouts across CPU cores.** `SubprocVecEnv`, worker count from the Stage 6 benchmark. This is
   where wall-clock time is actually won.
3. **The optimization target is `env.step`, not the network.** If training is slow, profile the environment —
   most likely the risk envelope ([risk-envelope.md](risk-envelope.md) §6 covers the per-session caching that
   makes it cheap). A faster GPU would change nothing.
4. **Re-benchmark if the policy grows — the margin is only 1.2x.** This is not a comfortable win. A wider
   per-asset encoder or an attention block over assets could flip it. Stage 6 re-runs the comparison whenever
   the architecture changes, so the decision stays measured rather than inherited.
5. **If the GPU is ever wanted, the lever is batch size at rollout.** More parallel envs means the policy
   forward pass batches across workers instead of running at batch 1, which is the single change that would
   move the crossover. Worth testing at 16+ workers in Stage 6.

The GPU is verified, available, and currently the wrong tool. That is a useful thing to have established
before the first long training run rather than after it.

---

## 4. Windows-specific notes

- **WDDM driver model** means the OS manages GPU memory and the desktop compositor shares the device. Expect
  slightly higher latency and less predictable free VRAM than a Linux/TCC setup. Irrelevant at our memory
  footprint; worth knowing before blaming a benchmark.
- **`SubprocVecEnv` uses `spawn` on Windows**, not `fork`. Every worker re-imports the module and re-pickles
  the env, so: keep module-level import cost low, guard all entry points with `if __name__ == "__main__":`
  (the stage harness does this), and make sure the env is picklable — a loaded DataFrame captured in a closure
  will either fail to pickle or be copied into every worker. Load the feature frames **inside** the env
  factory, not in the parent.
- Each spawned worker holds its own copy of the feature matrices. With ~25 tickers x ~20 years x ~80 features
  in float32 this is small (tens of MB), but check it before scaling to 16 workers.
- Memory-map the feature parquet if worker RSS becomes a problem. Do not solve a problem you have not measured.

---

## 5. Verification results — 2026-08-31

All checks executed against this machine. Every line below is measured output, not expectation.

```
torch:          2.13.0+cu126
cuda runtime:   12.6
is_available:   True
device:         NVIDIA GeForce RTX 3060 Ti
capability:     (8, 6)
arch_list:      sm_50 sm_60 sm_61 sm_70 sm_75 sm_80 sm_86 sm_90   <- sm_86 present
matmul gpu==cpu: True
vram free/total: 6.97 / 8.00 GB
numpy 2.4.6 | stable-baselines3 2.9.0 | gymnasium 1.3.0 | tensorboard 2.21.0

200x (4096x256 @ 256x256)   cpu 0.1347s   cuda 0.0156s   -> GPU 8.6x faster
PPO 5120 steps, [256,256]   cpu 2.35s     cuda 6.44s     -> CPU 2.7x faster

phase decomposition, 300->256->256->26:
  rollout step (1 obs + transfers)  cpu    43 us   cuda   449 us   -> CPU 10.4x
  rollout step (no transfer)        cpu    36 us   cuda   288 us   -> CPU  8.1x
  update (fwd+bwd+adam, batch 4096) cpu 19081 us   cuda  2043 us   -> CUDA 9.3x
  per iteration (2048 + 40)         cpu 851.8 ms   cuda 1000.8 ms  -> CPU  1.2x
```

**Conclusion: the GPU is fully functional and correctly configured, and CPU is the faster device for this
workload.** Both facts are recorded because both matter — a future architecture change could flip the second
without affecting the first.

### Versions to pin in `requirements.txt`

```
--extra-index-url https://download.pytorch.org/whl/cu126
torch==2.13.0+cu126
stable-baselines3[extra]==2.9.0
gymnasium==1.3.0
numpy==2.4.6
pandas==3.0.5
```

Note `numpy>=2` and `pandas>=3` — both are major versions with breaking changes from the 1.x/2.x era that most
online examples assume. Expect to hit them in the data layer, not to be warned about them.

### A failure this run actually caught

`pip install torch` from the PyTorch index installed **without numpy**, producing only:

```
UserWarning: Failed to initialize NumPy: No module named 'numpy'
```

torch imported fine, CUDA worked, and every tensor operation succeeded — but any `.numpy()` conversion would
have failed at runtime, deep inside SB3. A warning, not an error. This is exactly the class of silent
degradation Stage 0 exists to convert into a hard failure.

---

## 6. What Stage 0 checks, every run

`scripts/s00_check_env.py` asserts the following and prints a pass/fail table with fix hints. These are cheap
and catch the environment drifting after a driver update or a `pip install` that pulled a CPU-only wheel:

```python
import torch

torch.__version__                    # includes the +cuXXX build tag
torch.version.cuda                   # runtime CUDA version
torch.cuda.is_available()            # must be True
torch.cuda.get_device_name(0)
torch.cuda.get_device_capability(0)  # expect (8, 6)
torch.cuda.get_arch_list()           # must contain sm_86
torch.cuda.mem_get_info()            # (free, total) — surfaces the desktop's share
```

Then a real computation, because `is_available()` returning True proves only that a driver was found:

```python
a = torch.randn(2048, 2048, device="cuda")
assert torch.allclose((a @ a).cpu(), a.cpu() @ a.cpu(), atol=1e-3)
```

And an end-to-end check, because a working matmul does not prove SB3 is wired up:

```python
PPO("MlpPolicy", gym.make("Pendulum-v1"), device="cuda").learn(1000)
import numpy; numpy.asarray(torch.randn(4).cpu())   # catches the numpy-less torch install
```

**Two failure modes this specifically catches:**

- **A CPU-only torch wheel.** `pip install torch` from PyPI can resolve to one; `torch.cuda.is_available()`
  returns `False` and SB3 falls back to CPU with only a warning. Stage 0 **fails** instead, so it is impossible
  to discover mid-training that a run never had the GPU available at all.
- **torch installed without numpy** — observed on this machine. A `UserWarning`, nothing more, until a
  `.numpy()` call fails somewhere inside SB3.

Neither is caught by "did the install command exit 0".

---

## 7. Recorded decision

**D14 — CUDA stack pinned to `torch==2.13.0+cu126`. `training.device: cpu`, on measured evidence.**

The GPU is verified working and is **not** used for v1 training, because it benchmarked 2.7x slower than
CPU end-to-end for our policy size. Stage 0 verifies the GPU remains usable every run; Stage 6 re-measures
whether it is *worth* using, and re-measures again whenever the policy architecture changes. The device
actually used is recorded in every run manifest, so any result traces to the hardware path that produced it.
