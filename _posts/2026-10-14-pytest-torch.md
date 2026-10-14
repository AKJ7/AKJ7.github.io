---
title: "pytest-torch: Testing PyTorch Models Like You Mean It"
date: 2026-10-14 22:00:00 +0200
categories: [Machine Learning, PyTorch]
tags: [pytorch, testing, pytest, deep-learning, open-source]
toc: true
description: "A pytest plugin for the tests every model needs: inference contracts, invariance, deployment parity, checkpoints, dataset metrics with confidence intervals, and zero-code testing of any model."
image:
    path: img/pytest-torch-cover.png
    alt: pytest-torch logo
---

Most bugs in deep learning code don't crash. A pooling layer averages over padding tokens, and accuracy drops by a point. A model ships with dropout still active. A refactor reorders two operations, the loss curve looks fine, and the exported model disagrees with the one that was evaluated. Nothing raises; the numbers are just slightly wrong.

Ordinary unit tests rarely catch these, because writing the right test is tedious: seeding, building inputs, running on every device, choosing tolerances, comparing nested outputs, and then turning `assert torch.allclose(a, b)` failing into something you can act on.

[pytest-torch](https://github.com/AKJ7/pytest-torch) is a pytest plugin that turns those tests into one-liners. This post walks through what it catches, with real outputs from the library.

```bash
pip install pytest-torch
```

It needs Python 3.9+, pytest 7+ and PyTorch 2.0+, registers itself with pytest, and doesn't import `torch` until a test uses it.

## 1. Try it on any model, without writing a test

The fastest way to see what it does is to point it at a model:

```bash
pytest --torch-model=hf:google-bert/bert-base-uncased
```

```
pytest-torch: testing hf:google-bert/bert-base-uncased (random-init); checks: contract, invariance, checkpoint, gradients
torch-model::test_builds[hf:google-bert/bert-base-uncased-cpu] PASSED
torch-model::test_inference_contract[hf:google-bert/bert-base-uncased-cpu] PASSED
torch-model::test_invariance[hf:google-bert/bert-base-uncased-cpu] PASSED
torch-model::test_checkpoint_roundtrip[hf:google-bert/bert-base-uncased-cpu] PASSED
torch-model::test_gradients_finite[hf:google-bert/bert-base-uncased-cpu] PASSED
```

The model is built from its config with random weights (only `config.json` is downloaded), inputs are inferred (padded `input_ids` and `attention_mask` for a text model), and a suite of checks runs on every available device. It works the same with `timm:resnet50`, `torchvision:mobilenet_v3_small`, or your own `mypackage.models:Classifier`. When inputs can't be inferred, it says how to provide them.

## 2. A bug you won't see in the loss curve

Here is an encoder whose pooled output averages every position, padding included:

```python
import torch
from torch import nn
from pytest_torch import assert_mask_invariant


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(100, 16)
        self.proj = nn.Linear(16, 16)

    def forward(self, input_ids, attention_mask):
        h = self.proj(self.emb(input_ids))
        return {"hidden": h, "pooled": h.mean(dim=1)}   # bug: padding is averaged in


torch.manual_seed(0)
batch = {"input_ids": torch.randint(1, 100, (4, 12)),
         "attention_mask": torch.ones(4, 12, dtype=torch.long)}
batch["attention_mask"][1, 8:] = 0
batch["attention_mask"][3, 5:] = 0

assert_mask_invariant(Encoder().eval(), batch)
```

The check replaces the tokens under the mask with other tokens. If nothing leaks, nothing changes:

```
Mask invariance violated (masked tokens leak): 1 problem(s)
  output['pooled']: 32/64 elements differ beyond tolerance (rtol=1.3e-06, atol=1e-05)
    max abs error: 3.861e-01 at index (3, 11) (actual=0.357001, expected=-0.0290699)
    max rel error: 2.613e+01
```

The message names the output (`pooled`, not `hidden`), how much of it is wrong, and the worst element. That's the pattern throughout the library: failures say *where* and *how badly*, for any nested output of dicts, tuples, dataclasses or Hugging Face `ModelOutput`s.

## 3. Three questions, three starting points

Most model tests answer one of three questions.

### Does one prediction look right?

`check_inference_contract` runs the checks every inference model should pass. Here is a model someone forgot to put in eval mode:

```python
from pytest_torch import check_inference_contract

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Dropout(0.1),
                      nn.BatchNorm1d(32), nn.Linear(32, 4))
print(check_inference_contract(model, torch.randn(8, 16), raise_on_failure=False))
```

```
Inference contract: 1 passed, 4 failed, 2 skipped
  [FAIL] eval_mode
         6 module(s) in training mode (e.g. <root>, 0, 1)
         hint: Call model.eval() before inference; dropout and batch norm behave differently in training mode.
  [PASS] finite_outputs
  [FAIL] deterministic
         ...
  [FAIL] batch_invariance
         Batch invariance: the model fails on a single sample: ValueError: Expected more than 1 value
         per channel when training, got input size torch.Size([1, 32]). The model is in training mode
         (<root>, 0, 1); call model.eval().
  [SKIP] device_parity (no other device available)
  [SKIP] dtype_parity (no dtypes requested, e.g. dtypes=(torch.bfloat16,))
  [FAIL] no_state_mutation
         3 parameter/buffer tensor(s) changed during inference (e.g. 3.running_mean, 3.running_var, 3.num_batches_tracked)
```
*(abridged)*

On a machine with a GPU, `device_parity` also compares CPU and GPU outputs. Together with `assert_output_spec` (shapes with symbolic dimensions, dtypes, value ranges, probabilities that sum to one), this covers "does the model produce sane outputs".

### Does a changed model still match the original?

Refactors, ports, compilation and quantization all produce a "new" model that should behave like the old one. When it doesn't, the useful question is *where* it starts to differ. Here a refactor added a stray scale between two layers:

```python
from pytest_torch import localize_divergence


class Original(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(16, 32)
        self.norm = nn.LayerNorm(32)
        self.act = nn.GELU()
        self.head = nn.Linear(32, 4)

    def forward(self, x):
        return self.head(self.act(self.norm(self.encoder(x))))


class Refactored(Original):
    def forward(self, x):
        h = self.norm(self.encoder(x))
        return self.head(self.act(h * 1.5))   # a stray scale crept in


torch.manual_seed(0)
original = Original().eval()
refactored = Refactored().eval()
refactored.load_state_dict(original.state_dict())
print(localize_divergence(original, refactored, torch.randn(8, 16)).format())
```

```
Divergence: 2 of 4 layers exceed tolerance (rtol=0.0001, atol=0.0001). First: 'act' (GELU)
  layer                            type               max abs   max rel  inputs
  act                              GELU              1.49e+00  6.30e+01  DIFF
  head                             Linear            4.40e-01  2.73e+00  DIFF
The first diverging layer already receives different inputs: the error entered between layers (functional ops, residual adds, glue code).
```

`encoder` and `norm` match; `act` is the first layer that differs, and since its *inputs* already differ, the bug sits in the code between `norm` and `act`. The same idea covers deployment: `assert_compile_parity`, `assert_export_parity` (including a dynamic batch dimension), AOTInductor packages, ONNX Runtime, and `torch.autocast`. For quantized or distilled models, `assert_models_agree` checks top-1 agreement and cosine similarity against a budget.

### Is the release as good as the last one?

Comparing an accuracy against a baseline sounds simple, until a 0.5% drop on 200 validation samples fails CI for no reason. `metric_baseline` records each metric together with what it measured (the dataset and split you name, a fingerprint of the data, the sample count, the metric configuration and the checkpoint), refuses to compare evaluations of different things, and compares per-sample metrics *paired*, with a bootstrap confidence interval:

```python
def accuracy(output, batch):
    return (output.argmax(-1) == batch[1]).float()     # one value per sample


def test_accuracy(model, val_loader, metric_baseline):
    metric_baseline(model, val_loader, {"accuracy": accuracy},
                    context={"dataset": "sst2", "split": "validation"},
                    minimum={"accuracy": 0.9})
```

A regression needs a drop beyond the tolerance *and* a confidence interval that excludes zero. The same 5% drop means different things at different sample sizes:

```
# 40 samples, 2 predictions flipped: noise, passes
drop +0.05 (95% CI +0 .. +0.125, n=40, paired)
# 4000 samples, 200 predictions flipped: a regression, fails
drop +0.05 (95% CI +0.04299 .. +0.057, n=4000, paired)
```

Release checks also include checkpoint compatibility (`assert_checkpoint_loads` lists every missing, unexpected and reshaped key at once, and suggests `strip_prefix="module."` for DDP checkpoints) and per-machine latency and memory baselines.

## 4. Generated suites, and one suite per config

Writing the same tests for every model gets old. `ModelSuite` turns one class into a suite:

```python
from pytest_torch import ModelSuite, TensorSpec


class TestClassifier(ModelSuite):
    checks = ("contract", "compile", "export", "output_spec", "checkpoint", "training")
    output_spec = {"logits": TensorSpec(("B", 3))}

    def make_model(self):
        return Classifier()

    def make_inputs(self, make_tensor):
        return make_tensor(8, 16)

    def loss(self, output, inputs):
        return output["logits"].logsumexp(-1).mean()
```

Each check becomes a test on every device. Projects with many YAML configs, one per experiment, can run the whole suite once per config by setting `configs = sorted(Path("configs").glob("*.yaml"))`. The training checks include `assert_frozen`, which catches a "frozen" encoder that still drifts, either because the optimizer's weight decay moves it or because its BatchNorm statistics keep updating in training mode.

## 5. Beyond classifiers

A few more things that turned out to matter in practice:

- **Streaming and stateful models.** `assert_streaming_equivalent` checks that a model run chunk by chunk agrees with one full pass, that a reset really resets, and that two requests interleaved with separate states don't contaminate each other. The expected equivalence is yours to declare, because some models intentionally behave differently when chunked (look-ahead, warm-up).
- **Training sanity.** Finite gradients, every parameter receiving a gradient, the loss going down, overfitting a single batch, and `assert_gradcheck` for hand-written backward passes, a natural companion to [custom loss functions]({% post_url 2026-10-10-custom-loss-functions-pytorch %}).
- **Failure reproduction.** `pytest --torch-repro-dir=repro` saves the seed, the inputs and the environment of every failing test; `pytest --torch-repro=repro/<test>.pt` replays it bit for bit on another machine, or on the CPU when CI failed on a GPU.
- **Devices and dtypes.** Tests run on CPU, CUDA, Intel XPU and Apple MPS, optionally across dtypes, with tolerances that follow the dtype and can be set once in `pyproject.toml`.

## 6. Design choices

A few decisions shaped the library:

- **Small helpers first.** Every check is a plain function you can call in any test. `ModelSuite` and `--torch-model` are built on top of them, not the other way around.
- **Readable failures over boolean assertions.** Every comparison reports the path, the count, the worst element and the tolerance that was used.
- **Exact where exactness matters.** Integers compare exactly (including above 2⁵³, where float64 can't tell neighbours apart), complex tensors compare both parts, and NaN handling is explicit.
- **Reproducible inputs.** `make_tensor` draws on the CPU with a seeded generator, so every device sees identical inputs, and failing tests print their seed.
- **Cheap to install.** Importing the plugin doesn't import `torch`, so projects that don't test models pay nothing.

## Limitations

pytest-torch tests models; it isn't a training framework or a benchmark suite. Some checks depend on the PyTorch version: `torch.export` checks skip on torch 2.0, and AOTInductor needs torch 2.6 and a C++ compiler. Host-device synchronization checks are CUDA-only. Zero-code testing covers common model shapes; anything more specific is a `ModelSuite` away.

## Get started

```bash
pip install pytest-torch
pytest --torch-model=mypackage.models:MyModel
```

- Source and examples: [github.com/AKJ7/pytest-torch](https://github.com/AKJ7/pytest-torch), including runnable integration projects (a text classifier, an image pipeline, a small GPT with a KV cache, and a navigation policy tested per config)
- Documentation: [akj7.github.io/pytest-torch](https://akj7.github.io/pytest-torch/)
- Package: [pypi.org/project/pytest-torch](https://pypi.org/project/pytest-torch/)

Issues and pull requests are welcome.
