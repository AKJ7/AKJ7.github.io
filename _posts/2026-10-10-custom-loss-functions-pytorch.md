---
title: "Creating Custom Loss Functions in PyTorch: From Equations to Gradients"
date: 2026-10-10 19:28:00 +0200
categories: [Machine Learning, PyTorch]
tags: [pytorch, loss-functions, autograd, mathematics, deep-learning]
math: true
toc: true
description: "Build custom objectives with tensor operations, reusable modules, quantile regression, composite losses, and gradient checks."
image:
    path: img/nn_cover.png
    alt: Neural Networks cove
---

PyTorch provides loss functions for classification, regression, ranking, and representation learning. Sometimes, however, the error we care about does not fit a standard criterion. Underestimating demand may cost more than overestimating it. Some observations may be missing. A prediction may need to satisfy both a data-fitting objective and a geometric constraint.

A custom loss lets us encode those requirements explicitly. The difficult part is usually deciding what to optimize and how to normalize it—not writing a backward pass.

This companion to [Understanding PyTorch Loss Functions]({% post_url 2026-10-10-pytorch-loss-functions %}) develops several objectives from equations, discusses their tradeoffs, and shows how to check the resulting gradients. API references use the PyTorch 2.14 documentation; these examples mostly rely on longstanding tensor operations.

## 1. Start with a prediction contract

Before writing code, specify:

- What does the model output: values, logits, probabilities, or embeddings?
- What shapes and dtypes should predictions and targets have?
- Which errors should be expensive?
- What should happen with missing targets or an empty valid set?
- Does the objective average over samples, coordinates, or weights?

For example, a weighted regression objective might be

$$
\mathcal{L}=\frac{\sum_{i=1}^{N}w_i(\hat y_i-y_i)^2}{\sum_{i=1}^{N}w_i},\qquad w_i\geq0.
$$

This differs from dividing the numerator by $N$. In the first convention, multiplying every weight by the same positive constant leaves the loss unchanged. In the second, it changes the gradient scale.

## 2. The simplest custom loss is a Python function

Compose ordinary PyTorch tensor operations and autograd can differentiate their composition. For a first example, recreate MSE:

```python
import torch
from torch import nn
import torch.nn.functional as F


def custom_mse(prediction, target):
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    if prediction.numel() == 0:
        raise ValueError("empty tensors have no mean loss")
    return (prediction - target).square().mean()


prediction = torch.tensor([1.0, 3.0], requires_grad=True)
target = torch.tensor([0.0, 1.0])
loss = custom_mse(prediction, target)
loss.backward()

print(loss.item())       # 2.5
print(prediction.grad)   # tensor([1., 2.])
```

For $N$ scalar elements,

$$\frac{\partial\mathcal{L}}{\partial\hat y_i}=\frac{2}{N}(\hat y_i-y_i).$$

This derivative explains the example's gradient without needing a neural network.

**Pros:** minimal code, easy mathematical inspection, and automatic differentiation.

**Cons:** configuration and persistent state need explicit handling; shape and reduction conventions are your responsibility.

For real MSE training, use the built-in criterion. Reimplementing it here is an exercise in understanding the contract.

## 3. A weighted loss with an explicit zero-weight policy

Here is the weighted objective above. It requires floating tensors of identical shape on the same device, finite nonnegative weights, and finite predictions and targets wherever weight is positive. Weights are treated as fixed metadata.

```python
def weighted_mse(prediction, target, weights):
    if prediction.shape != target.shape or weights.shape != target.shape:
        raise ValueError("prediction, target, and weights must match")
    if prediction.numel() == 0:
        raise ValueError("empty tensors are not supported")

    # Float32 accumulation helps for float16/bfloat16 inputs.
    # Keep float64 for numerical gradient checks.
    dtype = torch.float64 if prediction.dtype == torch.float64 else torch.float32
    pred = prediction.to(dtype)
    truth = target.to(dtype)
    weight = weights.detach().to(dtype)

    if not torch.isfinite(weight).all() or (weight < 0).any():
        raise ValueError("weights must be finite and nonnegative")

    active = weight > 0
    # Sanitize excluded coordinates BEFORE subtraction/squaring.
    pred = torch.where(active, pred, torch.zeros_like(pred))
    truth = torch.where(active, truth, torch.zeros_like(truth))
    numerator = (weight * (pred - truth).square()).sum()
    denominator = weight.sum()
    safe_denominator = torch.where(
        denominator > 0, denominator, torch.ones_like(denominator)
    )
    return numerator / safe_denominator


prediction = torch.tensor([1.0, 3.0], requires_grad=True)
target = torch.tensor([0.0, 1.0])
weights = torch.tensor([1.0, 3.0])
loss = weighted_mse(prediction, target, weights)  # 13 / 4 = 3.25
```

With all weights zero, this returns a graph-connected zero loss and zero direct gradients. That policy means “this batch supplies no regression supervision.” It does not mean the optimizer cannot update parameters: momentum, weight decay, or another loss term may still do so. Skip the optimizer step if that is your intended policy.

For a binary mask, averaging valid elements also weights samples with more valid coordinates more heavily. To weight samples equally, first reduce within each sample, then average the valid sample losses.

**Pros:** expresses observation importance; supports excluded coordinates; explicit normalization makes scaling predictable.

**Cons:** large weights can dominate; weight design may bias the fitted predictor; validation checks can synchronize accelerator execution and may need moving outside a performance-critical compiled path.

## 4. Use nn.Module for a reusable criterion

A module is useful when a loss has configuration or tensor state. Register fixed tensors as buffers so they follow device moves and appear in `state_dict` by default.

```python
class WeightedMSELoss(nn.Module):
    def __init__(self, feature_weights):
        super().__init__()
        weights = torch.as_tensor(feature_weights, dtype=torch.float32)
        if weights.ndim != 1 or weights.numel() == 0:
            raise ValueError("feature_weights must be a nonempty vector")
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("feature_weights must be finite and nonnegative")
        if weights.sum() <= 0:
            raise ValueError("at least one feature weight must be positive")
        self.register_buffer("feature_weights", weights.detach().clone())

    def forward(self, prediction, target):
        if prediction.ndim != 2 or prediction.shape != target.shape:
            raise ValueError("expected matching (N, D) tensors")
        if prediction.shape[1] != self.feature_weights.numel():
            raise ValueError("feature count does not match weights")
        weights = self.feature_weights.unsqueeze(0).expand_as(prediction)
        return weighted_mse(prediction, target, weights)


criterion = WeightedMSELoss([1.0, 2.0, 0.5])
prediction = torch.randn(4, 3, requires_grad=True)
target = torch.randn(4, 3)
loss = criterion(prediction, target)
```

For accelerator training, move both model and criterion to the chosen device. Buffers are not learnable parameters. If the criterion does contain `nn.Parameter` objects, include its parameters in the optimizer.

**Pros:** reusable interface, organized configuration, and managed tensor state.

**Cons:** more boilerplate; a module does not automatically fix numerical or modeling errors. Unconstrained trainable loss weights can also create degenerate objectives—for example, learning to set every penalty weight to zero.

## 5. Quantile loss: asymmetric prediction errors

Suppose underestimation deserves a stronger penalty. For residual $r=y-\hat y$ and quantile level $\tau\in(0,1)$, the pinball loss is

$$\ell_\tau(r)=\max(\tau r,(\tau-1)r).$$

```python
def quantile_loss(prediction, target, tau=0.9):
    if not 0 < tau < 1:
        raise ValueError("tau must lie strictly between zero and one")
    if prediction.shape != target.shape or prediction.numel() == 0:
        raise ValueError("expected matching nonempty tensors")
    residual = target - prediction
    return torch.maximum(tau * residual, (tau - 1) * residual).mean()


prediction = torch.tensor([1.0, 3.0], requires_grad=True)
target = torch.tensor([2.0, 2.0])
loss = quantile_loss(prediction, target, tau=0.9)  # 0.5
```

Here underprediction by one costs `0.9`; overprediction by one costs `0.1`. At a population optimum, this estimates a conditional quantile under appropriate conditions. A 90th-percentile estimate is not a conditional mean.

**Pros:** makes asymmetry explicit and supports quantile forecasting without specifying a complete observation distribution.

**Cons:** nonsmooth at zero; different quantile models may produce crossing predictions; finite-sample coverage and calibration still require evaluation.

## 6. Combine objectives without losing track of their scales

An image reconstruction model might need both squared and absolute error:

$$\mathcal{L}=\alpha\mathcal{L}_{\mathrm{MSE}}+(1-\alpha)\mathcal{L}_{\mathrm{L1}}.$$

```python
class CombinedRegressionLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between zero and one")
        self.alpha = alpha

    def forward(self, prediction, target):
        return (
            self.alpha * custom_mse(prediction, target)
            + (1 - self.alpha) * F.l1_loss(prediction, target)
        )
```

MSE has squared target units; L1 has target units. Their numerical balance changes if you rescale the targets. Equal coefficients do not imply equal gradient contributions.

Log each component separately, evaluate different coefficients on validation data, and compare the metric that matters in the application.

**Pros:** combines complementary requirements using familiar building blocks.

**Cons:** coefficients require tuning; terms can conflict or have incompatible scales; adding more terms does not guarantee better predictions.

## 7. A geometry-aware regularizer

For a fixed graph with nonnegative edge weights, nearby samples can be encouraged to have nearby embeddings:

$$\mathcal{R}(Z)=\frac{\sum_{(i,j)\in E}w_{ij}\|z_i-z_j\|^2}{\sum_{(i,j)\in E}w_{ij}}.$$

```python
def graph_smoothness(embeddings, edge_index, edge_weights):
    # embeddings: (N, D); edge_index: (2, E), dtype=torch.long
    # edge_weights: (E,), finite nonnegative fixed weights
    source, destination = edge_index
    if source.numel() == 0:
        return embeddings.sum() * 0.0
    squared_distances = (
        embeddings[source] - embeddings[destination]
    ).square().sum(dim=1)
    weights = edge_weights.detach()
    denominator = weights.sum()
    safe_denominator = torch.where(
        denominator > 0, denominator, torch.ones_like(denominator)
    )
    return (weights * squared_distances).sum() / safe_denominator
```

Under the convention that each undirected edge is counted once, the unnormalized numerator equals $\operatorname{tr}(Z^T LZ)$ for the corresponding graph Laplacian. Counting both directions doubles the numerator before normalization.

Use this alongside an objective that prevents collapse. Constant embeddings minimize graph smoothness alone. Classification, reconstruction, variance constraints, or other information-preserving requirements can change that situation.

**Pros:** incorporates neighborhood structure; connects tensor objectives to graph-based numerical methods.

**Cons:** depends on graph quality; can smooth across genuine class boundaries; a graph built on a full dataset needs careful indexing or sampling when training mini-batches.

## 8. Check values and gradients before training

A falling training loss cannot prove that an implementation matches its equation. Begin with hand-computable values and compare against a reference where available:

```python
prediction = torch.tensor([1.0, 3.0], dtype=torch.float64, requires_grad=True)
target = torch.tensor([0.0, 1.0], dtype=torch.float64)
weights = torch.tensor([1.0, 3.0], dtype=torch.float64)

loss = weighted_mse(prediction, target, weights)
torch.testing.assert_close(loss, torch.tensor(3.25, dtype=torch.float64))
gradient, = torch.autograd.grad(loss, prediction)
torch.testing.assert_close(
    gradient, torch.tensor([0.5, 3.0], dtype=torch.float64)
)
torch.testing.assert_close(custom_mse(prediction, target), F.mse_loss(prediction, target))
```

For a smooth objective, compare autograd derivatives with numerical finite differences using `gradcheck`:

```python
prediction = torch.randn(3, dtype=torch.float64, requires_grad=True)
target = torch.randn(3, dtype=torch.float64)
weights = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

passed = torch.autograd.gradcheck(
    lambda value: weighted_mse(value, target, weights),
    (prediction,),
)
assert passed
```

Avoid nondifferentiable points when checking quantile or hinge losses: finite differences across a kink need not match the selected subgradient. Check zero-weight behavior, invalid inputs, output shape, and device behavior separately. A gradient check verifies a local derivative, not the suitability of the objective.

## 9. When do we need a custom backward pass?

Usually we do not. `torch.autograd.Function` is useful when integrating an operation outside ordinary differentiable PyTorch composition, or when implementing a specialized forward/backward pair for performance.

| Approach | Choose it when | Main cost |
|---|---|---|
| Python function | Tensor composition is enough | Manage configuration explicitly |
| `nn.Module` | Configuration, buffers, or trainable state matter | More state to manage |
| `autograd.Function` | A specialized operation needs an explicit derivative | Implement and validate backward correctly |

An explicit backward must apply the chain rule, multiply by the incoming gradient, and return gradients corresponding to the forward inputs. Test it with `gradcheck`; test higher-order derivatives separately if your application needs them. Advanced transformations may require additional integration work.

Implementing backward manually does not make a genuinely nondifferentiable operation differentiable. A surrogate derivative is a separate modeling choice and should be documented as such.

[Official guidance: extending autograd](https://docs.pytorch.org/docs/2.14/notes/extending.html)

## 10. Common mistakes

| Mistake | Consequence | Better approach |
|---|---|---|
| Convert predictions to NumPy for loss computation | Leaves the autograd computation | Keep computation in PyTorch |
| Return `loss.item()` | Returns a Python number | Return the tensor; use `.item()` for logging |
| Wrap an existing loss in `torch.tensor(loss)` | Creates a detached tensor | Return the existing loss |
| Detach the prediction | Disconnects the model from the objective | Detach only values meant to be fixed |
| Use `argmax` before a classification criterion | Discards differentiable class scores | Pass logits to the appropriate criterion |
| Let `(N, 1)` and `(N,)` broadcast in regression | Can produce `(N, N)` residuals | Require matching shapes |
| Apply sigmoid before BCE with logits | Uses the wrong input contract | Pass raw logits |
| Mask NaNs only after squaring residuals | Invalid arithmetic can contaminate gradients | Sanitize excluded values first |
| Divide by an empty valid count | Produces an undefined result | Define an explicit empty-set policy |

Hard masks depending only on metadata do not need gradients. If masks are supposed to be learned from predictions, hard comparisons will not differentiate their switching decisions; consider whether a smooth surrogate matches the intended objective.

For sensitive logarithms, exponentials, and small denominators, choose a numerically stable formulation and suitable accumulation dtype. Adding an epsilon changes the objective near that epsilon; treat it as a deliberate choice rather than a universal fix.

## 11. Put the criterion in a training loop

This small synthetic example demonstrates integration without downloading a dataset. Execute it after the definitions above:

```python
torch.manual_seed(42)
inputs = torch.randn(128, 5)
true_weights = torch.randn(5, 3)
targets = inputs @ true_weights + 0.1 * torch.randn(128, 3)

model = nn.Linear(5, 3)
criterion = WeightedMSELoss([1.0, 2.0, 0.5])
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

for step in range(200):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    prediction = model(inputs)
    loss = criterion(prediction, targets)
    loss.backward()
    optimizer.step()
    if step % 50 == 0:
        print(f"step={step:03d}, loss={loss.item():.4f}")
```

This is an integration demonstration, not a generalization benchmark. A real experiment needs held-out data, a baseline objective, and metrics tied to the intended use. For ordinary MNIST classification, retain cross entropy unless you have a specific reason to change the objective.

My preferred workflow is to write the equation, specify shapes and normalization, implement it with tensor operations, and verify a small numerical example. Only then do I put the loss into a training loop. A custom objective earns its complexity when it expresses a requirement that the baseline leaves out.

## References

- [Autograd mechanics](https://docs.pytorch.org/docs/2.14/notes/autograd.html)
- [Extending PyTorch and checking custom derivatives](https://docs.pytorch.org/docs/2.14/notes/extending.html)
- [nn.Module and registered buffers](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html)
- [Tensor.detach](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.detach.html)
- [torch.tensor and copying existing tensors](https://docs.pytorch.org/docs/2.14/generated/torch.tensor.html)
- [BCEWithLogitsLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BCEWithLogitsLoss.html)

Code blocks share imports and definitions introduced earlier. Their Python syntax was checked during drafting; runtime execution and the illustrated gradient checks were not performed because PyTorch was unavailable in the drafting environment.
