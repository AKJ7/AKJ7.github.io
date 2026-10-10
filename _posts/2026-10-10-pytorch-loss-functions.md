---
title: "Understanding PyTorch Loss Functions: A Practical Guide"
date: 2026-10-10 18:30:00 +0200
categories: [Machine Learning, PyTorch]
tags: [pytorch, loss-functions, mathematics, deep-learning]
math: true
toc: true
description: "A tour of every documented core PyTorch loss, with equations, input conventions, examples, and practical advice."
image:
    path: img/nn_cover.png
    alt: Neural Networks cover
---

A neural network produces predictions. A loss function decides which predictions deserve a penalty. That decision shapes what the network learns: accurate values, calibrated probabilities, useful rankings, or an embedding where related samples lie together.

When training an MNIST classifier, `CrossEntropyLoss` is a natural starting point. But why cross entropy rather than mean squared error? And what are the other losses in PyTorch actually for?

This post explores the complete **Loss Functions** lists in `torch.nn` and `torch.nn.functional` in the **PyTorch 2.14 documentation**, checked on 10 October 2026. It also covers `AdaptiveLogSoftmaxWithLoss`, a related output module. Older installations may lack newer APIs, particularly `LinearCrossEntropyLoss`.

“All losses” needs a boundary: this is the core neural-network loss API. Libraries such as TorchVision expose additional task-specific losses, and arbitrary differentiable tensor expressions can define custom objectives. Those form an open-ended ecosystem rather than a finite core list.

## What are we minimizing?

For model parameters $\theta$, a common training objective is

$$
\mathcal{L}(\theta)=\frac{1}{N}\sum_{i=1}^{N}\ell(f_\theta(x_i),y_i).
$$

Here $\ell$ is an individual penalty. An optimizer uses its gradient to update the model:

$$
\theta_{t+1}=\theta_t-\eta\nabla_\theta\mathcal{L}(\theta_t).
$$

The loss defines the objective; the optimizer defines how we move through parameter space. Changing Adam to SGD does not change the meaning of a prediction error. Changing MSE to a ranking loss does.

Throughout this post, formulas describe unweighted individual losses unless stated otherwise. Batch reduction and class weights can alter the final scaling. The pros and cons below describe practical tradeoffs, not universal performance rankings; validate the choice on your task.

## The complete core API at a glance

`nn` modules retain configuration; functional calls receive it explicitly. Both styles participate in autograd.

```python
import torch
from torch import nn
import torch.nn.functional as F

prediction = torch.tensor([0.2, 0.7], requires_grad=True)
target = torch.tensor([0.0, 1.0])

module_loss = nn.MSELoss()(prediction, target)
functional_loss = F.mse_loss(prediction, target)
```

| Family | Module | Functional equivalent |
|---|---|---|
| Regression | `L1Loss` | `l1_loss` |
| Regression | `MSELoss` | `mse_loss` |
| Regression | `HuberLoss` | `huber_loss` |
| Regression | `SmoothL1Loss` | `smooth_l1_loss` |
| Classification | `CrossEntropyLoss` | `cross_entropy` |
| Classification/output projection | `LinearCrossEntropyLoss` | `linear_cross_entropy` |
| Classification | `NLLLoss` | `nll_loss` |
| Binary/multilabel | `BCELoss` | `binary_cross_entropy` |
| Binary/multilabel | `BCEWithLogitsLoss` | `binary_cross_entropy_with_logits` |
| Binary margin | `SoftMarginLoss` | `soft_margin_loss` |
| Multilabel | `MultiLabelSoftMarginLoss` | `multilabel_soft_margin_loss` |
| Multiclass margin | `MultiMarginLoss` | `multi_margin_loss` |
| Multilabel margin | `MultiLabelMarginLoss` | `multilabel_margin_loss` |
| Probabilistic regression | `PoissonNLLLoss` | `poisson_nll_loss` |
| Probabilistic regression | `GaussianNLLLoss` | `gaussian_nll_loss` |
| Distribution matching | `KLDivLoss` | `kl_div` |
| Ranking | `MarginRankingLoss` | `margin_ranking_loss` |
| Pairwise embedding | `HingeEmbeddingLoss` | `hinge_embedding_loss` |
| Pairwise embedding | `CosineEmbeddingLoss` | `cosine_embedding_loss` |
| Triplet embedding | `TripletMarginLoss` | `triplet_margin_loss` |
| Triplet embedding | `TripletMarginWithDistanceLoss` | `triplet_margin_with_distance_loss` |
| Sequence alignment | `CTCLoss` | `ctc_loss` |

`LinearCrossEntropyOptions`, also listed in the documentation, configures the linear cross-entropy implementation; it is not a separate mathematical loss.

## 1. Regression: how expensive is an error?

Write $r=\hat y-y$ for a residual.

### MSELoss: punish large residuals strongly

$$\ell(r)=r^2.$$

Its derivative is $2r$: larger errors produce larger gradients. Squared error is a natural fit for Gaussian observation noise with fixed variance. At the population optimum, squared error estimates a conditional mean, assuming the relevant moments exist.

```python
prediction = torch.tensor([1.0, 3.0], requires_grad=True)
target = torch.tensor([0.0, 1.0])
loss = nn.MSELoss()(prediction, target)  # (1 + 4) / 2 = 2.5
```

Use it for continuous prediction when large deviations should be costly. Outliers and differently scaled target coordinates can dominate the objective.

**Pros:** Smooth gradients and strong penalties for large deviations; a simple baseline for conditional-mean prediction.

**Cons:** Outliers and target scaling can dominate gradients; averaging plausible outcomes can produce an implausible prediction.

[API: MSELoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MSELoss.html)

### L1Loss: a linear penalty

$$\ell(r)=|r|.$$

Away from zero, the gradient magnitude is constant. Extreme residuals therefore have less influence than under MSE. At the population optimum, absolute error estimates a conditional median.

```python
loss = nn.L1Loss()(prediction, target)  # (1 + 2) / 2 = 1.5
```

Use it when absolute deviations match the application or when you want less sensitivity to outliers. Its kink at zero does not prevent PyTorch from using a subgradient for optimization.

**Pros:** Less sensitive to extreme residuals than MSE; directly measures absolute error and targets a conditional median.

**Cons:** Has a kink at zero and nonvanishing gradient magnitude nearby; may need careful learning-rate scheduling near a solution.

[API: L1Loss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.L1Loss.html)

### HuberLoss: quadratic nearby, linear far away

For $\delta>0$,

$$
\ell_\delta(r)=
\begin{cases}
\frac12r^2,& |r|<\delta,\\
\delta\left(|r|-\frac12\delta\right),&\text{otherwise}.
\end{cases}
$$

Huber combines a smooth neighborhood around zero with bounded gradient magnitude for large residuals. The threshold has the same units as the residual, so target scaling matters.

```python
loss = nn.HuberLoss(delta=1.0)(prediction, target)
```

**Pros:** Smooth first derivative; combines quadratic behavior near zero with bounded residual gradients for large errors.

**Cons:** Delta needs tuning to target scale; changing it changes both the transition point and outer gradient magnitude.

[API: HuberLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.HuberLoss.html)

### SmoothL1Loss: similar shape, different scaling

For $\beta>0$,

$$
\ell_\beta(r)=
\begin{cases}
\frac{r^2}{2\beta},&|r|<\beta,\\
|r|-\frac12\beta,&\text{otherwise}.
\end{cases}
$$

At matching thresholds, Smooth L1 equals Huber divided by the threshold. They coincide at a threshold of one, but are not generally identical. Smooth L1's outer slope stays at one; Huber's outer slope is $\delta$. With `beta=0`, Smooth L1 becomes L1.

```python
loss = nn.SmoothL1Loss(beta=0.5)(prediction, target)
```

**Pros:** Smooth near zero and less sensitive to outliers than MSE; its outer slope remains fixed when beta changes.

**Cons:** Beta still depends on residual scale; changing it changes local curvature and does not supply an uncertainty model.

[API: SmoothL1Loss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.SmoothL1Loss.html)

## 2. Classification: logits, probabilities, and labels

A **logit** is an unconstrained score. Softmax converts a vector of logits into probabilities that sum to one. Sigmoid converts each logit independently into a probability.

That distinction separates multiclass classification from multilabel classification:

- **Multiclass:** an MNIST image has one label among ten digits.
- **Multilabel:** an image may contain a person, a bicycle, and a dog simultaneously.

### CrossEntropyLoss: one class among many

For logits $z$ and class index $y$,

$$\ell(z,y)=-z_y+\log\sum_c e^{z_c}.$$

Pass raw logits, not softmax outputs. For ordinary batch classification, logits have shape `(N, C)` and integer targets have shape `(N,)` with dtype `torch.long`.

```python
logits = torch.randn(4, 10, requires_grad=True)
labels = torch.tensor([2, 0, 7, 1], dtype=torch.long)
loss = nn.CrossEntropyLoss()(logits, labels)
```

Useful options include class `weight`, `ignore_index`, and `label_smoothing`. Soft probability targets are supported too: supply a floating tensor matching the logits shape, with valid class distributions. PyTorch does not fully validate those probability constraints. `ignore_index` applies to index targets.

**Pros:** Stable calculation from raw logits; supports class weighting, smoothing, and soft targets; a strong multiclass baseline.

**Cons:** Confident errors, including mislabeled examples, receive large penalties; does not directly optimize F1 or guarantee calibrated probabilities.

[API: CrossEntropyLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html)

### NLLLoss: log probabilities already available

$$\ell(\log p,y)=-\log p_y.$$

Unlike cross entropy, NLL expects **log probabilities**:

```python
loss = nn.NLLLoss()(F.log_softmax(logits, dim=1), labels)
```

For corresponding settings, this gives the same objective as cross entropy on logits. Use `log_softmax` directly rather than taking `log(softmax(...))`, which can lose numerical stability.

**Pros:** Useful when normalized log probabilities already exist; separates the probability model from its criterion.

**Cons:** Passing raw logits defines the wrong objective; no direct soft-target or label-smoothing interface like cross entropy.

[API: NLLLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.NLLLoss.html)

### LinearCrossEntropyLoss: learn the classifier inside the criterion

This API receives hidden features, applies an internal trainable linear projection, and computes cross entropy. It changes how the output computation is organized, not the underlying classification objective.

```python
# Requires an installation exposing this newer API.
head = nn.LinearCrossEntropyLoss(in_features=64, num_classes=10)
features = torch.randn(4, 64, requires_grad=True)
optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
loss = head(features, labels)
```

When training an encoder too, include both encoder and criterion parameters in the optimizer. Otherwise the internal classifier will not learn. `LinearCrossEntropyOptions` enables configurable chunking; optimized paths have autograd restrictions, so consult the version-specific documentation before using higher-order derivatives.

**Pros:** Combines a trainable classifier with cross entropy; configurable chunking can reduce intermediate-logit memory.

**Cons:** Availability depends on PyTorch version; its parameters must reach the optimizer, and chunked paths restrict advanced autograd usage.

[API: LinearCrossEntropyLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LinearCrossEntropyLoss.html)

### BCELoss: binary cross entropy on probabilities

For $p\in[0,1]$ and target $y\in[0,1]$,

$$\ell(p,y)=-y\log p-(1-y)\log(1-p).$$

```python
binary_logits = torch.randn(4, 1, requires_grad=True)
binary_targets = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
loss = nn.BCELoss()(binary_logits.sigmoid(), binary_targets)
```

Input and target shapes must match. BCE can handle independent labels, but for a model emitting logits, the next API is preferable.

**Pros:** Simple when probabilities are already available; supports binary and independent-label probability targets.

**Cons:** Separate sigmoid evaluation can lose numerical information at extreme logits; BCEWithLogitsLoss is preferable when logits are available.

[API: BCELoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BCELoss.html)

### BCEWithLogitsLoss: binary cross entropy on logits

For one logit $z$, the stable unweighted expression is

$$\ell(z,y)=\max(z,0)-zy+\log(1+e^{-|z|}).$$

```python
loss = nn.BCEWithLogitsLoss()(binary_logits, binary_targets)
```

It combines sigmoid and BCE in a numerically stable calculation. Do not apply sigmoid beforehand. For multilabel prediction, use logits and floating targets of shape `(N, C)`.

`pos_weight` weights positive contributions, while `weight` rescales losses more generally. Changing these changes the objective and can affect probability calibration; select decision thresholds using validation data.

**Pros:** Stable fused binary logistic objective; supports independent labels and positive-class weighting.

**Cons:** Does not enforce mutually exclusive classes; weighting and decision thresholds need validation, and missing labels require explicit masking.

[API: BCEWithLogitsLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BCEWithLogitsLoss.html)

### SoftMarginLoss: binary logistic loss with signed labels

$$\ell(z,y)=\log(1+e^{-yz}),\qquad y\in\{-1,+1\}.$$

```python
signed_labels = torch.tensor([1.0, -1.0, 1.0, -1.0])
loss = nn.SoftMarginLoss()(binary_logits.squeeze(1), signed_labels)
```

With $y_{\mathrm{binary}}=(y+1)/2$, this is the same unweighted binary logistic objective as BCE with logits. The label convention differs.

**Pros:** Smooth binary logistic objective; convenient for pipelines using signed labels.

**Cons:** Signed labels are easily confused with BCE targets; lacks the dedicated positive-class weighting interface of BCEWithLogitsLoss.

[API: SoftMarginLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.SoftMarginLoss.html)

### MultiLabelSoftMarginLoss: independent logistic labels

This applies a logistic penalty to each label and averages across classes for each sample before batch reduction.

```python
scores = torch.randn(2, 3, requires_grad=True)
multi_targets = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
loss = nn.MultiLabelSoftMarginLoss()(scores, multi_targets)
```

For ordinary unweighted mean reduction, it matches the mean BCE-with-logits objective. With `reduction="none"`, its output is per sample, whereas BCE-with-logits retains each label's loss. Prefer BCE with logits when you need `pos_weight` or explicit label-level masking.

**Pros:** Convenient independent-label logistic objective; supports class weights and per-sample unreduced losses.

**Cons:** No separate pos_weight option; label-level masking is less convenient than with BCEWithLogitsLoss, and label dependence is not explicitly modeled.

[API: MultiLabelSoftMarginLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MultiLabelSoftMarginLoss.html)

## 3. Margin losses: require separation

Margin losses can stop penalizing an example once it satisfies the required separation. Their scores need not be probabilities.

### MultiMarginLoss: multiclass hinge penalties

For one target class $y$,

$$\ell(z,y)=\frac1C\sum_{j\ne y}[\max(0,m-z_y+z_j)]^p,$$

where `p` is one or two. Every competing class should score below the correct class by at least $m$.

```python
loss = nn.MultiMarginLoss(margin=1.0, p=1)(logits, labels)
```

Use integer class indices, as with standard multiclass cross entropy. This is a margin-based classification objective rather than a likelihood model.

**Pros:** Explicitly enforces a gap against competing classes; satisfied examples stop contributing; supports linear or squared hinge penalties.

**Cons:** Does not train normalized probabilities; margin and score scale matter, and linear hinge penalties have a kink.

[API: MultiMarginLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MultiMarginLoss.html)

### MultiLabelMarginLoss: relevant classes above irrelevant ones

For a set $Y$ of relevant classes,

$$\ell(z,Y)=\frac1C\sum_{j\in Y}\sum_{k\notin Y}\max(0,1-z_j+z_k).$$

The target format is unusual: list relevant class indices first, then pad with `-1`. Do not use a binary indicator matrix.

```python
scores = torch.randn(2, 4, requires_grad=True)
index_targets = torch.tensor([[0, 2, -1, -1], [1, -1, -1, -1]])
loss = nn.MultiLabelMarginLoss()(scores, index_targets)
```

**Pros:** Directly encourages relevant classes to outrank irrelevant ones; useful for label ordering.

**Cons:** Unusual index-list targets are easy to misuse; comparisons grow with label-set size, and the fixed unit margin interacts with score scale.

[API: MultiLabelMarginLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MultiLabelMarginLoss.html)

### MarginRankingLoss: compare two scores

$$\ell(s_1,s_2,y)=\max(0,-y(s_1-s_2)+m).$$

A target of `+1` means the first score should be higher; `-1` means the second should be higher.

```python
first = torch.tensor([0.8, 0.2], requires_grad=True)
second = torch.tensor([0.3, 0.7], requires_grad=True)
order = torch.tensor([1.0, -1.0])
loss = nn.MarginRankingLoss(margin=0.5)(first, second, order)
```

Useful applications include preference learning and pairwise recommendation ranking.

**Pros:** Expresses relative preferences without absolute target scores; enforces an optional ordering gap.

**Cons:** Requires informative labeled pairs; satisfied pairs provide zero gradients, and relative constraints do not identify absolute score origins.

[API: MarginRankingLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MarginRankingLoss.html)

## 4. Probabilistic objectives: model the observation process

### PoissonNLLLoss: nonnegative count observations

For a Poisson rate $\lambda$, the negative log likelihood is

$$\ell(\lambda,y)=\lambda-y\log\lambda+\log(y!).$$

The default `log_input=True` expects $s=\log\lambda$ and computes $e^s-ys$, omitting the target-only factorial term. `full=True` adds a Stirling approximation to that term.

```python
log_rate = torch.randn(4, requires_grad=True)
counts = torch.tensor([0.0, 2.0, 1.0, 5.0])
loss = nn.PoissonNLLLoss(log_input=True)(log_rate, counts)
```

Use it when a Poisson observation model is plausible, such as event counts. Real counts may be overdispersed; a distribution-based custom objective can express a richer model.

**Pros:** Likelihood suited to count data; log-rate outputs parameterize positive rates naturally.

**Cons:** Mean-equals-variance assumptions can fail for overdispersed or zero-inflated data; extreme log rates can overflow, and omitted constants affect reported likelihood values.

[API: PoissonNLLLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.PoissonNLLLoss.html)

### GaussianNLLLoss: predict a mean and variance

For positive variance $v$,

$$\ell(\mu,v,y)=\frac12\left[\log v+\frac{(y-\mu)^2}{v}\right]+\text{constant}.$$

```python
mean = torch.randn(4, 1, requires_grad=True)
raw_variance = torch.randn(4, 1, requires_grad=True)
variance = F.softplus(raw_variance) + 1e-6
observations = torch.randn(4, 1)
loss = nn.GaussianNLLLoss(full=True)(mean, observations, variance)
```

The argument order is `(input_mean, target, variance)`. Supply variance, not standard deviation or log variance. `full=True` includes the Gaussian normalization constant. Predicted variance models observation uncertainty under the chosen Gaussian model; it does not automatically quantify all uncertainty about model parameters.

**Pros:** Learns a mean and input-dependent observation variance; balances residual size against uncertainty through a likelihood.

**Cons:** Requires positive variance and a plausible Gaussian model; very small variances can destabilize gradients, and predicted uncertainty needs independent evaluation.

[API: GaussianNLLLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GaussianNLLLoss.html)

### KLDivLoss: match distributions, with direction in mind

$$D_{\mathrm{KL}}(q\|p)=\sum_c q_c\log\frac{q_c}{p_c}.$$

PyTorch expects model **log probabilities** as `input`, and target probabilities by default. That computes $D_{\mathrm{KL}}(\text{target}\|\text{model})$.

```python
student_logits = torch.randn(4, 10, requires_grad=True)
teacher_logits = torch.randn(4, 10)
loss = nn.KLDivLoss(reduction="batchmean")(
    F.log_softmax(student_logits, dim=1),
    F.softmax(teacher_logits, dim=1),
)
```

`batchmean` divides the summed terms by batch size; `mean` divides by the number of elements. For log-space targets, use `log_target=True`. KL is asymmetric, so reversing the arguments changes the objective. Distillation with temperature also needs a deliberate scaling convention.

**Pros:** Uses a complete target distribution; supports log-space targets and distillation.

**Cons:** Direction, logarithm conventions, and reduction are easy to confuse; asymmetric and sensitive to support mismatch or invalid distributions.

[API: KLDivLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.KLDivLoss.html)

## 5. Embedding objectives: learn useful geometry

A classifier learns which label fits. An embedding model learns where samples should sit relative to one another.

### HingeEmbeddingLoss: similar pairs close, dissimilar pairs separated

For an input distance $d$,

$$
\ell(d,y)=\begin{cases}d,&y=+1,\\\max(0,m-d),&y=-1.\end{cases}
$$

```python
x1 = torch.randn(4, 8, requires_grad=True)
x2 = torch.randn(4, 8, requires_grad=True)
pair_labels = torch.tensor([1.0, -1.0, 1.0, -1.0])
distances = torch.linalg.vector_norm(x1 - x2, dim=1)
loss = nn.HingeEmbeddingLoss(margin=1.0)(distances, pair_labels)
```

The loss consumes distances, not two embedding tensors directly. This is also different from the usual signed-score hinge classification loss.

**Pros:** Simple pairwise attraction and separation; accepts externally computed differentiable distances.

**Cons:** Depends on pair selection and distance scale; negative pairs beyond the margin provide no gradients, while positive-only attraction can collapse embeddings.

[API: HingeEmbeddingLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.HingeEmbeddingLoss.html)

### CosineEmbeddingLoss: compare directions

With cosine similarity $s$,

$$
\ell(s,y)=\begin{cases}1-s,&y=+1,\\\max(0,s-m),&y=-1.\end{cases}
$$

```python
loss = nn.CosineEmbeddingLoss(margin=0.2)(x1, x2, pair_labels)
```

Positive pairs should align; negative pairs should have cosine similarity no greater than the margin. Because cosine emphasizes direction, it does not penalize scale differences in the same way as Euclidean distance.

**Pros:** Fits direction-based retrieval; largely ignores embedding magnitude and handles positive and negative pairs.

**Cons:** Cannot preserve meaningful norms by itself; near-zero vectors need care, and pairwise constraints do not guarantee useful global geometry.

[API: CosineEmbeddingLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CosineEmbeddingLoss.html)

### TripletMarginLoss: anchor, positive, negative

$$\ell(a,p,n)=\max(0,d(a,p)-d(a,n)+m).$$

```python
anchor = torch.randn(4, 8, requires_grad=True)
positive = torch.randn(4, 8, requires_grad=True)
negative = torch.randn(4, 8, requires_grad=True)
loss = nn.TripletMarginLoss(margin=1.0, p=2)(anchor, positive, negative)
```

The built-in distance uses a configurable $p$-norm with numerical stabilization. `swap=True` allows a distance-swap variant involving positive and negative samples. Choosing informative triplets matters: already-satisfied triplets provide zero loss.

**Pros:** Directly learns relative neighborhood relationships; supports configurable norm distance and optional distance swapping.

**Cons:** Triplet mining can be expensive; easy triplets provide no gradients, mislabeled hard examples can dominate, and embedding scale interacts with margin.

[API: TripletMarginLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.TripletMarginLoss.html)

### TripletMarginWithDistanceLoss: choose the distance yourself

This retains the triplet objective while accepting a custom nonnegative distance function.

```python
criterion = nn.TripletMarginWithDistanceLoss(
    distance_function=lambda a, b: 1 - F.cosine_similarity(a, b, dim=1),
    margin=0.2,
)
loss = criterion(anchor, positive, negative)
```

This is a useful starting point for geometry-aware representation learning. A custom distance must have suitable output shapes and usable gradients. Calling it a “distance” does not establish the metric axioms or make it a geodesic distance on a manifold.

**Pros:** Allows application-specific distances while retaining the triplet objective.

**Cons:** Requires a stable, nonnegative distance with suitable shapes and gradients; retains triplet-mining difficulties and supplies no automatic metric guarantees.

[API: TripletMarginWithDistanceLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.TripletMarginWithDistanceLoss.html)

## 6. CTCLoss: sequence labels without explicit alignment

Speech and text-recognition models often produce more time steps than target symbols. Connectionist Temporal Classification sums probabilities over valid alignment paths:

$$\ell=-\log\sum_{\pi\in\mathcal{B}^{-1}(y)}P(\pi\mid x).$$

The collapse operation merges adjacent repetitions and removes blank symbols.

```python
T, N, C = 12, 2, 5  # time, batch, classes including blank
sequence_logits = torch.randn(T, N, C, requires_grad=True)
log_probs = F.log_softmax(sequence_logits, dim=2)
# Concatenated targets for the two sequences; 0 is reserved as blank.
targets = torch.tensor([1, 2, 1, 3, 4], dtype=torch.long)
input_lengths = torch.tensor([12, 12], dtype=torch.long)
target_lengths = torch.tensor([3, 2], dtype=torch.long)
loss = nn.CTCLoss(blank=0)(
    log_probs, targets, input_lengths, target_lengths
)
```

Inputs are log probabilities with shape `(T, N, C)`. Repeated target symbols require extra alignment steps. `zero_infinity=True` suppresses infinite losses and their gradients, but investigate impossible alignments rather than silently relying on it.

**Pros:** Learns sequence recognition without frame-level alignment labels; sums valid alignments and handles variable lengths.

**Cons:** Assumes monotonic alignment and factorized path probabilities; short inputs can make alignment impossible, and blank handling, repeated symbols, lengths, and decoding require care.

[API: CTCLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CTCLoss.html)

## 7. AdaptiveLogSoftmaxWithLoss: large output vocabularies

This module combines a trainable output structure with negative log likelihood. It uses a shortlist and clusters to make large, unevenly distributed class vocabularies more practical.

```python
adaptive_head = nn.AdaptiveLogSoftmaxWithLoss(
    in_features=32, n_classes=1000, cutoffs=[100, 500]
)
features = torch.randn(4, 32, requires_grad=True)
targets = torch.tensor([3, 120, 700, 42], dtype=torch.long)
result = adaptive_head(features, targets)
loss = result.loss
```

Class indices should follow decreasing frequency so frequent classes use the efficient head. Include this module's parameters in the optimizer. It is a structured output model, not a drop-in criterion for already-computed full-vocabulary logits, and has no direct functional loss equivalent.

**Pros:** Can reduce training computation for large, frequency-skewed vocabularies by evaluating only needed tail clusters.

**Cons:** Requires frequency-aware indices and cutoff design; changes output parameterization, and requesting every class probability can reduce the efficiency advantage.

[API: AdaptiveLogSoftmaxWithLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.AdaptiveLogSoftmaxWithLoss.html)

## Reduction: what does “mean loss” actually mean?

Most core losses support `none`, `sum`, and `mean`. However, the denominator is part of the API contract:

- MSE and L1 typically average across all tensor elements.
- Weighted cross entropy with index targets normalizes by the sum of contributing target-class weights.
- KL's `batchmean` divides by batch size.
- CTC's mean reduction normalizes each sample loss by its target length before averaging the batch.

Loss magnitudes from different objectives are therefore not automatically comparable.

For padding or missing observations, an explicit masked reduction can help:

```python
prediction = torch.randn(2, 3, requires_grad=True)
target = torch.randn(2, 3)
mask = torch.tensor([[True, True, False], [True, False, False]])
per_element = F.mse_loss(prediction, target, reduction="none")
valid = mask.to(per_element.dtype)
loss = (per_element * valid).sum() / valid.sum().clamp_min(1)
```

This assumes all targets are finite: multiplying a NaN by zero still produces NaN. Sanitize missing values before evaluating the elementwise loss.

## Shape errors that quietly change the objective

For regression, check shape equality. A prediction shaped `(N, 1)` and a target shaped `(N,)` may broadcast to `(N, N)` rather than compare one prediction with one observation.

```python
assert prediction.shape == target.shape
```

For standard classification, use these conventions:

| Task | Model output | Target | Typical loss |
|---|---|---|---|
| Multiclass classification | `(N, C)` logits | `(N,)` long indices | Cross entropy |
| Multiclass segmentation | `(N, C, H, W)` logits | `(N, H, W)` long indices | Cross entropy |
| Binary classification | `(N, 1)` logits | `(N, 1)` floats | BCE with logits |
| Multilabel classification | `(N, C)` logits | `(N, C)` floats | BCE with logits |
| Continuous regression | `(N, D)` values | `(N, D)` floats | MSE, L1, Huber |

Do not apply `argmax` before a training loss: it discards the differentiable scores needed to train the classifier.

## Choosing a loss for MNIST

For an ordinary ten-class MNIST classifier, start with cross entropy:

```python
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# images: (N, 1, 28, 28); labels: (N,), dtype=torch.long
# Inside the training loop:
# optimizer.zero_grad(set_to_none=True)
# logits = model(images)
# loss = criterion(logits, labels)
# loss.backward()
# optimizer.step()
```

Use the training set to fit parameters, a validation set to choose configurations, and the held-out test set for final evaluation. Report accuracy alongside loss: they answer different questions. A prediction can keep the same winning class while its confidence changes considerably.

To explore representation learning, keep MNIST but change the experiment: learn digit embeddings with cosine or triplet objectives, then evaluate retrieval or a downstream classifier. The dataset stays the same; the meaning of a good prediction changes.

## Beyond the core list

Focal, Dice, IoU, contrastive, perceptual, and domain-specific objectives are common, but they are not all standalone loss modules in the core list above. For example, TorchVision supplies focal and several bounding-box IoU losses. Other objectives come from specialist libraries or custom implementations.

For probability models outside the built-in NLL criteria, a distribution's `log_prob` can define a negative log likelihood. For regularization, differentiable penalties can be added to a task objective:

$$\mathcal{L}=\mathcal{L}_{\mathrm{task}}+\lambda\mathcal{R}.$$

Choose normalization and $\lambda$ deliberately. A term averaged across pixels and one summed across parameters can differ greatly in scale.

## A practical selection guide

| What should the model learn? | Starting point |
|---|---|
| Continuous values with strong large-error penalties | `MSELoss` |
| Continuous values with less outlier influence | `L1Loss` or `HuberLoss` |
| One class per example | `CrossEntropyLoss` |
| Binary or independent label probabilities | `BCEWithLogitsLoss` |
| Count rates under a Poisson model | `PoissonNLLLoss` |
| Gaussian mean and observation variance | `GaussianNLLLoss` |
| Match a teacher's categorical distribution | `KLDivLoss` |
| Order pairs of scores | `MarginRankingLoss` |
| Align or separate embedding directions | `CosineEmbeddingLoss` |
| Place positives closer than negatives | Triplet losses |
| Recognize sequences with unknown alignment | `CTCLoss` |
| Predict a large, frequency-skewed vocabulary | `AdaptiveLogSoftmaxWithLoss` |

My starting point is the prediction contract: what does the output represent, what does the target encode, and which errors matter? Once those are clear, selecting the loss becomes a modeling decision rather than a search for the most sophisticated API name.

## References and version notes

- [PyTorch 2.14: torch.nn loss functions](https://docs.pytorch.org/docs/2.14/nn.html#loss-functions)
- [PyTorch 2.14: functional loss functions](https://docs.pytorch.org/docs/2.14/nn.functional.html#loss-functions)
- [TorchVision: sigmoid focal loss](https://docs.pytorch.org/vision/stable/generated/torchvision.ops.sigmoid_focal_loss.html)
- [PyTorch probability distributions](https://docs.pytorch.org/docs/2.14/distributions.html)

Examples illustrate input contracts; they are not benchmark results. Check `torch.__version__` against the linked API version when reproducing them. No runtime validation of these snippets was performed for this post.
