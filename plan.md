# FLAG: CFM Projection Error, Reward Drift, and Guidance Buffer Quantification Plan

## 1. Goal

This experiment is designed to empirically examine the two approximation terms that appear in the FLAG monotonic-improvement analysis:

1. **CFM projection fidelity**
   - How accurately does the practical CFM update realize the ideal moment-matching/KL projection target?
2. **Reward drift**
   - How much does the cross-entropy-augmented reward change after one policy update?
3. **Guidance buffer effect**
   - Does reusing stored guidance targets reduce the current projection residual?
   - When does target staleness begin to offset the benefit of repeated supervision?

### 1.1 Reviewer context and scope

The reviewer's concern is that the paper reads as if Theorem 4.5 directly certifies monotonic policy improvement for the neural-network implementation. These experiments therefore do **not** attempt to prove monotonic improvement of the full practical algorithm. They only check, empirically, to what extent the approximation conditions used in the analysis hold during actual training: CFM projection fidelity, the guidance-buffer effect on the projection residual, and the size of the policy-induced reward drift.

Target claim:

> The practical CFM update closely realizes the ideal projection target, while the guidance buffer reduces projection residuals and the policy-induced reward drift remains controlled during training.

### 1.2 Measurability caveat

The theorem's $\epsilon_k^{\mathrm{proj}}$ is defined per EM iteration against an ideal KL projection. The practical algorithm distills buffer targets across many interleaved actor updates, so no single logged number equals $\epsilon_k^{\mathrm{proj}}$; every quantity below is a per-update or periodic proxy. Phase 0 (Section 4) first verifies on the default setting that these proxies produce meaningful, interpretable signals before the full ablation is launched.

The main training policy always follows the practical CFM update. Any KL-based update is used only as a detached diagnostic branch and is discarded immediately after measurement.

---

## 2. Notation

At policy iteration $k$,

$$
\hat s=(s,z),
\qquad
\mu_k^\star(\hat s)=\mathbb E_{q_k(a\mid \hat s)}[a],
$$

and the ideal moment-matching projection is

$$
\theta_{k+1}^{\mathrm{ideal}}
\in
\arg\min_\theta
\mathbb E_{\hat s}
\left[
\left\|
T_\theta(s,z)-\mu_k^\star(\hat s)
\right\|_2^2
\right].
$$

The practical FLAG update instead trains $T_\theta$ using the CFM loss.

The augmented reward is

$$
\hat r_k(s,a)
=
r(s,a)-\alpha_k\log\tilde\pi_{\theta_k}(a\mid s).
$$

---

# Part I. Training Runs

## 3. Main ablation conditions

Guidance-buffer sizes, matching the paper's Table 3 ablation:

| Condition | Buffer size |
|---|---|
| No-buffer | 0 |
| Default | 10.24k |
| Large-1 | 51.2k |
| Large-2 | 102.4k |
| Large-3 | 204.8k |

All conditions must use the same:

- environment steps,
- actor update frequency,
- CFM optimizer steps per actor update,
- CFM mini-batch size,
- critic architecture and update budget,
- number of Q evaluations,
- covariance schedule,
- random seeds where possible.

The no-buffer run must not receive fewer gradient steps. It should reuse the current fresh guidance batch for the same number of optimizer steps as the buffer condition.

---

## 4. Task, seeds, and phased execution

- Task: DMC Dog-run only.
- Seeds: 5 per condition (same as the paper's Table 3 protocol).
- Evaluation protocol identical to the main paper.
- Compute estimate: 5 conditions × 5 seeds = 25 runs × ~5 GPU-h (1M steps, L40S) ≈ 125 GPU-h, plus Phase 0.

### Phase 0 — instrumentation validation (run first)

One run (1 seed) on the default setting (Dog-run, 10.24k buffer) with all Part II–V logging enabled. Purpose: confirm that the per-update residuals (Section 5), reward drift (Section 6), and especially the stored-target residuals (Section 7) yield interpretable values, given the caveat in Section 1.2. Metrics that turn out uninformative are adjusted or dropped here, before any ablation compute is spent.

### Phase 1 — buffer-size ablation

5 conditions × 5 seeds on Dog-run, with the logging set finalized in Phase 0.

---

# Part II. Per-Update Logging

## 5. Current training-target residual

For the exact guidance batch used in actor update $k$, record the endpoint error before and after the CFM update.

### 5.1 Before-update residual

$$
R_{k,\mathrm{train}}^{\mathrm{before}}
=
\frac{1}{B}
\sum_{i=1}^{B}
\left\|
T_{\theta_k}(s_i,z_i)
-
\hat\mu_{k,i}^{\star}
\right\|_2^2.
$$

### 5.2 After-update residual

$$
R_{k,\mathrm{train}}^{\mathrm{after}}
=
\frac{1}{B}
\sum_{i=1}^{B}
\left\|
T_{\theta_{k+1}}(s_i,z_i)
-
\hat\mu_{k,i}^{\star}
\right\|_2^2.
$$

Here $\hat\mu_{k,i}^{\star}$ is the target actually used by training.

### 5.3 Projection realization ratio

$$
\mathrm{Realization}_k
=
1-
\frac{
R_{k,\mathrm{train}}^{\mathrm{after}}
}{
R_{k,\mathrm{train}}^{\mathrm{before}}+\varepsilon
}.
$$

Interpretation:

- close to $1$: the target displacement is almost fully realized,
- close to $0$: the update makes little progress,
- negative: the updated flow endpoint moves farther from the target.

### 5.4 KL-scaled projection residual

For isotropic covariance $\Sigma_k=\sigma_k^2 I$,

$$
R_{k,\mathrm{train}}^{\mathrm{KL}}
=
\frac{
R_{k,\mathrm{train}}^{\mathrm{after}}
}{
2\sigma_k^2
}.
$$

This is proportional to the excess state-wise KL projection objective.

### Logging frequency

Record all four quantities at every actor update.

---

## 6. Per-update reward drift

Use exactly the same state-action pairs before and after the policy update.

Before updating the actor:

1. sample a probe batch $\{(s_i,a_i)\}_{i=1}^B$,
2. evaluate and detach
   $$
   \ell_{k,i}
   =
   \log\tilde\pi_{\theta_k}(a_i\mid s_i).
   $$

After the actor update, evaluate

$$
\ell_{k+1,i}
=
\log\tilde\pi_{\theta_{k+1}}(a_i\mid s_i).
$$

### 6.1 Policy-induced reward drift

Hold $\alpha_k$ fixed:

$$
D_k^{r,\mathrm{policy}}
=
\frac{\alpha_k}{B}
\sum_{i=1}^{B}
\left|
\ell_{k+1,i}-\ell_{k,i}
\right|.
$$

### 6.2 Full practical reward drift

If $\alpha$ is updated during training:

$$
D_k^{r,\mathrm{full}}
=
\frac{1}{B}
\sum_{i=1}^{B}
\left|
-\alpha_{k+1}\ell_{k+1,i}
+
\alpha_k\ell_{k,i}
\right|.
$$

### 6.3 Distributional summaries

Record:

$$
\mathrm{mean},
\qquad
\mathrm{median},
\qquad
\mathrm{p95},
\qquad
\mathrm{empirical\ max}.
$$

The empirical maximum must not be described as the true $\ell_\infty$ norm.

### 6.4 Parameter displacement

Also record

$$
\Delta\theta_k
=
\left\|
\theta_{k+1}-\theta_k
\right\|_2,
$$

or preferably a normalized form,

$$
\Delta\theta_k^{\mathrm{rel}}
=
\frac{
\|\theta_{k+1}-\theta_k\|_2
}{
\|\theta_k\|_2+\varepsilon
}.
$$

This allows checking whether reward drift scales approximately linearly with the actual update magnitude.

Decision: keep this logging. It costs one norm per actor update and provides the scale reference for "reward drift is small relative to the update magnitude." The step-size scaling shadow test that previously accompanied it has been removed from this plan; if $\Delta\theta_k$ ends up unused in the final figures, drop it as well.

### Logging frequency

Record reward drift and parameter displacement at every actor update.

---

## 7. Lightweight guidance-buffer logging

Caveat (see Section 1.2): buffer targets are distilled over multiple actor updates, so the stored-target residuals below mix fitting progress with target staleness and are not a per-iteration $\epsilon_k^{\mathrm{proj}}$. Treat them as diagnostics; Phase 0 decides whether they are informative enough to report.

Each guidance-buffer entry should additionally store:

```text
state s
latent z
target mu_star
creation actor-update index
creation environment step
sigma at creation
alpha at creation
optional: number of times sampled
```

For every actor update, record:

- mean age of sampled guidance entries,
- median age,
- p95 age,
- fraction of the batch generated at the current iteration,
- number of times each sampled entry has previously been reused,
- stored-target fitting residual before update,
- stored-target fitting residual after update.

For a sampled buffer mini-batch $\mathcal G_k$,

$$
E_{k,\mathrm{stored}}^{\mathrm{before}}
=
\frac{1}{|\mathcal G_k|}
\sum_{j\in\mathcal G_k}
\left\|
T_{\theta_k}(s_j,z_j)-\mu_j^\star
\right\|_2^2,
$$

$$
E_{k,\mathrm{stored}}^{\mathrm{after}}
=
\frac{1}{|\mathcal G_k|}
\sum_{j\in\mathcal G_k}
\left\|
T_{\theta_{k+1}}(s_j,z_j)-\mu_j^\star
\right\|_2^2.
$$

These values show whether repeated buffer reuse actually improves fitting to the stored target.

They do **not** by themselves establish that the current target is well matched, because the stored target may be stale.

---

# Part III. Periodic Probe Evaluation

## 8. Current-target projection residual

Every fixed interval, sample a fresh held-out probe set from the current replay-buffer distribution:

$$
\mathcal P_k=\{(s_i,z_i)\}_{i=1}^{B_{\mathrm{probe}}}.
$$

Using the current critic, policy, $\alpha_k$, and $\sigma_k$, recompute a high-accuracy reference target

$$
\mu_k^{\star,\mathrm{ref}}(s_i,z_i)
$$

with

$$
N_{\mathrm{ref}}
\gg
N_{\mathrm{train}}.
$$

Recommended values:

- $B_{\mathrm{probe}}=256$–$1024$,
- $N_{\mathrm{ref}}=512$–$2048$.

Then compute

$$
E_k^{\mathrm{current}}
=
\frac{1}{B_{\mathrm{probe}}}
\sum_i
\left\|
T_{\theta_k}(s_i,z_i)
-
\mu_k^{\star,\mathrm{ref}}(s_i,z_i)
\right\|_2^2,
$$

and

$$
E_k^{\mathrm{current,KL}}
=
\frac{
E_k^{\mathrm{current}}
}{
2\sigma_k^2
}.
$$

This is the main metric for testing the guidance-buffer claim.

Desired result:

$$
E_{k,\mathrm{default}}^{\mathrm{current}}
<
E_{k,\mathrm{no\ buffer}}^{\mathrm{current}}
$$

over a meaningful portion of training.

### Evaluation frequency

Every:

- 500–2000 actor updates, or
- 1k–5k environment steps.

Choose the largest frequency that is computationally affordable.

---

## 9. Fixed and moving probe sets

Use both:

### 9.1 Fixed probe set

A fixed set constructed early in training.

Purpose:

- compare all checkpoints on identical inputs,
- obtain a clean temporal curve.

Limitation:

- may become off-distribution later in training.

### 9.2 Current-distribution probe set

Resample from the current replay buffer at each evaluation point.

Purpose:

- measure projection fidelity where the current policy actually operates.

Report these separately.

---

# Part IV. Guidance Buffer Mechanism Analysis

## 10. Fitting-versus-staleness decomposition

For a buffer item generated at iteration $j$ and evaluated at iteration $k$, define its age as

$$
\tau=k-j.
$$

For sampled buffer entries, recompute the current reference target

$$
\mu_k^{\star,\mathrm{ref}}(s_j,z_j).
$$

### 10.1 Stored-target fitting error

$$
E_{\mathrm{fit}}(\tau)
=
\mathbb E_{j:\,k-j=\tau}
\left[
\left\|
T_{\theta_k}(s_j,z_j)
-
\mu_j^\star
\right\|_2^2
\right].
$$

This measures the benefit of repeated supervision.

### 10.2 Target staleness

$$
E_{\mathrm{stale}}(\tau)
=
\mathbb E_{j:\,k-j=\tau}
\left[
\left\|
\mu_j^\star
-
\mu_k^{\star,\mathrm{ref}}(s_j,z_j)
\right\|_2^2
\right].
$$

This measures how outdated the stored target has become.

### 10.3 Current projection error on stored states

$$
E_{\mathrm{current}}(\tau)
=
\mathbb E_{j:\,k-j=\tau}
\left[
\left\|
T_{\theta_k}(s_j,z_j)
-
\mu_k^{\star,\mathrm{ref}}(s_j,z_j)
\right\|_2^2
\right].
$$

By the triangle inequality,

$$
E_{\mathrm{current}}(\tau)
\le
2E_{\mathrm{fit}}(\tau)
+
2E_{\mathrm{stale}}(\tau).
$$

Expected qualitative behavior:

- 0 (no buffer): no repeated fitting benefit, highest current projection residual;
- 10.24k (default): low fitting error, moderate staleness, lowest current projection residual — expected best, matching the return ordering in paper Table 3;
- 51.2k–204.8k: stored-target fitting error stays low, but staleness grows with buffer size; the current projection residual is expected to rise again, mirroring the return degradation at 102.4k and 204.8k.

### Recommended age bins

Use logarithmic or coarse bins, for example:

```text
0
1–4
5–16
17–64
65–256
>256 actor updates
```

---

## 11. Target-estimation noise floor

To distinguish CFM error from SNIS target-estimation noise, independently estimate the current target twice:

$$
\mu_{k,A}^{\star,\mathrm{ref}},
\qquad
\mu_{k,B}^{\star,\mathrm{ref}}.
$$

Define

$$
E_k^{\mathrm{SNIS}}
=
\frac{1}{2}
\mathbb E
\left[
\left\|
\mu_{k,A}^{\star,\mathrm{ref}}
-
\mu_{k,B}^{\star,\mathrm{ref}}
\right\|_2^2
\right].
$$

Plot this as a noise-floor reference.

Also report effective sample size:

$$
\mathrm{ESS}
=
\frac{1}{
\sum_i \bar w_i^2
}.
$$

---

# Part V. Sparse KL Shadow Evaluation

## 12. Purpose

The endpoint residual directly measures how well the CFM policy matches the ideal moment target.

However, the theorem defines practical projection error relative to an ideal KL-gradient iterate. To measure a closer empirical analogue, periodically construct a shadow KL branch.

This branch must never affect the main training trajectory.

Scope: run this shadow evaluation only in Phase 0 on the default condition. Do not run it across the Phase 1 ablation grid.

---

## 13. Shadow-branch procedure

At evaluation iteration $k$:

1. freeze
   $$
   \theta_k,\ Q_k,\ \alpha_k,\ \sigma_k;
   $$
2. construct one common held-out batch;
3. compute one common high-sample target
   $$
   \mu_k^{\star,\mathrm{ref}};
   $$
4. copy the policy into two detached branches.

### 13.1 KL branch

Perform the one-step ideal KL/moment-projection gradient update:

$$
\theta_{k+1}^{\mathrm{KL}}
=
\theta_k
-
\beta
\nabla_\theta
\mathbb E_{\hat s}
\left[
\frac{
\|T_\theta(\hat s)-\mu_k^{\star,\mathrm{ref}}(\hat s)\|_2^2
}{
2\sigma_k^2
}
\right]_{\theta=\theta_k}.
$$

### 13.2 CFM branch

Perform the actual FLAG CFM update using the same target batch:

$$
\theta_{k+1}^{\mathrm{CFM}}.
$$

### 13.3 Projection objectives

$$
R_k^{\mathrm{KL}}
=
\mathbb E_{\hat s}
\left[
\frac{
\|T_{\theta_{k+1}^{\mathrm{KL}}}(\hat s)
-\mu_k^{\star,\mathrm{ref}}(\hat s)\|_2^2
}{
2\sigma_k^2
}
\right],
$$

$$
R_k^{\mathrm{CFM}}
=
\mathbb E_{\hat s}
\left[
\frac{
\|T_{\theta_{k+1}^{\mathrm{CFM}}}(\hat s)
-\mu_k^{\star,\mathrm{ref}}(\hat s)\|_2^2
}{
2\sigma_k^2
}
\right].
$$

### 13.4 Empirical excess projection gap

$$
\Delta_k^{\mathrm{proj}}
=
R_k^{\mathrm{CFM}}
-
R_k^{\mathrm{KL}}.
$$

Also report

$$
\Delta_k^{\mathrm{proj},+}
=
\max(0,\Delta_k^{\mathrm{proj}})
$$

only as a nonnegative visualization; retain the signed quantity in the raw logs.

### 13.5 Endpoint discrepancy

$$
D_k^{\mathrm{KL-CFM}}
=
\mathbb E_{\hat s}
\left[
\left\|
T_{\theta_{k+1}^{\mathrm{CFM}}}(\hat s)
-
T_{\theta_{k+1}^{\mathrm{KL}}}(\hat s)
\right\|_2^2
\right].
$$

With identical local covariance, this is the conditional squared $W_2$ distance between the two local Gaussian policies.

### Evaluation frequency

Every 100–500 actor updates, or more sparsely if expensive.

After logging, delete the KL branch. Continue training only with the actual CFM branch.

---

# Part VI. Optional Drift Diagnostics

## 14. E-step target drift

At sparse checkpoints, estimate

$$
\|q_{k+1}(\cdot\mid\hat s)-q_k(\cdot\mid\hat s)\|_1
$$

using a common candidate-action set.

For each probe state, sample candidate actions from a mixture proposal covering both iterations and compute normalized weights

$$
\bar w_i^k,
\qquad
\bar w_i^{k+1}.
$$

Then estimate

$$
D_k^{q,L_1}
=
\mathbb E_{\hat s}
\left[
\sum_i
\left|
\bar w_i^{k+1}-\bar w_i^k
\right|
\right].
$$

Use identical actions for both weight computations.


---

# Part VII. Recommended Logging Schedule

| Metric | Frequency |
|---|---:|
| Training target residual before/after | Every actor update |
| Projection realization ratio | Every actor update |
| KL-scaled training residual | Every actor update |
| Reward drift mean/median/p95/max | Every actor update |
| Parameter update norm | Every actor update |
| $\sigma_k,\alpha_k,\mathrm{ESS}_k$ | Every actor update |
| Sampled buffer-item age statistics | Every actor update |
| Stored-target fitting residual | Every actor update |
| Current-target high-sample residual | Every 500–2000 actor updates |
| Buffer fitting/staleness decomposition | Every 500–2000 actor updates |
| KL-vs-CFM shadow comparison | Phase 0 only, every 100–500 actor updates |
| E-step target drift (optional) | Every 1k–5k environment steps |

---

# Part VIII. Implementation Skeleton

Pseudocode only — the actual codebase is JAX/`flax.nnx`; `torch.no_grad()` corresponds to computing with stopped gradients outside the jitted update.

```python
for actor_update_idx in range(num_actor_updates):
    # ---------------------------------------------------------
    # 1. Build current guidance target and/or sample buffer
    # ---------------------------------------------------------
    fresh_batch = sample_replay_batch()
    fresh_target = compute_snis_target(
        fresh_batch,
        num_samples=N_train,
    )

    guidance_buffer.add(
        state=fresh_batch.state,
        latent=fresh_batch.z,
        target=fresh_target,
        creation_update=actor_update_idx,
        sigma=current_sigma,
        alpha=current_alpha,
    )

    guidance_batch = guidance_buffer.sample(batch_size)

    # ---------------------------------------------------------
    # 2. Save common probe quantities before actor update
    # ---------------------------------------------------------
    with torch.no_grad():
        train_endpoint_before = flow_endpoint(
            guidance_batch.state,
            guidance_batch.z,
        )
        train_residual_before = mse(
            train_endpoint_before,
            guidance_batch.target,
        )

        probe_state, probe_action = sample_common_reward_probe()
        old_log_prob = flow_log_prob(
            probe_state,
            probe_action,
        ).detach()

        old_params = snapshot_flat_parameters(actor)

    # ---------------------------------------------------------
    # 3. Main practical CFM update
    # ---------------------------------------------------------
    cfm_loss = compute_cfm_loss(guidance_batch)
    actor_optimizer.zero_grad()
    cfm_loss.backward()
    actor_optimizer.step()

    # ---------------------------------------------------------
    # 4. Lightweight after-update logging
    # ---------------------------------------------------------
    with torch.no_grad():
        train_endpoint_after = flow_endpoint(
            guidance_batch.state,
            guidance_batch.z,
        )
        train_residual_after = mse(
            train_endpoint_after,
            guidance_batch.target,
        )

        realization = 1.0 - (
            train_residual_after
            / (train_residual_before + eps)
        )

        new_log_prob = flow_log_prob(
            probe_state,
            probe_action,
        )

        reward_drift_policy = current_alpha * abs(
            new_log_prob - old_log_prob
        )

        new_params = snapshot_flat_parameters(actor)
        parameter_displacement = l2_norm(
            new_params - old_params
        )

    log_light_metrics(...)

    # ---------------------------------------------------------
    # 5. Periodic current-target probe
    # ---------------------------------------------------------
    if actor_update_idx % current_probe_interval == 0:
        probe = sample_heldout_current_distribution_batch()
        current_target_ref = compute_snis_target(
            probe,
            num_samples=N_ref,
            evaluation_rng=True,
        )

        with torch.no_grad():
            current_endpoint = flow_endpoint(
                probe.state,
                probe.z,
            )
            current_projection_error = mse(
                current_endpoint,
                current_target_ref,
            )

        log_current_probe(...)

    # ---------------------------------------------------------
    # 6. Periodic buffer age/staleness analysis
    # ---------------------------------------------------------
    if actor_update_idx % staleness_interval == 0:
        age_stratified_items = guidance_buffer.sample_by_age_bins()

        current_targets = recompute_current_reference_targets(
            age_stratified_items.state,
            age_stratified_items.z,
            num_samples=N_ref,
        )

        with torch.no_grad():
            endpoints = flow_endpoint(
                age_stratified_items.state,
                age_stratified_items.z,
            )

            fit_error = squared_error(
                endpoints,
                age_stratified_items.stored_target,
            )

            stale_error = squared_error(
                age_stratified_items.stored_target,
                current_targets,
            )

            current_error = squared_error(
                endpoints,
                current_targets,
            )

        log_by_age_bin(...)

    # ---------------------------------------------------------
    # 7. Sparse KL shadow branch
    # ---------------------------------------------------------
    if actor_update_idx % kl_shadow_interval == 0:
        run_detached_kl_vs_cfm_shadow_evaluation(...)
```

---

# Part IX. Figures for the Paper

## Figure 1. Current projection residual across training

x-axis:

$$
\text{environment steps}
$$

y-axis:

$$
E_k^{\mathrm{current,KL}}
$$

curves: one per buffer size, $\{0,\ 10.24\text{k},\ 51.2\text{k},\ 102.4\text{k},\ 204.8\text{k}\}$.

This is the main evidence that the guidance buffer reduces the practical projection residual.

---

## Figure 2. Projection realization and reward drift

Two separate panels:

1. projection realization ratio,
2. reward drift mean and p95.

Show seed-level confidence intervals.

---

## Figure 3. Buffer mechanism by target age

x-axis:

$$
\tau=\text{target age}
$$

curves:

$$
E_{\mathrm{fit}}(\tau),
\qquad
E_{\mathrm{stale}}(\tau),
\qquad
E_{\mathrm{current}}(\tau).
$$

This figure explains why a moderate buffer can outperform both no buffer and an excessively large buffer.

---

## Figure 4. Practical CFM versus KL shadow update

x-axis:

$$
\text{environment steps}
$$

y-axis:

$$
R_k^{\mathrm{CFM}},
\qquad
R_k^{\mathrm{KL}},
\qquad
\Delta_k^{\mathrm{proj}}.
$$

Use sparse markers because this diagnostic is not run every actor update. Default condition only (Phase 0).

---

## Figure 5. Return versus current projection residual

Scatter over seeds and checkpoints:

$$
E_k^{\mathrm{current}}
\quad\text{versus}\quad
\text{evaluation return}.
$$

Report correlation as descriptive evidence, not causality.

---

# Part X. Success Criteria

The empirical story is strong if the following are observed:

1. The CFM after-update residual is consistently lower than the before-update residual.
2. The 10.24k buffer produces lower current-target projection error than no buffer.
3. Larger buffers (51.2k–204.8k) show low stored-target fitting error but increasing target staleness.
4. The current projection residual is lowest near the default size and rises again for the largest buffers, mirroring the return ordering in paper Table 3.
5. Practical CFM residual remains close to the KL shadow residual (Phase 0).
6. Reward drift remains small relative to the policy-update magnitude.
7. Lower projection residual is associated with stronger return, at least descriptively.

---

# Part XI. Claims That Are Safe to Make

Recommended:

> We empirically quantify how accurately the practical CFM update realizes the ideal moment-matching projection. Under the fixed isotropic covariance used by FLAG, the endpoint residual is proportional to the excess state-wise KL projection objective.

> The guidance buffer reduces the current-target projection residual by repeatedly supervising recent improved targets, while excessively old targets introduce measurable staleness.

> The measured reward drift remains small and scales approximately with the policy-update magnitude, supporting the first-order approximation used in the analysis.

Avoid:

> We directly verify the exact value-level $\epsilon_k^{\mathrm{proj}}$ in the theorem.

> Small projection residual proves monotonic improvement of the implemented neural algorithm.

> The guidance buffer guarantees that all sufficient conditions of the theorem hold.

A safe interpretation is:

> The experiments empirically support the two design assumptions underlying the analysis, rather than certifying end-to-end monotonic improvement of the practical implementation.

---

# Part XII. Minimal Version Under Limited Compute

This plan is already the compute-limited version (one task, 5 seeds). If it must shrink further, cut in this order:

1. drop the 51.2k and 204.8k conditions (keep 0 / 10.24k / 102.4k),
2. reduce the buffer fitting/staleness decomposition to a few checkpoints,
3. drop the KL shadow branch (keep only the endpoint residuals of Section 5),
4. drop the E-step target drift diagnostic.

Per-update residual and reward-drift logging (Sections 5–6) are cheap and are never cut.
