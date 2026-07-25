# Phase 1: Dog-run에서 FLAG update mechanism과 $N$의 영향 검증

## 1. 실험 목적과 구성

Dog-run에서 $N_{\mathrm{train}}\in\{2,4,8,16,32\}$를 각각 독립적으로 학습한다.

- 각 $N$에 대해 seeds $0,1,2,3,4$를 학습한다.
- 전체 학습 run 수는 $5\ N\text{-values}\times5\ \text{seeds}=25$개다.
- 각 run은 1M environment steps 동안 학습한다.
- Evaluation은 10k steps마다 5 episodes로 수행한다.
- Checkpoint는 100k steps마다 저장하고 1M-step final checkpoint를 반드시 포함한다.
- Checkpoint 경로와 W&B run name에는 `N`과 `seed`를 모두 포함한다.
- 무거운 KL/BPTT 진단은 seeds $0,1,2$에서만 수행한다.
- $N$ 이외의 hyperparameter, 초기화 방식, 환경 설정과 evaluation protocol은 동일하게 유지한다.
- Phase 1에서는 no-buffer 대조군을 추가하지 않는다.

검증할 질문은 다음 세 가지다.

1. 실제 actor update의 reward/log-probability drift가 이론에서 예상한 local drift와 일치하는가?
2. CFM update가 fresh/stored target에 대한 projection residual을 실제로 감소시키는가?
3. $N>8$에서 추가 local samples가 새로운 개선 방향을 거의 제공하지 않아 성능이 plateau하는가?

## 2. 학습 중 기록할 지표

### 2.1 Reward drift

동일한 state, action과 flow tangent를 actor update 직전과 직후에 평가한다.

$$
D_{\log\pi}
=
\mathbb E\left[
\left|
\log\pi_{\theta_{k+1}}(a\mid s)
-
\log\pi_{\theta_k}(a\mid s)
\right|
\right],
$$

$$
D_{\mathrm{policy}}
=
\alpha_kD_{\log\pi},
$$

$$
D_{\mathrm{full}}
=
\mathbb E\left[
\left|
-\alpha_{k+1}\log\pi_{\theta_{k+1}}(a\mid s)
+\alpha_k\log\pi_{\theta_k}(a\mid s)
\right|
\right].
$$

각 quantity의 mean, median, p95와 함께 다음 parameter-update 크기를 기록한다.

$$
\|\Delta\theta_k\|_2,
\qquad
\frac{\|\Delta\theta_k\|_2}
{\|\theta_k\|_2+\varepsilon}.
$$

$D_{\log\pi}$와 parameter-step 크기의 관계를 함께 제시하여 실제 update가 local한지 확인한다. $D_{\mathrm{policy}}$는 이론에서 사용하는 fixed-$\alpha_k$ drift에 대응하고, $D_{\mathrm{full}}$은 실제 $\alpha$ 변화까지 포함한 regularization-term drift로 구분한다.

### 2.2 Fresh-target CFM projection

Actor loss 내부에서 실제로 생성된 동일한 fresh target $\hat\mu_{k,i}^{\star}$를 사용한다.

$$
R_{k,\mathrm{fresh}}^{\mathrm{before/after}}
=
\frac1B\sum_i
\left\|
T_{\theta_{k/k+1}}(s_i,z_i)
-
\hat\mu_{k,i}^{\star}
\right\|_2^2.
$$

Fresh target 생성 직후 $R^{\mathrm{before}}$를 계산하고, 해당 optimizer mini-step 직후이면서 다음 mini-step 이전에 동일한 $(s,z,\hat\mu^\star)$로 $R^{\mathrm{after}}$를 계산한다.

$$
\Delta R_{\mathrm{fresh}}
=
R_{\mathrm{fresh}}^{\mathrm{before}}
-
R_{\mathrm{fresh}}^{\mathrm{after}},
$$

$$
\mathrm{Realization}_{\mathrm{fresh}}
=
1-
\frac{R_{\mathrm{fresh}}^{\mathrm{after}}}
{R_{\mathrm{fresh}}^{\mathrm{before}}+\varepsilon},
$$

$$
R_{\mathrm{fresh}}^{\mathrm{KL,after}}
=
\mathbb E\left[
\frac12
\left\|
\frac{T_{\theta_{k+1}}-\hat\mu^\star}{\sigma}
\right\|_2^2
\right].
$$

$R^{\mathrm{after}}<R^{\mathrm{before}}$인 update의 비율도 기록한다.

각 quantity의 해석은 다음처럼 구분한다.

- 감소 비율: target 방향으로 이동한 update의 비율
- Realization: 한 update에서 residual이 감소한 상대적 크기
- $R^{\mathrm{after}}$: update 이후 target과의 절대적인 거리

따라서 감소 비율이 99%라면 “99% projection realization”이 아니라 “99%의 update에서 target에 더 가까워졌다”고 표현한다.

### 2.3 Stored-target projection

현재 코드가 계산하는 기존 metric은 다음 이름으로 유지한다.

$$
R_{k,\mathrm{stored}}^{\mathrm{before/after}}
=
\frac1B\sum_j
\left\|
T_{\theta_{k/k+1}}(s_j,z_j)
-
\hat\mu_j^\star
\right\|_2^2.
$$

Fresh metric과 동일하게 다음을 기록한다.

- $R_{\mathrm{stored}}^{\mathrm{before/after}}$
- $\Delta R_{\mathrm{stored}}$
- $\mathrm{Realization}_{\mathrm{stored}}$
- Move-closer update 비율
- 평균 buffer reuse count

Age bin은 사용하지 않는다. No-buffer 대조군이 없으므로 buffer의 인과적 효과가 아니라 다음 mechanism evidence로만 해석한다.

> Repeated supervision improves fitting to sampled stored targets.

### 2.4 Importance-sampling mechanism

Candidate의 augmented value를 다음과 같이 정의한다.

$$
F_i
=
Q(s,u_i)-\alpha\log\tilde\pi(u_i\mid s),
$$

$$
A_i=F_i-F_{\mathrm{anchor}}.
$$

기존 target 생성 과정에서 계산한 candidate Q와 log-probability를 재사용한다. 학습 중에는 다음 네 가지 값만 기록한다.

#### Improving-candidate hit rate

$$
P_{\mathrm{hit}}(N)
=
\Pr\left(\max_{i=1,\ldots,N}A_i>0\right).
$$

#### Candidate categorical composition

$$
p_{\mathrm{better}}
=
\frac1N\sum_i\mathbf 1[A_i>0],
\qquad
p_{\mathrm{non\text{-}improving}}
=
1-p_{\mathrm{better}}.
$$

이를 `better / non-improving` stacked bar로 표시한다. 실제 IS weight는 거의 one-hot이므로 weight entropy 분석에는 사용하지 않는다.

#### Target augmented-value gain

$$
\Delta F_{\mathrm{target}}
=
F(\hat\mu_N^\star)-F(\mu).
$$

Target action에 대해서만 critic evaluation을 한 번 추가한다. 보조적으로 다음도 저장한다.

$$
\Delta Q_{\mathrm{target}}
=
Q(s,\hat\mu_N^\star)-Q(s,\mu).
$$

#### Local target displacement

$$
D_{\mathrm{local}}
=
\frac1{\sqrt d}
\left\|
\frac{\hat\mu_N^\star-\mu}{\sigma}
\right\|_2.
$$

Gaussian proposal에서 $N$이 증가한다고 mean 근처 sample의 비율이 증가하는 것은 아니다. 동일한 local proposal 영역에서 candidate의 절대 개수가 증가하는 것으로 해석한다.

확인할 예상 패턴은 다음과 같다.

- $N=2\rightarrow8$: $P_{\mathrm{hit}}$과 $\Delta F_{\mathrm{target}}$ 증가
- $N=8\rightarrow16,32$: 두 값이 plateau
- $D_{\mathrm{local}}$ 역시 plateau하거나 감소

이 패턴이 관측되면 다음과 같이 설명한다.

> Additional samples provide denser coverage of the same local proposal region, but beyond $N=8$ they rarely produce meaningfully different update directions.

ESS, radial shell, normalized weight entropy와 cross-$N$ $5\times5$ projection matrix는 사용하지 않는다.

### 2.5 KL–CFM theoretical alignment

100k steps마다 seeds $0,1,2$에서만 sparse diagnostic을 수행한다.

동일한 fresh target에 대해 다음 세 방향을 계산한다.

- Solver를 통과한 endpoint KL gradient $g_{\mathrm{KL}}$
- BPTT가 필요 없는 pure CFM gradient $g_{\mathrm{CFM}}$
- 실제 momentum을 포함한 live AdamW update $\Delta\theta_{\mathrm{Adam}}$

다음을 기록한다.

$$
\cos(g_{\mathrm{KL}},g_{\mathrm{CFM}}),
\qquad
\cos(-g_{\mathrm{KL}},\Delta\theta_{\mathrm{Adam}}),
$$

$$
\langle-g_{\mathrm{KL}},\Delta\theta_{\mathrm{Adam}}\rangle,
\qquad
\|g_{\mathrm{KL}}\|,
\quad
\|g_{\mathrm{CFM}}\|,
\quad
\|\Delta\theta_{\mathrm{Adam}}\|.
$$

Norm-matched KL comparator는 다음과 같이 정의한다.

$$
\Delta\theta_{\mathrm{KL,norm}}
=
-\|\Delta\theta_{\mathrm{Adam}}\|
\frac{g_{\mathrm{KL}}}
{\|g_{\mathrm{KL}}\|+\varepsilon}.
$$

Norm-matched KL step과 실제 AdamW step의 fresh-target post-step residual을 비교한다.

별도의 KL Adam optimizer는 만들지 않는다. Diagnostic 계산은 live parameter, optimizer state와 training RNG를 변경하지 않는다. 메모리 부족 시 diagnostic batch 128을 $4\times32$로 나눈다.

## 3. Checkpoint 생성

각 $N$과 seed 조합에 대해 독립적인 checkpoint directory를 사용한다.

```text
checkpoints/dog-run/N2/seed0/
checkpoints/dog-run/N2/seed1/
...
checkpoints/dog-run/N32/seed4/
```

각 run에서 다음 checkpoint를 생성한다.

```text
step_000100000.pkl
step_000200000.pkl
...
step_001000000.pkl
```

즉 run마다 10개, 전체 25개 run에서 총 250개 checkpoint를 생성한다. 각 checkpoint에는 현재 학습 재개에 필요한 actor, critic, temperature, optimizer, replay buffer, guidance buffer와 RNG state를 포함한다.

저장 전에 다음을 검증한다.

- 경로에 현재 $N$과 seed가 정확히 포함됨
- 서로 다른 run이 동일한 checkpoint를 overwrite하지 않음
- 1M-step final checkpoint가 존재함
- checkpoint를 load했을 때 저장 step과 actor configuration의 $N$이 일치함

## 4. 결과 집계와 reviewer rebuttal

### 4.1 성능 통계

- 각 $N$의 5-seed final return을 모두 표시한다.
- 중심 추정치와 95% bootstrap CI를 함께 보고한다.
- 같은 seed끼리 pairing하여 $N=16-8$, $N=32-8$ return difference를 계산한다.
- Paired difference의 개별 seed 값과 95% bootstrap CI를 함께 제시한다.
- Point estimate가 non-monotonic하고 paired CI가 0을 포함하면 “performance plateaus and is non-monotonic beyond $N=8$”로 기술한다.
- 음의 paired difference가 seeds 전반에서 일관되고 CI가 0을 제외할 때만 $N>8$에서 성능이 감소한다고 주장한다.
- Seeds가 5개이므로 CI 폭과 개별 seed 결과를 숨기지 않으며, non-significant difference를 동일 성능의 증명으로 해석하지 않는다.

### 4.2 Figure 1: 이론과 실제 update의 정렬

- Fresh projection before/after 및 realization
- Stored projection before/after
- Raw/scaled/full reward drift
- Supplementary panel: KL–CFM–AdamW gradient alignment

### 4.3 Figure 2: $N$과 performance plateau

세 패널로 제한한다.

1. Final return vs. $N$, 모든 seed point와 95% CI
2. $P_{\mathrm{hit}}$과 better/non-improving categorical fraction
3. $\Delta F_{\mathrm{target}}$과 $D_{\mathrm{local}}$

결과가 예상 패턴을 지지하면 reviewer 답변은 다음 문장을 중심으로 구성한다.

> Increasing $N$ initially improves the probability of finding a better local target. Beyond $N=8$, this probability and the achieved target-value gain saturate, because additional candidates continue to cover the same local proposal region rather than producing meaningfully different update directions.

## 5. 구현 설정과 검증

### 5.1 Diagnostic 설정

- `enabled`
- `light_every: 10`
- `drift_probe_batch: 128`
- `shadow_interval: 100000`
- `shadow_batch: 128`
- `shadow_seeds: [0, 1, 2]`

각 run은 `num_train_action_samples`를 $2,4,8,16,32$ 중 하나로 override한다. Checkpoint 경로는 `env_id/N/seed`를 모두 포함하고 저장 간격은 100k로 고정한다.

### 5.2 필수 테스트

- Fresh residual이 actor loss와 정확히 동일한 $\hat\mu^\star$를 사용함
- Post residual을 다음 optimizer mini-step 이전에 계산함
- Reward drift before/after에서 state, action과 tangent가 동일함
- 코드의 scaled advantage와 raw $A_i$의 수치적 관계가 일치함
- Candidate mechanism metric은 기존 Q evaluation을 재사용함
- Target augmented-value 계산에만 critic evaluation 1회가 추가됨
- Sparse KL diagnostic이 live parameters, optimizer state와 training RNG를 변경하지 않음
- Diagnostics on/off 동일-seed short run의 training trajectory가 일치함
- $N=2$와 $N=32$, seed 0 short run에서 모든 metric이 finite임
- 25개 run의 checkpoint 경로가 서로 충돌하지 않음
- 각 run의 final checkpoint에서 정상적으로 학습을 resume할 수 있음

Phase 1에서는 separate-$N$ training 결과만 비교한다. Diversity 분석, cross-$N$ prefix probe, age-bin 분석과 no-buffer causal comparison은 수행하지 않는다.
