# Agentic Plan: Reframing the Langevin MCMC Report Without Rewriting Everything

## 1. Core aim

The goal is **not** to replace the work already written by the group. The goal is to make the existing material read as one coherent story:

> Langevin MCMC begins with an ideal continuous-time diffusion whose invariant distribution is the target. In practice we must discretise this diffusion, and discretisation creates a trade-off between statistical accuracy, burn-in, mixing efficiency, and computational cost. ULA, MALA, and KLMC can then be understood as different responses to this trade-off.

This framing preserves the current theory, implementation, and experiment sections, but changes how they are introduced and connected.

---

## 2. What the current report already covers

Based on the current LaTeX file, the report already contains substantial material on:

- motivation for Langevin MCMC;
- why sampling matters in Bayesian inference;
- Langevin diffusion and its invariant distribution;
- discretisation and ULA;
- MALA as a Metropolis-adjusted version of the Langevin proposal;
- convergence intuition and assumptions;
- Bayesian linear Gaussian regression;
- implementation of ULA and MALA;
- ESS and autocorrelation diagnostics;
- a strongly correlated Gaussian benchmark;
- KLMC / kinetic Langevin as an extension;
- numerical experiments, discussion, and conclusion placeholders.

The current notebook already contains experiments involving:

- ULA and MALA;
- step-size behaviour;
- ESS and autocorrelation;
- burn-in handling;
- covariance/mean error diagnostics;
- correlated Gaussian examples;
- Bayesian linear regression;
- kinetic Langevin comparisons.

So the new roadmap should **reuse nearly all of this**. The main work is to add short linking paragraphs and reorganise the interpretation.

---

## 3. What is currently missing or only lightly present

### 3.1 Geometry and preconditioning

The current report has a subsection called **"Why Langevin in High Dimensions?"** and a correlated Gaussian benchmark. This already gestures toward geometry, because the correlated Gaussian is a non-spherical target where different directions behave differently.

However, the current report does **not** appear to contain a real treatment of:

- condition number;
- ill-conditioning;
- preconditioning;
- affine transformations;
- preconditioned ULA or preconditioned MALA.

Therefore, do **not** force a full preconditioning section into the report unless there is enough space. Instead, leave an optional short section or paragraph open for it.

Recommended placeholder:

> **Optional geometry/preconditioning paragraph:** The strongly correlated Gaussian benchmark also illustrates the role of target geometry. When the target has elongated contours or a large condition number, a single scalar step size may be too large in steep directions and too small in flat directions. Preconditioning addresses this by rescaling directions, but we leave a full empirical study of preconditioned Langevin algorithms to future work.

This integrates geometry without undermining the existing report or requiring new implementation.

### 3.2 Burn-in versus mixing

The report already discusses post-burn-in samples and ESS, but the distinction should be made more explicit:

- **burn-in**: time taken to reach the typical set / approximate stationarity;
- **mixing**: efficiency of exploration after burn-in;
- **ESS**: a diagnostic for mixing through autocorrelation.

This is a low-cost, high-value clarification.

### 3.3 Hyperparameter tuning

The current results use step sizes and sweeps, but the report should explicitly state the comparison protocol:

- ULA and MALA should be compared across a common grid of step sizes \(h\);
- MALA acceptance rates should be reported alongside ESS and accuracy;
- KLMC should state how \(h\) and friction \(\gamma\) are chosen;
- comparisons should be made using both raw ESS and cost-aware ESS, e.g. ESS per gradient evaluation.

This does not require rewriting the theory. It mostly belongs in the implementation/results section.

---

## 4. Recommended report structure

The following structure is close to what already exists. The main edits are transition paragraphs and clearer section purposes.

### Section 1: Introduction and motivation

Current material to preserve:

- motivation of Langevin MCMC;
- why sampling is needed;
- why gradients are useful;
- why Gaussian/logistic/KLMC examples are studied.

Suggested reframing:

End the introduction with a research question:

> This project investigates how discretised Langevin methods trade off statistical accuracy, burn-in, mixing efficiency, and computational cost when approximating a target distribution.

Avoid framing the project as simply "comparing algorithms". The algorithms are examples of different ways to approximate or modify Langevin dynamics.

---

### Section 2: Ideal Langevin diffusion

Current material to preserve:

- target distribution \(\pi\);
- Langevin SDE;
- invariant distribution;
- convergence intuition.

Purpose of this section:

Show the ideal object:

\[
dX_t = \nabla \log \pi(X_t)\,dt + \sqrt{2}\,dW_t.
\]

The key message should be:

> In continuous time, Langevin dynamics provide a principled route to sampling from \(\pi\). The practical difficulty is that this SDE cannot generally be simulated exactly.

This sets up the rest of the report.

---

### Section 3: ULA as direct discretisation

Current material to preserve:

- ULA theory;
- ULA implementation;
- Gaussian benchmark;
- mean and covariance error;
- ESS/autocorrelation diagnostics.

Suggested framing:

ULA should be introduced as the simplest practical algorithm arising from Euler-Maruyama discretisation:

\[
X_{n+1} = X_n + h\nabla \log \pi(X_n) + \sqrt{2h}Z_n.
\]

The key question:

> What is lost when the ideal diffusion is replaced by finite steps?

Results to emphasise:

- larger \(h\) can improve mixing / ESS;
- larger \(h\) can also increase discretisation bias;
- ULA is cheap but samples from an \(h\)-dependent approximation to the target.

This should be the first major demonstration of the trade-off.

---

### Section 4: MALA as bias correction

Current material to preserve:

- MALA theory;
- Metropolis-Hastings correction;
- MALA implementation;
- acceptance rates;
- step-size sweep results.

Suggested framing:

Do not present MALA as an unrelated second algorithm. Present it as a response to the ULA problem:

> MALA keeps the Langevin proposal used by ULA, but adds a Metropolis-Hastings accept/reject step to correct the discretisation bias.

The key question:

> Does correcting the bias introduce a cost in acceptance rate, mixing, or computation?

Results to emphasise:

- MALA targets the correct distribution asymptotically;
- the acceptance probability decreases when \(h\) becomes too large;
- MALA may have lower raw ESS than ULA at some step sizes, but better accuracy;
- therefore ESS alone is not enough.

---

### Section 5: Geometry and scaling — optional / minimal

Current material to preserve:

- "Why Langevin in High Dimensions?";
- strongly correlated Gaussian benchmark.

Current status:

The report does **not** currently include a full treatment of preconditioning. Because space is limited, keep this section short unless your supervisor specifically wants more.

Suggested integration:

Add a short paragraph near the correlated Gaussian benchmark or high-dimensional motivation:

> The correlated Gaussian also highlights the role of geometry. In targets with elongated contours, different directions have different natural scales. A single scalar step size may be too small for flat directions and too large for steep directions. This issue is often quantified by the condition number. Preconditioning modifies the Langevin proposal by replacing the scalar step-size geometry with a matrix scaling, but a full implementation of preconditioned samplers is beyond the present scope.

This paragraph connects the benchmark to geometry without creating a new unfinished project.

---

### Section 6: KLMC as alternative dynamics

Current material to preserve:

- overdamped versus underdamped motivation;
- kinetic Langevin diffusion;
- KLMC discretisation;
- convergence theory;
- Gaussian case;
- numerical experiments.

Suggested framing:

KLMC should not be presented as simply a third sampler. It should be introduced as a different response to the same discretisation/efficiency problem:

> Instead of correcting the overdamped Euler proposal with accept/reject, kinetic Langevin changes the continuous-time dynamics by augmenting position with momentum.

Key distinction:

- ULA and MALA are based on the overdamped diffusion;
- KLMC is based on an underdamped diffusion in \((X,V)\);
- momentum may improve exploration, especially along long valleys.

Make sure the numerical integrator used for KLMC is explicitly stated. If the current implementation uses a particular splitting or Euler-style update, name it and do not overclaim theoretical guarantees beyond that implementation.

---

### Section 7: Overall comparison and discussion

This section is essential. Without it, the report may still feel like a list of algorithms.

Add a summary table like:

| Method | Interpretation | Accuracy | Burn-in | Mixing / ESS | Cost | Main weakness |
|---|---|---|---|---|---|---|
| ULA | direct discretisation | biased for finite \(h\) | often fast | can be high | cheap | discretisation bias |
| MALA | ULA proposal + correction | asymptotically exact | acceptance-dependent | can fall with low acceptance | extra accept/reject cost | rejection/tuning |
| KLMC | alternative underdamped dynamics | integrator-dependent | may improve | momentum can help | more complex | tuning \(h,\gamma\) |

Then state the main conclusion:

> The most efficient sampler is not necessarily the one with the highest ESS. A fair comparison must consider whether the sampler is accurate, how long it takes to reach stationarity, how well it mixes once there, and how much computation each effective sample costs.

---

## 5. What not to change too much

To avoid undermining your project mates' work:

- Do not delete the existing theory sections.
- Do not rewrite the KLMC theory from scratch.
- Do not replace their implementation narrative with a completely new structure.
- Do not insert a large preconditioning chapter unless there is time and space.
- Do not turn the project into a condition-number/preconditioning project.

Instead, add short bridges:

1. diffusion \(\rightarrow\) discretisation;
2. ULA \(\rightarrow\) bias-efficiency trade-off;
3. ULA \(\rightarrow\) MALA as correction;
4. MALA \(\rightarrow\) KLMC as alternative dynamics;
5. individual results \(\rightarrow\) final comparison.

---

## 6. Action checklist

### High priority edits

- [ ] Add a clear research question at the end of the introduction.
- [ ] Add a transition from Langevin diffusion to discretisation.
- [ ] In the ULA section, explicitly state that finite step size creates an \(h\)-dependent bias.
- [ ] In the MALA section, explicitly state that MALA corrects ULA's discretisation bias using Metropolis-Hastings.
- [ ] Separate burn-in from mixing/ESS in the diagnostics section.
- [ ] Add or complete the results table/figures requested by the existing `INSERT` comments.
- [ ] Add one final comparison table across ULA, MALA, and KLMC.

### Medium priority edits

- [ ] State the hyperparameter tuning protocol for \(h\), acceptance rates, and \(\gamma\).
- [ ] Report ESS per gradient evaluation or another cost-aware efficiency measure.
- [ ] Clarify that long-run MALA/reference chains are numerical references, not exact truth, except in Gaussian cases.
- [ ] Add a short geometry paragraph near the correlated Gaussian benchmark.

### Optional edits if space allows

- [ ] Add a small condition-number discussion.
- [ ] Add a preconditioning paragraph as future work.
- [ ] Add a preconditioned Gaussian experiment only if implementation time and report space allow.

---

## 7. Coherence warnings

The roadmap is coherent with the current report, but there are a few risks.

### Risk 1: KLMC may feel detached

If KLMC is introduced only after ULA/MALA results, it may feel like an extra topic. Fix this by explicitly saying that KLMC is an alternative way to improve exploration: it changes the underlying dynamics rather than simply correcting the overdamped discretisation.

### Risk 2: Geometry may become too large

Geometry and preconditioning are relevant, but they are not currently developed in the report. Treat them as a supporting explanation for the correlated Gaussian benchmark, not as a new main objective.

### Risk 3: ESS may dominate the narrative

Avoid implying that high ESS is the end goal. The end goal is accurate estimation at reasonable computational cost. ESS is only one diagnostic for post-burn-in mixing.

### Risk 4: Results may not support every claim yet

Only claim what the notebook actually shows. In particular:

- use exact Gaussian errors only where the target is analytically known;
- call regression references "numerical references" rather than ground truth;
- avoid saying KLMC is universally better unless the experiments clearly show it;
- avoid saying preconditioning improves results unless you actually implement it.

---

## 8. Minimal thesis statement to guide editing

Use this as the guiding sentence while editing:

> This project studies how Langevin-based MCMC methods approximate an ideal continuous-time diffusion in practice. The central issue is that discretisation makes simulation possible but creates trade-offs between accuracy, burn-in, mixing efficiency, and computational cost. ULA, MALA, and KLMC illustrate different ways of navigating these trade-offs.

