# Confidence-Weighted NSR

Confidence-Weighted Hard Negative Reinforcement (CW-HNSR) extends Negative Sample Reinforcement (NSR) by penalizing high-confidence wrong answers more strongly than low-confidence wrong answers.

This repository is based on RLVR-Decomposed and adapts its RLVR training pipeline for a targeted negative-reinforcement variant focused on reasoning errors that the model strongly commits to.

## One-Line Research Claim

Confidence-weighted NSR extends binary negative reinforcement by using the model's own belief in wrong generations to control penalty strength, improving correction of harmful errors while preserving NSR's diversity benefits.

## Motivation

The source paper, *The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning*, shows that NSR can preserve diversity and improve Pass@k. However, standard NSR treats all wrong answers similarly.

This project targets that gap:

- High-confidence wrong answers are often more harmful and indicate stronger failure modes.
- Low-confidence wrong answers can represent exploration and should be penalized less aggressively.

## Proposed Method: CW-HNSR

For incorrect samples only, compute a confidence-derived hardness weight and scale the negative update by this weight.

- Wrong + high confidence -> stronger penalty
- Wrong + low confidence -> weaker penalty

This keeps the overall RLVR framework unchanged and only modifies how negative reinforcement is weighted.

## Confidence Definition

Let a generated sequence be $y = (y_1, \dots, y_T)$. We define confidence as the geometric mean token probability:

$$
\mathrm{conf}(y) = \exp\left(\frac{1}{T}\sum_{t=1}^{T}\log p_\theta\left(y_t \mid x, y_{1:t-1}\right)\right)
$$

Where:

- $T$ is sequence length
- $p_\theta(y_t \mid x, y_{1:t-1})$ is model token probability at step $t$

Interpretation:

- Measures internal model belief, not correctness
- Length-normalized and numerically stable
- Useful as a relative error-severity signal

## Practical Training Idea

Use confidence only when reward indicates an incorrect answer:

$$
w_{\text{neg}} = g\left(\text{conf}(y)\right), \quad g(\cdot) \in [w_{\min}, w_{\max}]
$$

Then apply weighted negative reinforcement:

$$
\mathcal{L}_{\text{CW-HNSR}} \propto -\, w_{\text{neg}}\,\log \pi_\theta(y\mid x) \quad \text{for } r(y)=0
$$

Recommended safeguards:

- Clip confidence weights to avoid unstable gradients
- Add a minimum penalty floor so wrong answers are never ignored
- Normalize confidence by sequence length (already built into the geometric mean)
- Run ablations against plain NSR and weighted REINFORCE

## Expected Benefits

- More targeted correction of harmful mistakes
- Better sample efficiency by focusing updates where overconfidence is highest
- Preserves NSR's exploration/diversity behavior better than uniformly aggressive penalties

## Limitations

- Confidence is not perfectly calibrated
- Very low-confidence errors may be under-penalized without a floor
- Confidence can vary with decoding settings and prompt format

These are handled through clipping, normalization, minimum penalty, and controlled ablations.

## Repository Setup

This codebase follows RLVR-Decomposed conventions.

```bash
conda create -y -n verl python=3.10.14
conda activate verl
pip install -e .
pip install vllm==0.8.2
pip install latex2sympy2 fire tensordict==0.7.2
python -m pip install flash-attn --no-build-isolation
```

For Qwen3 runs, upgrade:

```bash
pip install "vllm==0.8.5" "transformers==4.52.2"
```

## Training Entry Points

- PSR / NSR / Weighted-REINFORCE: `run_qwen2.5-math-7b_psr_nsr.sh`
- PPO: `run_qwen2.5-math-7b_ppo.sh`
- GRPO: `run_qwen2.5-math-7b_grpo.sh`

## Evaluation

```bash
bash eval.sh
python calculate_metrics.py --file_path <file_to_evaluate>
```

## Proposal Document

The original concept note is included at:

- `Confidence_Weighted_NSR_Proposal.docx`

## Acknowledgment

This repository builds on the RLVR-Decomposed codebase and the paper *The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning*.
