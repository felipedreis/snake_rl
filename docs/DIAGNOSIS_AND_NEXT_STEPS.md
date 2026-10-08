# NEC on Snake: why the 25×25 board fails, and what to test next

*Status as of 2026-10-08. A follow-up to [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md), sections 7–9.
**Exploratory and not pre-registered:** the ablations in section 2 use one seed, short runs, and a scratch
script outside the repo (section 7 describes how to rebuild it). Each next step in section 5 is meant to be
run as its own small change or experiment.*

## In short

Two separate problems are stacked on the 25×25 board.

1. **NEC·CNN's embedding divergence has an identified trigger and an identified enabler.**
   - **Trigger:** the constant walls channel. It is identical in every frame, so it carries no information
     and acts only as a large constant input.
   - **Enabler:** Adam's normalised steps, on a loss that does not pin the embedding's scale.
   - **Ruled out:** the SGD into memory keys and values, and stale keys.
   - **What stops it:** zeroing the walls channel, Adam at lr ≤ 1e-4, or RMSProp with ε = 10⁻².
2. **Using the paper's learning rate exactly is not faithful in effect.** The NEC paper names RMSProp but
   reports no learning rate. The value others use for Atari (7.92e-6) leaves our encoder effectively frozen,
   which turns NEC into frozen-embedding EC.
3. **No agent steers on 25×25 because what it learns is tied to board positions, and there are too many
   positions for the data.** The paper's own architecture has a partial answer: strided convolutions leave
   only a 7×7 map for the dense layer. This has not been tested here, because NEC·DQN-shaped-CNN diverged and
   DQN·DQN-shaped-CNN was never probed.
4. **Atari agents see the whole screen, downsampled to 84×84.** They do not get a window around the agent.
   Our full, top-down observation is the faithful analogue. The real differences are resolution and
   absolute actions.

## 1. What the Atari agents see

- **Preprocessing (Mnih et al., 2015):**
  - take the max over 2 consecutive frames, to remove flicker;
  - convert to luminance (grayscale);
  - rescale the 210×160 frame to 84×84;
  - stack the last 4 frames;
  - repeat each action 4 times.
- **What each paper says:**
  - **NEC:** "We apply the same preprocessing steps as (Mnih et al., 2015), including repeating each action
    four times."
  - **MFEC** (`mfec.pdf`, section 4): observations "were rescaled to 84 by 84 pixels and converted to
    gray-scale".
- **Where the "window" idea comes from:** probably the 2013 DQN paper. It downsampled to 110×84, then cut a
  fixed 84×84 square "that roughly captures the playing area" (to drop the score bar and make the image
  square). The cut does not follow the player.
- **Most Atari games have a fixed camera,** so the agent sees the whole game from above while its sprite moves
  around (Pong, Breakout, Ms. Pac-Man, Space Invaders). Our observation is the same kind of view. An
  egocentric view would be the departure from the paper.
- **The real differences:**
  - **Resolution.** Objects span several pixels. DQN's convolutions (32 8×8/4, 64 4×4/2, 64 3×3/1) shrink
    84 → 20 → 9 → 7.
    - Each final unit sees a 36×36-pixel patch, so local relations ("ball next to paddle") are computed
      inside the convolutions.
    - The dense layer only has to tell apart **49 coarse positions**. Our MLP has to tell apart 581 free cells,
      and our small CNN 625.
    - Our DQN-shaped CNN maps 25 → 13 → 7, and each final unit sees 9×9 cells.
  - **Actions.** Atari joysticks are absolute: UP means up.
    - Ours are relative (straight, right, left), so the right move is the food offset rotated by the snake's
      heading.
    - That is an extra 4-way combination Atari agents never have to learn.

## 2. NEC·CNN divergence: one change at a time

**Setup:**
- agent `nec_dqncnn`, seed 601;
- 25×25 `rooms`, bonus 5, food radius held at 2, relocation on;
- ε from 1 to 0.05 over 50k steps;
- one BLAS thread;
- details in section 7.

**Measures:**
- **Key norm:** a snapshot of the mean ‖h‖ over the latest training minibatch. `run.py` logs 1,000-step window
  means instead, so these numbers differ from the overview's table.
- **Dead ReLUs:** last-layer hidden units that are inactive on the whole minibatch, as in the telemetry.

| Change from baseline | Key norm (≈ 1.1 at init) | Dead ReLUs | Result |
|---|---|---|---|
| None (Adam 5e-4) | 1.6 × 10⁶ at 3k | 80% | diverges |
| No SGD into stored keys or values (`mem_lr = 0`) | 1.0 × 10⁴ at 3k | 79% | diverges |
| No SGD into stored keys only | 4.1 × 10⁶ at 3k | 79% | diverges |
| Re-embed all stored keys every 100 steps | 1.7 × 10⁵ at 3k | 77% | diverges |
| L2-normalised keys and queries | keys = 1, but raw output 350 at 3k | 92% | still breaks underneath |
| Adam with ε = 10⁻² | 4.3 × 10³ at 3k | 66% | diverges, more slowly |
| Adam lr 1e-4 (or 5e-5) | 1.2 at 20k | 46–51% | stable |
| RMSProp lr 5e-4 (ρ 0.95, ε 10⁻²) | 1.8 at 20k | 14% | stable, and it learns |
| RMSProp lr 7.92e-6 (the Atari NEC value) | 1.11 at 20k | 29% | stable, but frozen |
| **Adam 5e-4, walls channel zeroed (walls still lethal)** | **0.31 at 20k** | 19% | **stable** |
| Adam 5e-4, 25×25 `open` board, no bonus | 0.38 at 3k | 2% | stable |
| MLP encoder, Adam 5e-4 | 0.9 at 12k | 48% | stable |

**Does the encoder still learn?** For the stable variants, we compared the embedding at 12k steps with the
initial encoder's, on every third probe state (431 states). The column below is the Pearson correlation of
the two sets of pairwise distances; 1 means the geometry is unchanged.

| Variant | Correlation of distances with initialisation |
|---|---|
| RMSProp 7.92e-6 | **0.999** (frozen) |
| MLP, Adam 5e-4 | 0.67 |
| Adam 1e-4 | 0.57 |
| RMSProp 5e-4 | 0.41 |

**Reading:**
- **The memory SGD is not the cause, and neither are stale keys.** Turning either off still diverges.
- **H1 (no restoring force on scale) holds, and there is a specific trigger:**
  - In the baseline, the gradient reaching the keys fell from 3 × 10⁻⁴ (step 50) to 10⁻⁷ (step 1,200), while
    Adam kept taking full-size steps. The largest step into a stored key was never more than 10⁻⁴.
  - By step 1,200 the weight norms had grown only about 2× in total across layers, from
    [7.8, 11.2, 11.3, 32.0, 8.1] to [8.1, 12.9, 14.4, 40.7, 8.4], while the output had grown 10⁵×. The layers
    lined up into a single high-gain direction.
  - **What gets amplified is the walls channel** (`env.py:158`). It is identical in every frame and makes up 44
    of the ~47 non-zero inputs, so it carries no information and only acts as a large constant.
  - **DQN is spared** because its Q-values must match the targets, which fixes the output's scale. NEC's
    kernel weights do not change under a common rescaling or shift of all keys, so nothing holds the
    encoder's output in place.
- **H2 (dead ReLUs) is a symptom.** The ~33% of units dead at initialisation also come from the walls: each
  unit's on/off state is set by the constant walls input, the same for every state. Without the walls channel
  it is 5–7% (rooms) or 1–2% (open board). The later rise to 80% comes with the alignment.
- **Why the MLP is spared:** this was not tested. The MLP has two weight layers to align, against five in the
  CNN.
- **Zeroing the walls channel is a diagnosis, not the recommended fix for CNNs.** A convolution can use walls
  locally ("wall next to the head"), and that knowledge carries across positions. For the MLP and other
  position-bound heads, the channel is pure bias.

**Caveats:** one seed, at most 20k steps, and a scratch subclass that re-implements `NECAgent._train` with the
same arithmetic. Experiment E1 in section 5 repeats it properly.

## 3. Learning rate and reproduction faithfulness

- **What the paper does and does not report:**
  - It names RMSProp.
  - It does **not** report the learning rate. It says it ran "a hyperparameter sweep on six games" over the SGD
    learning rate, α, embedding size, N and ε.
- **The table others use** (Agostinelli et al. 2019, Table 5, "as in the original papers"):
  - RMSProp: lr 7.92e-6, momentum 0.95, ε 10⁻².
  - NEC settings: p = 50, δ = 10⁻³, α = 0.1, N = 100, key size 128, batch 32.
  - Data: replay 10⁵, no reward clipping, training starts at 5 × 10⁴ steps.
  - Exploration: ε from 1 to 0.001, annealed between steps 5k and 25k.
- **At 7.92e-6 our encoder is frozen** (correlation 0.999 in section 2).
  - Our guess at why (inference, not tested): with ±1 rewards, the per-parameter gradients sit below
    RMSProp's ε of 0.01, so the steps are tiny.
  - NEC does not clip rewards, and Atari returns run into the hundreds, so the same learning rate moves the
    encoder much further there.
- **The faithful choice is the paper's optimizer (RMSProp) with a re-tuned learning rate,** the same way the
  paper tuned its own. Nature DQN also uses RMSProp (lr 2.5e-4, gradient momentum 0.95, squared-gradient
  momentum 0.95, min squared gradient 0.01), so the whole ladder can use one optimizer family.
- **The same issue shows on 10×10** ([`figures/table_g10_d0.txt`](../figures/table_g10_d0.txt)).
  - Frozen-embedding EC scores 0.40 fruit per episode over the second half, and NEC 0.23, at Adam 5e-4.
  - So the learned embedding did worse than a random one, the opposite of the paper's NEC > MFEC.
  - The 10×10 board has no walls channel, so this is not the divergence. The learning rate is the first thing
    to check.

## 4. Why no agent steers on 25×25

**A rough count of radius-2 situations** (cells × 4 headings × ~10 egocentric food offsets):
- 7×7: about 2k situations.
- 25×25 `rooms` (581 free cells): about 23k situations.

**Data against situations:**
- The held phase (250k steps) gives about 6× the data of a 40k-step 7×7 run, for 12× the situations.
- That data is bunched near the start cell: 83% of NEC·MLP's states are within 5 cells of it.
- Each situation needs many samples, because the advantage of the right move is small next to the noise in
  50-step returns.

**This fits every result so far:**
- every agent learns on 7×7;
- NEC weakens at 10×10;
- every agent fails at 25×25;
- NEC·MLP steers at 0.91 near the start and at chance 10 cells away.

**Encoders:**
- **The stride-1 CNN did not help:** its dense layer reads 625 position-specific feature maps, which restores
  the position count.
- **The DQN-shaped CNN cuts the dense layer down to 49 positions,** and each final unit's 9×9-cell view covers
  radius-2 food. This is the paper's own design and the obvious next test.

**Secondary factors:**
- **Every episode starts at the centre, facing right,** so experience piles up there.
- **NEC is sensitive to exploration:**
  - on 10×10, NEC goes from 0.23 to 0.46 when the ε floor rises from 0.02 to 0.1;
  - on 25×25, matching ε raised NEC·MLP's held-phase fruit rate from 1.4 to 25.6 per 100 steps.

## 5. Next steps, as small separate experiments

Ordered so that each one informs the next. Code changes should stay RNG-neutral, with defaults unchanged, as
`CLAUDE.md` requires.

| # | Change or experiment | Cost | Read-out | Prediction / decision |
|---|---|---|---|---|
| E0 | **Probe split by distance from the start cell**, plus greedy steering on the agent's own states, in `probe.py`. This promotes the post-hoc split in the encoder-probe report. | small code change | steering per distance band | tooling for E5 |
| E1 | **Confirm section 2 with 3 seeds:** baseline, walls zeroed, Adam 1e-4, RMSProp 5e-4, RMSProp 7.92e-6. 20k steps each; a post-hoc script under the encoder-probe experiment. | ~5 min per run, in parallel | key norm, dead ReLUs, distance correlation | the same pattern on every seed |
| E2 | **Add RMSProp to `nn.py`** (ρ, ε = 10⁻², global-norm clip as in Adam), selectable per agent in the registry; Adam stays the default. Unit test, and check RNG-neutrality against committed runs. | small code change | — | enables E3–E5 |
| E3 | **NEC learning-rate sweep on 10×10 `open`** (no walls channel, so the divergence does not confound it): `ec_frozen` against NEC at RMSProp lr {7.92e-6, 3e-5, 1e-4, 5e-4}, plus Adam 5e-4 as a reference. 60k steps, 5 seeds. | minutes per run | LATE fruit per episode, distance correlation | Is there a learning rate at which NEC (rung 3) beats frozen EC (rung 2)? If none, the learned embedding adds nothing at this budget, and that is itself a finding. |
| E4 | **Health check for `nec_dqncnn` on 25×25 (held phase)** with E3's best optimizer. 100k steps, 1–2 seeds. | ~20 min per run | pre-registered health: key norm < 10× its initial value; distance correlation < 0.9 (it learns) | a stable NEC·CNN |
| E5 | **The architecture test:** DQN·DQN-shaped CNN and NEC·DQN-shaped CNN (healthy, from E4) against their MLP arms, with E0's distance-split probe. Encoder-probe setting, 250k–400k steps. | hours | steering at each distance from the start | If positions are the bottleneck, the DQN-shaped CNN arms steer well beyond the start cell. If not, the dense layer is not the whole story. |
| E6 | **Absolute actions** (`--actions absolute`: 4 directions, reversing continues straight), closer to Atari's joystick. DQN only, 2×2: {relative, absolute} × {MLP, DQN-shaped CNN}. | env flag + short runs | held-phase fruit per 100 steps, steering | It should help the MLP most, since the heading rotation disappears. |
| E7 | **Egocentric observation, as a control arm and not the main condition:** head-centred, rotated to the heading, padded to (2n−1)² so the whole board stays visible. A local window would hide the food and break the Markov property. | env flag | steering at every distance | DQN steers ≥ 0.8 everywhere, and the episodic agents gain the most (states repeat). If the ladder works here but not top-down, the remaining gap is the representation problem NEC's learned embedding claims to solve. |
| E8 | **Change one factor at a time on the way to 25×25.** v1 jumped from 10×10 `open` to 25×25 + `rooms` + bonus at once: try 13×13 `open` → 25×25 `open` → + `rooms` → + bonus. | short DQN and NEC runs | where learning breaks | We already know the walls channel alone decides NEC·CNN's stability. |
| E9 | *(Optional, maximum fidelity.)* **Atari-like rendering:** the board as an 84×84 grayscale image (3 px per cell), a 4-frame stack, and the exact Nature CNN (84 → 20 → 9 → 7). | about 9.3M multiply-adds per forward pass, against about 7.4M for the current DQN-shaped CNN | as E5 | Only worth it if E5 shows that architecture fidelity matters. |

**Also worth considering:** random start cell and heading as an env option.
- **Use it as a diagnostic, not a fix:** it makes the probe's states match training, but it spreads data
  thinner for position-bound learners.
- **Expectation:** it should *lower* MLP scores if section 4 is right.

## 6. Where this leaves the overview's section 9 questions

1. **Is the divergence a known failure mode, and does H1 make sense?** H1 is now supported, with the walls
   channel as its trigger (section 2). We have not found it reported for NEC. The paper's tiny learning rate
   may be why it was never seen, but that is speculation.
2. **Which fix keeps the comparison fair?**
   - **Faithful:** RMSProp with a re-tuned learning rate (section 3).
   - **L2-normalised keys:** not in the paper, and they did not work here.
   - **Clipping memory-key updates:** irrelevant, since the memory SGD is not the cause.
   - **Weight decay:** untested, and not in the paper.
3. **Dead-ReLU spike: fix it directly, or a symptom?** A symptom: of the constant walls input at
   initialisation, and of the alignment afterwards. Do not fix it directly.
4. **Egocentric observation, or architecture (pooling)?**
   - Atari is top-down and full-screen, so an egocentric view belongs in a control arm (E7).
   - The paper's own answer is the strided 7×7 map (E5). Pooling would be a departure.
5. **Is the probe sound?** Yes, but it samples states the agents never visit. Add the distance split and
   on-policy steering (E0).
6. **Is 25×25 `rooms` + bonus a reasonable testbed?** As a target, yes. But v1 changed three factors at once,
   so step through them (E8).

## 7. Rebuilding the scratch ablations

**Script:** the script lived in a temporary session folder and is not in the repo. To rebuild it:

- **Environment:**
  `Snake(size=25, seed=1601, map="rooms", bonus=5, food_radius=2, relocate_food=True)`.
- **Agent:** `NECAgent(env.obs_dim, 3, np.random.default_rng(601), encoder="dqncnn",
  obs_shape=env.obs_shape, N=50)` with defaults otherwise.
- **Exploration:** `run.epsilon(t, 0.05, 50000)`. One BLAS thread.
- **Loop:** the plain act/observe loop of `run.main`, without probes or evaluation.
- **Variants:** a subclass of `NECAgent` overriding `_train` with the same arithmetic, plus:
  - separate key and value rates (`mem_lr = 0`; key rate 0);
  - `refresh_every = 100`, with DNDs that store observations;
  - L2 normalisation of `_embed`'s output and of the touched stored keys, with the gradient projected through
    h = z / ‖z‖;
  - a swapped optimizer: `Adam(lr, eps)`, or an RMSProp with v ← ρv + (1 − ρ)g², p ← p − lr·g/(√v + ε), ρ = 0.95,
    ε = 10⁻², and the same global-norm clip of 10;
  - a wrapped `env._obs` that zeroes channel 3 (the walls), or `map="open", bonus=0`.
- **Drift:** compute embeddings on every third state of `make_probe_set(25, "rooms", 5)` (431 states), with a
  deep copy of the encoder taken at step 0 for reference.

## Sources

- Pritzel et al. 2017, *Neural Episodic Control*, [arXiv:1703.01988](https://arxiv.org/abs/1703.01988)
- Blundell et al. 2016, *Model-Free Episodic Control*, [arXiv:1606.04460](https://arxiv.org/abs/1606.04460)
  (local copy: `mfec.pdf`)
- Agostinelli et al. 2019, *Memory-Efficient Episodic Control Reinforcement Learning with Dynamic Online
  k-means*, [arXiv:1911.09560](https://arxiv.org/abs/1911.09560), Table 5
- Mnih et al. 2015, *Human-level control through deep reinforcement learning*, Nature 518; Mnih et al. 2013,
  *Playing Atari with Deep Reinforcement Learning*, [arXiv:1312.5602](https://arxiv.org/abs/1312.5602)
