# E6: absolute actions on the pixel render

**Status: planned, not run.** Branch `explore/pixels-rmsprop`. Follows E6 in `docs/DIAGNOSIS_AND_NEXT_STEPS.md`, adapted to pixels.

## Question

Does DQN learn faster or better when the action is a board direction (up, right, down, left, as an Atari joystick) than
when it is relative to the heading (straight, right, left)? With relative actions the network has to combine the food
offset with a heading that it must itself infer from the frames; with absolute actions "food on the left" maps to "go left".

## Design

- Agent `dqn_naturecnn` (Adam, 5e-4, the agent's defaults), `--render pixels` (default wall gray 60, 4-frame stack).
- Boards 7x7 and 10x10, `open`, 60k steps, default epsilon schedule, seeds 1-3. Same settings as the earlier DQN-pixels
  control, so those numbers are a reference, not a substitute: all 12 runs are rerun here with the same code.
- Factor: `--actions relative` against `--actions absolute`. Reversal under absolute actions keeps going straight.
- `--probe-every 10000`: greedy steering accuracy on the fixed probe set. The chance rate differs between the two
  action spaces, so compare steering minus its chance rate, or the raw score only.

## Measures

1. Fruit per episode, second half of training (the plot's usual number).
2. Probe steering and its chance rate, at each probe.

## Predictions (written before the runs)

1. Absolute is ahead of relative on both boards, with the larger gap at 10x10, where relative DQN on pixels is weak
   (0.64 fruit per episode in the earlier control).
2. The gap is smaller at 7x7 (relative: 2.08 earlier), where the board is small enough to learn by brute force.
3. Not predicted: whether absolute actions make the frame stack unnecessary. That is a separate test.

## What would count against it

Absolute not ahead on either board in at least 2 of 3 seeds. Then the rotation is not what limits the CNN here, and
the next suspect is the representation itself (the probe reads chance for NEC everywhere).

## Out of scope

NEC and `ec_frozen` under absolute actions, the RMSProp learning-rate sweep, and the map curriculum. They follow once
the action space is chosen, because the best learning rate may depend on it.
