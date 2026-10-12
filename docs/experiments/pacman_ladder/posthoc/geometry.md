Mean (s.d. over 5 seeds) [chance]. Purity = share of the 10 (action purity: 50) nearest states from other episodes that share the label.

| Agent | Space | pac cell | nearest pellet | nearest ghost | power pellet on | action purity | greedy steering |
|---|---|---|---|---|---|---|---|
| dqn_naturecnn | agent embedding | 0.58 (0.04) [0.08] | 0.51 (0.03) [0.07] | 0.24 (0.03) [0.03] | 0.75 (0.01) [0.51] | 0.81 (0.02) [0.63] | 0.38 (0.03) [0.36] |
| ec_frozen_naturecnn | agent embedding | 0.36 (0.02) [0.09] | 0.31 (0.02) [0.07] | 0.19 (0.01) [0.03] | 0.63 (0.00) [0.50] | 0.75 (0.01) [0.62] | 0.40 (0.03) [0.33] |
| ec_frozen_naturecnn | raw pixels | 0.78 (0.02) [0.09] | 0.68 (0.02) [0.07] | 0.27 (0.01) [0.03] | 0.73 (0.01) [0.50] | 0.87 (0.01) [0.62] |  |
| nec_naturecnn | agent embedding | 0.43 (0.05) [0.10] | 0.39 (0.04) [0.08] | 0.21 (0.02) [0.04] | 0.65 (0.01) [0.50] | 0.78 (0.03) [0.62] | 0.37 (0.03) [0.33] |

| Agent | memory coherence | probe vs memory distance |
|---|---|---|
| ec_frozen_naturecnn | 0.88 (0.03) | 0.98 (0.03) |
| nec_naturecnn | 0.70 (0.05) | 1.26 (0.09) |
