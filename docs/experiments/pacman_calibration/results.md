| Arm | O1 (points), mean (s.d.) | Seeds | Eval episode length | Ghosts eaten / episode | Pellets / episode |
|---|---|---|---|---|---|
| random (ladder) | 56 (6) | 63, 61, 46, 51, 58 | 7 | 0.05 | 4.8 |
| DQN Adam 5e-4 (ladder) | 426 (39) | 425, 485, 442, 367, 409 | 34 | 1.01 | 15.0 |
| A  DQN RMSProp 2.5e-4 | 406 (46) | 389, 445, 373, 351, 472 | 28 | 0.85 | 15.2 |
| B  DQN RMSProp, Atari units, clipped | 478 (34) | 467, 467, 539, 434, 481 | 31 | 0.41 | 34.2 |
| NEC 1e-4, small rewards (ladder) | 357 (49) | 370, 302, 340, 444, 329 | 31 | 0.71 | 17.4 |
| C  NEC 7.92e-6, Atari units | 483 (39) | 504, 536, 433, 500, 441 | 34 | 1.04 | 18.1 |
| D  NEC 1e-4, Atari units | 519 (58) | 562, 559, 493, 417, 565 | 38 | 1.16 | 19.4 |
| ec_frozen, small rewards (ladder) | 389 (20) | 419, 405, 379, 371, 369 | 32 | 0.73 | 16.5 |
| E  ec_frozen, Atari units | 364 (44) | 333, 353, 306, 426, 404 | 31 | 0.79 | 16.3 |
| MFEC, small rewards (ladder) | 390 (33) | 337, 418, 413, 419, 366 | 29 | 1.01 | 17.0 |
| F  MFEC, Atari units | 385 (40) | 315, 400, 369, 415, 426 | 27 | 1.06 | 16.7 |
