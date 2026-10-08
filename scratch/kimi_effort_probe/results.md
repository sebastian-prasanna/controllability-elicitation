# Kimi K3 reasoning_tokens by OpenRouter reasoning.effort (8 val questions, temp 0, 2026-09-17)

| effort | median | mean | per-question |
|---|---|---|---|
| none | 0 | 0 | [0, 0, 0, 0, 0, 0, 0, 0] |
| low | 215 | 412 | [505, 158, 127, 120, 131, 272, 1542, 442] |
| medium | 349 | 382 | [614, 280, 709, 108, 303, 395, 235, 416] |
| default | 1232 | 2019 | [1030, 136, 1433, 168, 4285, 5435, 2864, 803] |
| high | 445 | 1584 | [6071, 195, 325, 183, 2312, 87, 2931, 565] |
| xhigh | 744 | 2048 | [8232, 0, 344, 165, 1008, 715, 5144, 772] |

Providers were mixed across calls (Moonshot, Fireworks, Chutes, Parasail, Together, ...); n=8 so per-question noise is large.

## Pinned replication (provider=moonshotai only, 40 val questions, subsample_seed=1, temp 0)

| effort | median | mean | median ratio to default | longer than default on |
|---|---|---|---|---|
| default (no field) | 1221 | 6710 | 1.00 | - |
| low | 234 | 375 | 0.16 | 5/40 |
| medium | 238 | 571 | 0.17 | 5/40 |
| high | 734 | 3763 | 0.53 | 6/40 |
| xhigh | 782 | 4141 | 0.61 | 11/40 |

Conclusion: omitting `reasoning.effort` is the *longest* setting on Kimi K3; every explicit level shortens reasoning. Traces at all levels end naturally (not budget-truncated). low ~= medium; high ~= xhigh.
