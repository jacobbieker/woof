## Throughput

| run | size | s/step | Mpts*steps/s | sim s / wall s | vs woof km3 | acoustic steps/dt | timed reps | finite |
|---|---|---|---|---|---|---|---|---|
| woof_km2 | 128x128x64 | 0.0581 | 18.0 | 8.61 | 0.93 | 19 | 2 | True |
| woof_km3 | 128x128x64 | 0.0540 | 19.4 | 9.26 | 1.00 | 19 | 2 | True |
| breeze_anelastic_weno5_ci | 128x128x64 | 0.0119 | 87.9 | 41.89 | 4.52 | - | 2 | True |
| breeze_se_matched | 128x128x64 | 0.0561 | 18.7 | 8.91 | 0.96 | 19 | 2 | True |

## Boundary-layer statistics (trailing-window means)

| run | size | t (h) | z_i grad (m) | z_i flux (m) | theta_ml (K) | max <w'w'> (m2/s2) | at z (m) | min <w'th'>/Q | U_ml (m/s) | bulk z_i A=0 / 0.2 (m) |
|---|---|---|---|---|---|---|---|---|---|---|
| woof_km3 | 128x128x64 | 2.00 | 1604 | 1441 | 311.53 | 2.63 | 593 | -0.118 | 8.21 | 1273 / 1457 |
| woof_km2 | 128x128x64 | 2.00 | 1584 | 1447 | 311.52 | 2.66 | 580 | -0.098 | 8.18 | 1273 / 1457 |
| breeze_anelastic_weno5_ci | 128x128x64 | 2.00 | 1614 | 1436 | 311.51 | 2.30 | 569 | -0.088 | 8.46 | 1273 / 1457 |
| breeze_se_matched | 128x128x64 | 2.00 | 1621 | 1490 | 311.56 | 2.69 | 583 | -0.122 | 8.20 | 1273 / 1457 |

FastEddy tutorial figure at 4 h (eyeball reading, 20 m grid): z_i(flux) ~ 1800 m, theta_ml ~ 312.3 K, max <w'w'> ~ 2.85 m2/s2 at ~800 m.

