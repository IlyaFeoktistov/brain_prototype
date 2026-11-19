# CA3 по Kopsick et al.: что переносим (шаг 3 CHAT_PLAN, следующий цикл)

Зачем: CA3 словесного мозга двустабилен — в одном экземпляре проводки молчит,
в другом горит 17 Гц и растёт (CHAT_PLAN шаг 3, циклы 3–4). Готовая модель,
у которой покой CA3 устойчив, — Kopsick et al. 2023 (Cognitive Computation
15:1190, https://doi.org/10.1007/s12559-021-09954-2; PMC10473858): 8 типов
клеток, 51 типоспецифичная связь, ~84–90 тыс. клеток; **сети без
специфичности типов клеток или связей неустойчивы** (там же). Та же сеть с
таблицами — Kopsick et al. 2024 (J Comput Neurosci,
https://doi.org/10.1007/s10827-024-00881-3; PMC11470887), код анализа —
github.com/jkopsick/cell_assembly_formation_retrieval; сеть — CARLsim4-Hippocampome.
Модель — МЫШЬ (данные Hippocampome.org).

**Не хватает (ищет пользователь):** число клеток каждого типа и вероятности
связей между типами — в обеих статьях только на рис. 1a (в статье 2023 —
возможно, таблицей; PMC закрыт проверкой браузера), или в файлах
Hippocampome.org / CARLsim4-Hippocampome.

Ниже — таблицы 1–3 статьи 2024 (как в тексте; единицы — Izhikevich 2007:
k нС/мВ, a 1/мс, b нС, d пА, C пФ, V мВ; g нС, τ мс).

```
===== Izhikevich parameters by neuron type
Neuron Type | k | a | b | d | C | Vr | Vt | Vmin | Vpeak
CA3 Pyramidal | 1.54 | 0.008 | -35.78 | 235 | 102 | -63.12 | -23.96 | -38.70 | 36.40
CA3 Axo-Axonic | 3.961 | 0.005 | 8.684 | 15 | 165 | -57.100 | -51.719 | -73.969 | 27.799
CA3 Basket | 0.995 | 0.004 | 9.264 | -6 | 45 | -57.506 | -23.379 | -47.556 | 18.455
CA3 BC CCK+ | 0.583 | 0.006 | -1.245 | 54 | 135 | -58.997 | -39.398 | -42.771 | 18.275
CA3 Bistratified | 3.935 | 0.002 | 16.580 | 19 | 107 | -64.673 | -58.744 | -59.703 | -9.929
CA3 Ivy | 1.916 | 0.009 | 1.908 | 45 | 364 | -70.435 | -40.859 | -53.400 | -6.920
CA3 MFA ORDEN | 1.380 | 0.008 | 12.933 | 0 | 209 | -57.076 | -39.102 | -40.681 | 16.313
CA3 QuadD-LM | 1.776 | 0.006 | -3.449 | 52 | 186 | -73.482 | -54.937 | -64.404 | 7.066
===== Tsodyks-Markram parameters for each connection type in the model
Presynaptic | Postsynaptic | g | τ d | τ r | τ f | U
CA3 Pyramidal | CA3 Pyramidal | 0.55 | 7.55 | 318.51 | 21.45 | 0.27
CA3 Pyramidal | CA3 Axo-Axonic | 0.70 | 4.92 | 630.73 | 26.26 | 0.20
CA3 Pyramidal | CA3 Basket | 0.88 | 3.97 | 691.42 | 21.16 | 0.22
CA3 Pyramidal | CA3 Basket CCK+ | 0.64 | 4.29 | 530.40 | 22.45 | 0.20
CA3 Pyramidal | CA3 Bistratified | 0.66 | 5.37 | 569.15 | 23.85 | 0.20
CA3 Pyramidal | CA3 Ivy | 0.99 | 5.67 | 552.27 | 26.73 | 0.19
CA3 Pyramidal | CA3 Mossy Fiber-Associated ORDEN | 0.66 | 5.95 | 444.99 | 29.01 | 0.20
CA3 Pyramidal | CA3 QuadD-LM | 0.66 | 5.82 | 453.29 | 27.16 | 0.20
CA3 Axo-Axonic | CA3 Pyramidal | 1.40 | 7.62 | 361.03 | 12.93 | 0.24
CA3 Basket | CA3 Pyramidal | 1.18 | 7.64 | 384.34 | 16.74 | 0.23
CA3 Basket | CA3 Axo-Axonic | 2.02 | 3.80 | 725.03 | 23.21 | 0.27
CA3 Basket | CA3 Basket | 3.28 | 3.01 | 689.51 | 11.19 | 0.27
CA3 Basket | CA3 Basket CCK+ | 1.69 | 4.21 | 636.76 | 16.72 | 0.24
CA3 Basket | CA3 Bistratified | 1.77 | 4.72 | 680.33 | 16.72 | 0.25
CA3 Basket | CA3 Mossy Fiber-Associated ORDEN | 1.81 | 5.23 | 581.94 | 19.60 | 0.24
CA3 Basket | CA3 QuadD-LM | 1.75 | 5.16 | 589.20 | 19.31 | 0.25
CA3 Basket CCK+ | CA3 Pyramidal | 0.98 | 9.10 | 376.87 | 13.76 | 0.15
CA3 Basket CCK+ | CA3 Axo-Axonic | 1.49 | 5.44 | 477.43 | 18.50 | 0.18
CA3 Basket CCK+ | CA3 Basket | 1.75 | 4.69 | 505.12 | 14.86 | 0.20
CA3 Basket CCK+ | CA3 Basket CCK+ | 0.97 | 4.89 | 283.28 | 23.38 | 0.12
CA3 Basket CCK+ | CA3 Bistratified | 1.37 | 5.97 | 478.31 | 15.25 | 0.18
CA3 Basket CCK+ | CA3 Mossy Fiber-Associated ORDEN | 1.36 | 6.54 | 421.42 | 17.84 | 0.17
CA3 Basket CCK+ | CA3 QuadD-LM | 1.33 | 6.48 | 398.15 | 17.34 | 0.17
CA3 Bistratified | CA3 Pyramidal | 1.07 | 7.49 | 481.85 | 16.61 | 0.21
CA3 Bistratified | CA3 Axo-Axonic | 1.66 | 4.57 | 686.28 | 19.16 | 0.24
CA3 Bistratified | CA3 Basket | 1.99 | 3.86 | 695.21 | 14.60 | 0.25
CA3 Bistratified | CA3 Basket CCK+ | 1.44 | 4.58 | 592.19 | 17.69 | 0.22
CA3 Bistratified | CA3 Bistratified | 1.55 | 4.58 | 775.04 | 13.60 | 0.25
CA3 Bistratified | CA3 Ivy | 2.06 | 5.33 | 649.83 | 18.17 | 0.23
CA3 Bistratified | CA3 Mossy Fiber-Associated ORDEN | 1.57 | 5.54 | 605.25 | 18.30 | 0.23
CA3 Bistratified | CA3 QuadD-LM | 1.49 | 5.53 | 594.33 | 17.89 | 0.24
CA3 Ivy | CA3 Pyramidal | 1.16 | 9.01 | 439.50 | 23.01 | 0.22
CA3 Ivy | CA3 Axo-Axonic | 1.76 | 5.67 | 651.64 | 25.51 | 0.24
CA3 Ivy | CA3 Basket | 2.11 | 4.75 | 665.16 | 19.12 | 0.25
CA3 Ivy | CA3 Basket CCK+ | 1.54 | 5.40 | 614.01 | 20.98 | 0.23
CA3 Ivy | CA3 Bistratified | 1.66 | 6.24 | 660.48 | 22.69 | 0.25
CA3 Ivy | CA3 Ivy | 2.14 | 5.51 | 675.54 | 17.72 | 0.23
CA3 Ivy | CA3 Mossy Fiber-Associated ORDEN | 1.69 | 6.96 | 578.90 | 28.45 | 0.24
CA3 Ivy | CA3 QuadD-LM | 1.57 | 6.89 | 563.47 | 26.15 | 0.24
CA3 Mossy Fiber-Associated ORDEN | CA3 Pyramidal | 1.02 | 7.15 | 496.05 | 20.62 | 0.22
CA3 Mossy Fiber-Associated ORDEN | CA3 Axo-Axonic | 1.63 | 4.55 | 762.60 | 21.45 | 0.24
CA3 Mossy Fiber-Associated ORDEN | CA3 Basket | 1.97 | 3.90 | 759.12 | 15.70 | 0.25
CA3 Mossy Fiber-Associated ORDEN | CA3 Basket CCK+ | 1.42 | 4.32 | 693.92 | 17.08 | 0.22
CA3 Mossy Fiber-Associated ORDEN | CA3 Bistratified | 1.54 | 4.96 | 776.57 | 17.27 | 0.24
CA3 Mossy Fiber-Associated ORDEN | CA3 Ivy | 2.08 | 5.39 | 712.27 | 21.22 | 0.22
CA3 Mossy Fiber-Associated ORDEN | CA3 Mossy Fiber-Associated ORDEN | 1.55 | 5.53 | 642.10 | 22.52 | 0.23
CA3 Mossy Fiber-Associated ORDEN | CA3 QuadD-LM | 1.47 | 5.52 | 637.95 | 21.01 | 0.23
CA3 QuadD-LM | CA3 Pyramidal | 0.89 | 9.11 | 382.14 | 24.79 | 0.19
CA3 QuadD-LM | CA3 Axo-Axonic | 1.47 | 5.17 | 635.01 | 22.34 | 0.22
CA3 QuadD-LM | CA3 Basket | 1.82 | 4.29 | 663.25 | 16.42 | 0.23
CA3 QuadD-LM | CA3 Basket CCK+ | 1.31 | 4.83 | 596.50 | 17.78 | 0.21
===== Firing rates (mean ± s.d.) for each neuron type as recorded in our model and in vivo
Neuron Type | Background Mean (µ) Lognormal Input (pA) | Background Variance Lognormal(σ 2 ) Input (pA) | Model Firing Rate (Hz) | Immobility Firing Rate (Hz) | Animal | Animal State | Reference
Pyramidal | 4.0 During pattern presentations: 4.0 | 1.5 During pattern presentations: 1.0 | 0.42 ± 0.63 | 0.2 | Rat | Awake; freely-moving | (Mizuseki et al., 2013 )
0.5 | Rat | Awake; freely-moving | (Kay, K., 2016 ;)
0.72 ± 0.51 | Rat | urethane-anesthetized | (Lasztóczi 2011 )
CA3a: 0.4 CA3b: 0.3 | Rat | Awake; freely-moving | (Oliva, A., 2016 ;)
1.74 ± 1.45 | Mice | Awake; freely-moving | (Ding, L., 2020 )
Axo-Axonic | 4.0 | 1.25 | 19.17 ± 1.43 | 22.8 ± 3.1 | Rat | urethane-anesthetized | (Viney, T. J., 2013 )
Basket | 5.5 | 1.0 | 8.2 ± 3.22 | 20 ± 7 | Rat | urethane-anesthetized | (Tukker, J. J., 2013 )
17 ± 7 | Rat | isoflurane-anesthetized | (Lapray, D., 2012 )
8.2 ± 5.6 | Rat | Awake; head-fixed | (Varga, C., 2012 )
Basket CCK+ | 4.0 | 1.0 | 0.49 ± 0.65 | 0.99 | Rat | urethane-anesthetized | (Lasztóczi 2011 )
Bistratified | 4.0 | 1.0 | 14.08 ± 1.32 | 0.9 ± 0.26* | Rat | urethane-anesthetized | (Klausberger, T., 2004 )
30.4* | Rat | Awake; Freely moving | (Katona, L., 2014 )
Ivy | 5.5 | 1.0 | 1.40 ± 1.03 | 1.7 ± 0.3* | Rat | urethane-anesthetized | (Fuentealba, P., 2008 )
3.0 ± 3.6* | Rat | Awake; freely-moving
MFA-ORDEN | 4.0 | 1.25 | 1.04 ± 1.02 | N/A | --- | --- | ---
QuadD-LM | 4.0 | 1.25 | 5.70 ± 0.95 | 6.14 | Rat | urethane-anesthetized | (Lasztóczi 2011 )
```
