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

**Источник чисел — код авторов** (github.com/jkopsick/simulate_formation_and_retrieval,
formation/project/ca3_full_net_lognormalI_09_05_23_assem_275: generateCONFIGStateSTP.h —
сеть, src/main_*.cpp — запуск; разбор ниже). Сводка пользователя (2026-11-19):
численности по табл. 1 статьи 2023 — 89 226 клеток, в коде 2024 — 84 053 (как в
тексте статьи 2024); вероятности по табл. 3 статьи 2023 в основном совпадают с
кодом, но **корзинчатые, CCK+ и аксо-аксональные → PC в коде 0.0265, в
сводке 0.150**, QuadD → PC 0.072 против 0.119 — берём код (на нём авторы
получили устойчивый покой).

Запуск у авторов: клетки Izhikevich 2007 (9 параметров), Рунге–Кутта 4, 5
подшагов на 1 мс; синапсы **проводимостные (COBA)** — в ветке CARLsim6
feat/ca3net COBA включён по умолчанию и обязателен, setConductances() не
вызывают (README ветки, https://github.com/UCI-CARL/CARLsim6/tree/feat/ca3net);
у каждой связи своё время спада (STPtdAMPA у PC, STPtdGABAa у тормозных) и
Tsodyks–Markram (setSTP 9-параметровый — без него связь не работает);
задержка 1 мс; фоновый ток каждую 1 мс каждой клетке заново:
lognormal(μ, σ своего типа) × uniform(0, 1), потолок 10 000 пА; μ, σ — табл. 3
ниже (PC 4.0/1.5; AAC 4.0/1.25; BC 5.5/1.0; CCK 4.0/1.0; BiC 4.0/1.0; Ivy 5.5/1.0;
MFA 4.0/1.25; QuadD 4.0/1.25). Единицы синапса CARLsim6 (вес × g × ресурс TM
→ проводимость, нС?) и обращение спада — сверить по исходникам ветки перед
переносом. Контроль авторов (их покой по типам, табл. 3) — запуском их кода
на CARLsim6 feat/ca3net (только GPU, задержка 1 мс, RK4).

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


## Сеть из кода авторов (разбор generateCONFIGStateSTP.h)

```
ПОПУЛЯЦИИ (тип, клеток, знак):
  CA3_QuadD_LM         2609 INHIBITORY_NEURON
  CA3_Axo_Axonic       1482 INHIBITORY_NEURON
  CA3_Basket            115 INHIBITORY_NEURON
  CA3_BC_CCK            168 INHIBITORY_NEURON
  CA3_Bistratified     2281 INHIBITORY_NEURON
  CA3_Ivy              1459 INHIBITORY_NEURON
  CA3_MFA_ORDEN         562 INHIBITORY_NEURON
  CA3_Pyramidal       75377 EXCITATORY_NEURON
  всего 84053

КЛЕТКИ Izhikevich (C, k, Vr, Vt, a, b, Vpeak, Vmin, d):
  CA3_QuadD_LM       186 1.776 -73.48 -54.94 0.005843 -3.449 7.066 -64.4 52
  CA3_Axo_Axonic     165 3.961 -57.1 -51.72 0.004639 8.684 27.8 -73.97 15
  CA3_Basket         45 0.9952 -57.51 -23.38 0.003846 9.264 18.45 -47.56 -6
  CA3_BC_CCK         135 0.583 -59 -39.4 0.005745 -1.245 18.27 -42.77 54
  CA3_Bistratified   107 3.935 -64.67 -58.74 0.001952 16.58 -9.929 -59.7 19
  CA3_Ivy            364 1.916 -70.43 -40.86 0.009151 1.908 -6.92 -53.4 45
  CA3_MFA_ORDEN      209 1.38 -57.08 -39.1 0.007838 12.93 16.31 -40.68 0
  CA3_Pyramidal      102 1.54 -63.12 -23.96 0.008 -35.78 36.4 -38.7 235

СВЯЗИ (пре → пост: вероятность, нач. вес, макс. вес, задержка, g быстр.):
  CA3_QuadD_LM       → CA3_Axo_Axonic     p 0.00529  вес 1 (макс 2)  задержка 1  g 1.4731
  CA3_QuadD_LM       → CA3_Basket         p 0.06672  вес 1 (макс 2)  задержка 1  g 1.8153
  CA3_QuadD_LM       → CA3_BC_CCK         p 0.04957  вес 1 (макс 2)  задержка 1  g 1.3082
  CA3_QuadD_LM       → CA3_Pyramidal      p 0.07234  вес 0.75 (макс 2)  задержка 1  g 1.1832
  CA3_Axo_Axonic     → CA3_Pyramidal      p 0.02653  вес 0.75 (макс 2)  задержка 1  g 1.8696
  CA3_Basket         → CA3_QuadD_LM       p 0.00500  вес 1 (макс 2)  задержка 1  g 1.7508
  CA3_Basket         → CA3_Axo_Axonic     p 0.02500  вес 1 (макс 2)  задержка 1  g 2.0248
  CA3_Basket         → CA3_Basket         p 0.00500  вес 1 (макс 2)  задержка 1  g 3.2816
  CA3_Basket         → CA3_BC_CCK         p 0.00500  вес 1 (макс 2)  задержка 1  g 1.6862
  CA3_Basket         → CA3_Bistratified   p 0.02500  вес 1 (макс 2)  задержка 1  g 1.7704
  CA3_Basket         → CA3_MFA_ORDEN      p 0.00500  вес 1 (макс 2)  задержка 1  g 1.8087
  CA3_Basket         → CA3_Pyramidal      p 0.02653  вес 0.75 (макс 2)  задержка 1  g 1.5724
  CA3_BC_CCK         → CA3_QuadD_LM       p 0.02500  вес 1 (макс 2)  задержка 1  g 1.3350
  CA3_BC_CCK         → CA3_Axo_Axonic     p 0.02500  вес 1 (макс 2)  задержка 1  g 1.4938
  CA3_BC_CCK         → CA3_Basket         p 0.00500  вес 1 (макс 2)  задержка 1  g 1.7452
  CA3_BC_CCK         → CA3_BC_CCK         p 0.00500  вес 1 (макс 2)  задержка 1  g 0.9656
  CA3_BC_CCK         → CA3_Bistratified   p 0.02500  вес 1 (макс 2)  задержка 1  g 1.3715
  CA3_BC_CCK         → CA3_MFA_ORDEN      p 0.00500  вес 1 (макс 2)  задержка 1  g 1.3552
  CA3_BC_CCK         → CA3_Pyramidal      p 0.02653  вес 0.75 (макс 2)  задержка 1  g 1.3063
  CA3_Bistratified   → CA3_QuadD_LM       p 0.00775  вес 1 (макс 2)  задержка 1  g 1.4905
  CA3_Bistratified   → CA3_Axo_Axonic     p 0.00659  вес 1 (макс 2)  задержка 1  g 1.6552
  CA3_Bistratified   → CA3_Basket         p 0.00926  вес 1 (макс 2)  задержка 1  g 1.9943
  CA3_Bistratified   → CA3_BC_CCK         p 0.00432  вес 1 (макс 2)  задержка 1  g 1.4426
  CA3_Bistratified   → CA3_Bistratified   p 0.03258  вес 1 (макс 2)  задержка 1  g 1.5470
  CA3_Bistratified   → CA3_Ivy            p 0.00386  вес 1 (макс 2)  задержка 1  g 2.0612
  CA3_Bistratified   → CA3_MFA_ORDEN      p 0.00862  вес 1 (макс 2)  задержка 1  g 1.5677
  CA3_Bistratified   → CA3_Pyramidal      p 0.01372  вес 0.75 (макс 2)  задержка 1  g 1.4311
  CA3_Ivy            → CA3_QuadD_LM       p 0.00198  вес 1 (макс 2)  задержка 1  g 1.5673
  CA3_Ivy            → CA3_Axo_Axonic     p 0.00425  вес 1 (макс 2)  задержка 1  g 1.7584
  CA3_Ivy            → CA3_Basket         p 0.01630  вес 1 (макс 2)  задержка 1  g 2.1114
  CA3_Ivy            → CA3_BC_CCK         p 0.01074  вес 1 (макс 2)  задержка 1  g 1.5401
  CA3_Ivy            → CA3_Bistratified   p 0.01703  вес 1 (макс 2)  задержка 1  g 1.6601
  CA3_Ivy            → CA3_Ivy            p 0.00375  вес 1 (макс 2)  задержка 1  g 2.1427
  CA3_Ivy            → CA3_MFA_ORDEN      p 0.01693  вес 1 (макс 2)  задержка 1  g 1.6872
  CA3_Ivy            → CA3_Pyramidal      p 0.01360  вес 0.75 (макс 2)  задержка 1  g 1.5407
  CA3_MFA_ORDEN      → CA3_QuadD_LM       p 0.00376  вес 1 (макс 2)  задержка 1  g 1.4720
  CA3_MFA_ORDEN      → CA3_Axo_Axonic     p 0.00438  вес 1 (макс 2)  задержка 1  g 1.6285
  CA3_MFA_ORDEN      → CA3_Basket         p 0.00729  вес 1 (макс 2)  задержка 1  g 1.9723
  CA3_MFA_ORDEN      → CA3_BC_CCK         p 0.00527  вес 1 (макс 2)  задержка 1  g 1.4150
  CA3_MFA_ORDEN      → CA3_Bistratified   p 0.00480  вес 1 (макс 2)  задержка 1  g 1.5365
  CA3_MFA_ORDEN      → CA3_Ivy            p 0.00268  вес 1 (макс 2)  задержка 1  g 2.0820
  CA3_MFA_ORDEN      → CA3_MFA_ORDEN      p 0.00211  вес 1 (макс 2)  задержка 1  g 1.5527
  CA3_MFA_ORDEN      → CA3_Pyramidal      p 0.00634  вес 0.75 (макс 2)  задержка 1  g 1.3603
  CA3_Pyramidal      → CA3_QuadD_LM       p 0.01337  вес 0.75 (макс 2)  задержка 1  g 0.8744
  CA3_Pyramidal      → CA3_Axo_Axonic     p 0.01482  вес 0.75 (макс 2)  задержка 1  g 0.9326
  CA3_Pyramidal      → CA3_Basket         p 0.01974  вес 0.75 (макс 2)  задержка 1  g 1.1725
  CA3_Pyramidal      → CA3_BC_CCK         p 0.01730  вес 0.75 (макс 2)  задержка 1  g 0.8475
  CA3_Pyramidal      → CA3_Bistratified   p 0.01586  вес 0.75 (макс 2)  задержка 1  g 0.8837
  CA3_Pyramidal      → CA3_Ivy            p 0.02516  вес 0.75 (макс 2)  задержка 1  g 1.3143
  CA3_Pyramidal      → CA3_MFA_ORDEN      p 0.02099  вес 0.75 (макс 2)  задержка 1  g 0.8803
  CA3_Pyramidal      → CA3_Pyramidal      p 0.02507  вес 0.625 (макс 21.8182)  задержка 1  g 0.5531
  связей 51

STP (пре → пост: U, τ_f мс, τ_r мс, τ распада AMPA/GABAa мс):
  CA3_QuadD_LM       → CA3_Axo_Axonic     U 0.218  τ_f 22.3  τ_r 635.0  τ_d 5.17
  CA3_QuadD_LM       → CA3_Basket         U 0.232  τ_f 16.4  τ_r 663.2  τ_d 4.29
  CA3_QuadD_LM       → CA3_BC_CCK         U 0.208  τ_f 17.8  τ_r 596.5  τ_d 4.83
  CA3_QuadD_LM       → CA3_Pyramidal      U 0.194  τ_f 24.8  τ_r 382.1  τ_d 9.11
  CA3_Axo_Axonic     → CA3_Pyramidal      U 0.237  τ_f 12.9  τ_r 361.0  τ_d 7.62
  CA3_Basket         → CA3_QuadD_LM       U 0.245  τ_f 19.3  τ_r 589.2  τ_d 5.16
  CA3_Basket         → CA3_Axo_Axonic     U 0.273  τ_f 23.2  τ_r 725.0  τ_d 3.80
  CA3_Basket         → CA3_Basket         U 0.269  τ_f 11.2  τ_r 689.5  τ_d 3.01
  CA3_Basket         → CA3_BC_CCK         U 0.236  τ_f 16.7  τ_r 636.8  τ_d 4.21
  CA3_Basket         → CA3_Bistratified   U 0.251  τ_f 16.7  τ_r 680.3  τ_d 4.72
  CA3_Basket         → CA3_MFA_ORDEN      U 0.241  τ_f 19.6  τ_r 581.9  τ_d 5.23
  CA3_Basket         → CA3_Pyramidal      U 0.228  τ_f 16.7  τ_r 384.3  τ_d 7.64
  CA3_BC_CCK         → CA3_QuadD_LM       U 0.175  τ_f 17.3  τ_r 398.2  τ_d 6.48
  CA3_BC_CCK         → CA3_Axo_Axonic     U 0.176  τ_f 18.5  τ_r 477.4  τ_d 5.44
  CA3_BC_CCK         → CA3_Basket         U 0.196  τ_f 14.9  τ_r 505.1  τ_d 4.69
  CA3_BC_CCK         → CA3_BC_CCK         U 0.119  τ_f 23.4  τ_r 283.3  τ_d 4.89
  CA3_BC_CCK         → CA3_Bistratified   U 0.180  τ_f 15.2  τ_r 478.3  τ_d 5.97
  CA3_BC_CCK         → CA3_MFA_ORDEN      U 0.168  τ_f 17.8  τ_r 421.4  τ_d 6.54
  CA3_BC_CCK         → CA3_Pyramidal      U 0.146  τ_f 13.8  τ_r 376.9  τ_d 9.10
  CA3_Bistratified   → CA3_QuadD_LM       U 0.236  τ_f 17.9  τ_r 594.3  τ_d 5.53
  CA3_Bistratified   → CA3_Axo_Axonic     U 0.239  τ_f 19.2  τ_r 686.3  τ_d 4.57
  CA3_Bistratified   → CA3_Basket         U 0.252  τ_f 14.6  τ_r 695.2  τ_d 3.86
  CA3_Bistratified   → CA3_BC_CCK         U 0.221  τ_f 17.7  τ_r 592.2  τ_d 4.58
  CA3_Bistratified   → CA3_Bistratified   U 0.247  τ_f 13.6  τ_r 775.0  τ_d 4.58
  CA3_Bistratified   → CA3_Ivy            U 0.225  τ_f 13.5  τ_r 706.9  τ_d 4.38
  CA3_Bistratified   → CA3_MFA_ORDEN      U 0.233  τ_f 18.3  τ_r 605.2  τ_d 5.54
  CA3_Bistratified   → CA3_Pyramidal      U 0.213  τ_f 16.6  τ_r 481.9  τ_d 7.49
  CA3_Ivy            → CA3_QuadD_LM       U 0.239  τ_f 26.1  τ_r 563.5  τ_d 6.89
  CA3_Ivy            → CA3_Axo_Axonic     U 0.245  τ_f 25.5  τ_r 651.6  τ_d 5.67
  CA3_Ivy            → CA3_Basket         U 0.255  τ_f 19.1  τ_r 665.2  τ_d 4.75
  CA3_Ivy            → CA3_BC_CCK         U 0.230  τ_f 21.0  τ_r 614.0  τ_d 5.40
  CA3_Ivy            → CA3_Bistratified   U 0.246  τ_f 22.7  τ_r 660.5  τ_d 6.24
  CA3_Ivy            → CA3_Ivy            U 0.232  τ_f 17.7  τ_r 675.5  τ_d 5.51
  CA3_Ivy            → CA3_MFA_ORDEN      U 0.236  τ_f 28.4  τ_r 578.9  τ_d 6.96
  CA3_Ivy            → CA3_Pyramidal      U 0.220  τ_f 23.0  τ_r 439.5  τ_d 9.01
  CA3_MFA_ORDEN      → CA3_QuadD_LM       U 0.232  τ_f 21.0  τ_r 637.9  τ_d 5.52
  CA3_MFA_ORDEN      → CA3_Axo_Axonic     U 0.235  τ_f 21.4  τ_r 762.6  τ_d 4.55
  CA3_MFA_ORDEN      → CA3_Basket         U 0.250  τ_f 15.7  τ_r 759.1  τ_d 3.90
  CA3_MFA_ORDEN      → CA3_BC_CCK         U 0.221  τ_f 17.1  τ_r 693.9  τ_d 4.32
  CA3_MFA_ORDEN      → CA3_Bistratified   U 0.241  τ_f 17.3  τ_r 776.6  τ_d 4.96
  CA3_MFA_ORDEN      → CA3_Ivy            U 0.220  τ_f 13.9  τ_r 806.5  τ_d 4.32
  CA3_MFA_ORDEN      → CA3_MFA_ORDEN      U 0.228  τ_f 22.5  τ_r 642.1  τ_d 5.53
  CA3_MFA_ORDEN      → CA3_Pyramidal      U 0.216  τ_f 20.6  τ_r 496.0  τ_d 7.15
  CA3_Pyramidal      → CA3_QuadD_LM       U 0.200  τ_f 27.2  τ_r 453.3  τ_d 5.82
  CA3_Pyramidal      → CA3_Axo_Axonic     U 0.197  τ_f 26.3  τ_r 630.7  τ_d 4.92
  CA3_Pyramidal      → CA3_Basket         U 0.221  τ_f 21.2  τ_r 691.4  τ_d 3.97
  CA3_Pyramidal      → CA3_BC_CCK         U 0.205  τ_f 22.5  τ_r 530.4  τ_d 4.29
  CA3_Pyramidal      → CA3_Bistratified   U 0.204  τ_f 23.9  τ_r 569.2  τ_d 5.37
  CA3_Pyramidal      → CA3_Ivy            U 0.192  τ_f 18.8  τ_r 623.5  τ_d 4.48
  CA3_Pyramidal      → CA3_MFA_ORDEN      U 0.196  τ_f 29.0  τ_r 445.0  τ_d 5.95
```

## Эталон на torch: сеть авторов в их масштабе (2026-11-19)

Контроль их кодом на CARLsim не запустить (setupNetwork на ~10⁸ синапсов
PC → PC съел всю ОЗУ, WSL упал). Вместо него — их сеть, пересчитанная на
torch по ядру CARLsim6 feat/ca3net (снято с исходников ветки,
https://github.com/UCI-CARL/CARLsim6/tree/feat/ca3net, коммит c495599:
snn_gpu_module.cu — доставка, спад, RK4; snn_manager.cpp — порядок шага,
connectRandom, спад 1 − 1/τ):

- tools/ca3_kopsick_extract.py — разбор файла сети авторов (createGroup,
  setNeuronParameters, connect, setSTP, фон из main) в
  data/hippocampus/ca3_kopsick.json; числа не переписаны руками;
- tools/ca3_kopsick.py — счёт: STP и проводимость своя у каждой связи
  (у CARLsim — на каждом синапсе; одна пара пре–связь ведёт себя одинаково,
  поэтому состояние на (пре, связь) и (получатель, связь) — то же самое),
  Δg = вес × g × u⁺x⁻/U, E_AMPA 0 / E_GABAa −70 мВ, спад ×(1 − 1/τ_d) за
  1 мс, Izhikevich 9 параметров RK4 5 × 0.2 мс, рефрактерность 1 мс (только
  после спайка на последнем подшаге — как в ядре), фон lognormal × U(0, 1)
  каждую 1 мс. Отличия: прирост g на синапсе у CARLsim заменяет остаток, у
  нас складывается (остаток < 0.4% при ISI ≥ 50 мс); STDP PC → PC в покое
  выключен (веса начальные).

Сеть: 84 053 клетки, 175.9 млн синапсов, 1.4 ГБ GPU, 2 с покоя — 12 с.
**Хорошо:** покой устойчив, все 8 типов в пределах s.d. таблицы 3, три
зерна совпадают до 0.1 Гц (100–2000 мс):

| тип | наш (зёрна 1/2/3) | статья, модель |
|---|---|---|
| PC | 0.46 / 0.46 / 0.46 ± 0.47 | 0.42 ± 0.63 |
| AAC | 18.9 / 18.9 / 18.8 | 19.17 ± 1.43 |
| BC | 8.2 / 8.0 / 8.0 | 8.2 ± 3.22 |
| BC CCK+ | 0.99 / 1.04 / 0.96 | 0.49 ± 0.65 |
| BiC | 12.4 / 12.4 / 12.4 | 14.08 ± 1.32 |
| Ivy | 1.97 / 1.93 / 1.95 | 1.40 ± 1.03 |
| MFA-ORDEN | 1.29 / 1.29 / 1.21 | 1.04 ± 1.02 |
| QuadD-LM | 6.6 / 6.55 / 6.55 | 5.70 ± 0.95 |

Значит, семантика их кода понята верно, и этот эталон — образец для
переноса CA3 в мозг. Дальше — встраивание: клетки Izhikevich, 8 типов,
связи по таблице, фон по типу; масштаб CA3 в мозге (у нас 10 000 клеток,
у авторов 84 053) — решить, сохраняя число входов на клетку.

**CA3 размером с наш (10 000 клеток, доли типов авторов), входов на клетку
столько же** (вероятности ÷ 0.119, `degree=1`; 20.9 млн синапсов, 0.25 ГБ):
**хорошо** — те же частоты, три зерна (PC 0.46 ± 0.47 у всех трёх, AAC
18.9–19.2, BC 7.7–8.3, BiC 12.4–12.6, QuadD 6.6–6.7, CCK 0.9–1.4). Покой
держит число входов, а не размер сети — в мозг CA3 идёт с 10 000 клеток,
связи по числу входов авторов (без среднего поля для недостающих).

## Встраивание в мозг (2026-11-20)

brain/circuit.py — TypedCircuit: сеть зоны по файлу авторов на клетках
зоны мозга (AreaSpec.circuit; anatomy: CA3 = PC 10 000 + тормозные по долям
авторов, 1151). Счёт эталона перенесён на шаг мозга 0.5 мс: спад g ×(1 −
dt/τ_d) (тот же заряд, что у CARLsim за 1 мс), STP Эйлером за dt, задержка
1 мс = 2 шага, фон новый каждую 1 мс, RK4 3 × 0.167 мс; шаг без
синхронизаций (EventTable движка, CUDA Graph). Местного контура «E + 10% I»
и пути «ca3: коллатерали» больше нет.

**Отдельно на шаге 0.5 мс — хорошо** (зерно 1, без входа мозга): PC 0.46 ±
0.48, AAC 18.8, BC 7.5, BiC 12.3, QuadD 6.7, CCK 1.1, Ivy 2.0, MFA 1.1 —
как эталон на 1 мс.

**В мозге (словесный, зерно 42) — плохо, но понятно почему:**
1. Сначала сеть горела (все типы 30–390 Гц): вход мозга (EC + DG) шёл поверх
   фона авторов — вход в покое дважды (PC +22 мВ ≈ +365 пА). Исправлено:
   в сеть идёт вход мозга минус его величина при частотах покоя источников.
2. Множитель мВ → пА у 4 типов был 5·10⁸ (малый сигнал терялся в float32) —
   исправлено (float64).
3. Остались вспышки: после develop (10 с + проверка 10 с) PC 4.4 ± 0.28 Гц —
   все клетки одинаково, синхронные эпизоды; 29 из 60 популяций в ±20%
   (было 34). По 5 мс: собственный ритм CA3 ~14 Гц (PC 0.2 → 2.5 Гц →
   залп тормозных 40 Гц → тишина), раз в ~2 с — полный эпизод петли
   (PC 29 Гц → CA1 56 → EC Vb 16). Причина: **зубчатая молчит** (0–0.2 Гц
   при покое 0.3), и тормозные CA3 теряют вход мшистых волокон (DG → ca3.I,
   4100 входов по 0.3 мВ) — ~10 мВ ≈ 150–450 пА тонуса; без него они
   стреляют только залпами. Диагностическая перерезка DG → ca3.I и EC →
   ca3.I (с пересчётом входа покоя): ритма нет, PC 0.2–0.6, тормозные 4–13 Гц.

Следующее — покой зубчатой (CHAT_PLAN шаг 3, строка «DG»; готовые модели —
Santhakumar, Aradi & Soltesz 2005; Dyhrfjeld-Johnsen et al. 2007), не
перерезка путей.
