# data/

- `learning/` — то, на чём мозг учится: `course/` (курс по ступеням: буквы →
  слоги → слова → фразы → факты → вопрос-ответ, проверочные наборы),
  `corpus/` (тексты), `quick/` (справки для быстрых опытов).
- `anatomy/` — устройство мозга из исследований, не обучение: сети зон с
  типами клеток (`hippocampus/ca3_kopsick.json` — CA3 Kopsick et al. 2024,
  `hippocampus/dg_dyhrfjeld.json` — зубчатая Dyhrfjeld-Johnsen et al. 2007 +
  Hippocampome). Получены разбором кода авторов (tools/ca3_kopsick_extract.py,
  tools/dg_extract.py), читает brain/circuit.py.
- `presets/` — обученные веса и сдвиги гомеостаза (копии для веба из свежего
  клона; рабочие — в ~/.brain).
