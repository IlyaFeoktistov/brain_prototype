# Artificial Brain — NEST SNN prototype

Прототип самоорганизующейся спайковой нервной системы. Симуляция нейронов и синапсов не самописная: используется [NEST Simulator](https://nest-simulator.readthedocs.io/).

## Архитектура

```text
                    BRAIN
                      │
       ┌──────────────┼──────────────┐
       │              │              │
   NEURONS         SYNAPSES       NEUROMODULATION
       │              │              │
   AdEx/HH         STDP/Clopath    dopamine
       │              │              │
       └──────────────┼──────────────┘
                      │
               STRUCTURE
                      │
          ┌───────────┴───────────┐
          │                       │
      neurogenesis             pruning
          │                       │
          └───────────┬───────────┘
                      │
                HOMEOSTASIS
                      │
              self-organization
```

### Что реализовано готовыми моделями NEST

- **AdEx**: `aeif_psc_delta_clopath` для возбуждающих клеток и `aeif_psc_delta` для тормозных. NEST также содержит HH-модели, поэтому HH можно включить следующим этапом.
- **Clopath STDP**: voltage-based STDP из Clopath et al. 2010.
- **Dopamine**: `volume_transmitter` + `stdp_dopamine_synapse`.
- **Structural plasticity**: динамическое создание/удаление синапсов через synaptic elements и homeostatic growth curves.
- **Homeostasis**: calcium-dependent growth curve structural plasticity.

NEST документирует `clopath_synapse` как voltage-based STDP, совместимый с `aeif_psc_delta_clopath` и `hh_psc_alpha_clopath`. Dopamine реализован через `volume_transmitter` и `stdp_dopamine_synapse`.

## Жизненный цикл нейронов

Нейроны создаются через `nest.Create()` при нейрогенезе. Контроллер следит за активностью и возрастом.

```python
neuron_death_enabled = False
```

Если `False`, смерть нейронов запрещена.

Если `True`, хронически неактивные и почти изолированные нейроны отключаются: все входящие/исходящие связи удаляются, затем node переводится в `frozen`. Это ограничение текущего публичного API NEST: обычного физического `Delete(node)` для освобождения уже созданного neuron node нет. Синапсы NEST удаляет штатно.

## Как разговаривать с SNN

```bash
python -m brain.chat
```

```text
                 внешний текст
                       │
                   tokenizer
                       │
          SensoryEncoder (стабильный sensory-код)
                       │
     256 parrot-каналов ──static──► E/I CORTEX (800 E / 200 I)
                                     recurrent E -> E: STDP + dopamine
                                     I -> E: конкуренция
                                     synaptic scaling, structural plasticity
                                          │
                                    current state
                         ┌────────────────┴───────────────┐
                         ▼                                ▼
           ожидание = drive через выученные          Readout (decoder)
           E -> E синапсы в молчащие нейроны         state -> token
                         │
                prediction error ──► dopamine ──► закрепление
```

Где что хранится:

- **Память** — только веса E -> E синапсов (`stdp_dopamine_synapse`). Связь «столица -> франции -> париж» нигде больше не записана.
- **Readout** (`Readout.prototypes`) — усреднённое состояние коры для каждого известного токена. Он отвечает только на вопрос «какому токену соответствует это состояние» и не знает переходов между токенами.
- **Ожидание** — возбуждение, которое текущий ансамбль посылает через синапсы с весом выше врождённого (`ee_weight_max`) в нейроны, которые сейчас молчат.

Команды:

```text
/read путь [проходов]     прочитать файл/папку (.txt/.md/.rst) — основной способ обучения
/read: текст              прочитать текст из строки
/learn a b c [=> d]       учебный опыт: последовательность × N + reward
/reward [сила]            dopamine прямо сейчас
/predict слова            чего мозг ожидает дальше
/debug слово              текущий ансамбль слова и совпадение с прошлым опытом
/debug_association a b    E -> E синапсы a -> b, b -> a и фон всей коры
/stats, /brain, /help, /quit
любой текст               генерация: prompt -> state -> ожидание -> token -> снова в мозг
```

### Как проверять фундамент

```text
Ты: /learn столица франции => париж
Ты: /debug_association столица франции
столица -> франции:   32 синапсов, avg 0.2559 ...
франции -> столица:   35 синапсов, avg 0.1272 ...
фон E -> E:       32089 синапсов, avg 0.1278 ...
Ты: столица
Мозг: франции париж
```

Ассоциация считается настоящей, только если `a -> b` заметно сильнее и обратного направления, и фона.

### Обучение

- `/learn` — та же последовательность `trials` раз, после каждого прохода dopamine-залп. Ответы нигде не записываются.
- `/read` — мозг просто проживает поток. После каждого токена считается prediction error = 1 − cos(ожидание, фактическое состояние), и dopamine пропорционален ошибке: неожиданное закрепляется, ожидаемое почти нет.
- После каждой фразы — пауза и synaptic scaling (входной E -> E вес нейрона не выше `synaptic_scaling_budget` × врождённый).

### Известные ограничения

- Ожидание строится только из текущего токена (цепь первого порядка): после «файл» в «открыть файл для…» и «закрыть файл после…» мозг не различает контекст. Нужна рабочая память / медленно затухающее состояние.
- Скорость: ~3–4× медленнее реального времени на одном потоке. NEST не поддерживает structural plasticity в нескольких потоках; для экспериментов можно поставить `threads=8, structural_plasticity_enabled=False` (~4× быстрее).
- Наблюдение (`/debug`, `/predict`) тоже стимулирует мозг и оставляет eligibility traces.

## Запуск

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m brain.chat
```

Или Docker:

```bash
docker build -t brain-prototype .
docker run --rm -it brain-prototype python3 -m brain.chat
```

## Следующий научный этап

1. HH для выбранных областей вместо одного AdEx-класса.
2. Рекуррентная working memory.
3. Разделение сенсорных, ассоциативных и моторных областей.
4. Dopamine reward prediction и delayed reward.
5. Short-term plasticity.
6. Астроциты (`aeif_cond_alpha_astro` доступен в NEST).
7. Пространственная топология и расстояния.
8. Рабочая память для контекста длиннее одного токена.
