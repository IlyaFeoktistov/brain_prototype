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

Получишь REPL:

```text
Ты: /learn столица России => Москва
Мозг: ассоциация записана через spike-pattern + dopamine/STDP

Ты: столица России
Мозг: Москва

Ты: /brain
neurons=1000, connections=..., time=...
```

Команды:

```text
/learn вопрос => ответ   обучить ассоциации
/stats                   показать изученные ассоциации
/brain                   состояние мозга
/help                    помощь
/quit                    выход
```

### Что реально происходит при `/learn`

```text
текст вопроса
      │
      ▼
 sparse temporal spike pattern
      │
      ▼
 sensory SNN neurons
      │
      ▼
 dopamine-modulated STDP
      │
      ▼
 output ensemble
      │
      ▼
 learned answer
```

Это **не LLM**. Интерфейс специально сделан честным: SNN не генерирует произвольный русский текст. Он пока является ассоциативной памятью. Для настоящего диалога следующий этап — научить несколько выходных ансамблей формировать последовательности символов/токенов и добавить рекуррентную рабочую память.

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
8. Реальный neural-language readout вместо словаря ответов.
