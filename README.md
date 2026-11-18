**English** | [Русский](README.ru.md)

# Anatomical brain — a spiking network on the GPU

A brain model built from spiking neurons. Every part is neurons, synapses
and chemistry taken from brain data — no hand-written decision logic in the
code. The simulator runs on PyTorch and CUDA: AdEx/LIF neurons,
conductance synapses, event-driven spike delivery, plasticity
(STDP → eligibility trace → weight, gated by a third factor: dopamine,
acetylcholine, climbing fibre). Knowledge lives only in synaptic weights.

The long-term goal is a brain that learns from the world and talks like a
chat assistant. **It does not talk yet.** Below is an honest list of what
already works, what does not, and what you will see if you run it.

- [ROADMAP.md](ROADMAP.md) — the goal, the links in the chain to it and
  what blocks each one (in Russian).
- [docs/DESIGN.md](docs/DESIGN.md) — decisions and measurements;
  [docs/SPAUN_TRANSFER.md](docs/SPAUN_TRANSFER.md) — the current main line
  of work; [docs/ISSUES.md](docs/ISSUES.md) — open problems and every
  known stub (all in Russian).

## Status (2026-11-18)

### Works, measured

- **The whole brain runs on one 6 GB GPU.** The default brain has ~320 000
  neurons and ~183 million synapses: 11 cortical areas as layered columns,
  thalamus, basal ganglia, cerebellum, hippocampus, amygdala,
  neuromodulator nuclei, sleep and wake. At rest the firing rate of every
  nucleus and layer is fitted to recordings from awake animals and humans.
- **Vision up to letters.** Text on the screen goes through retina → LGN →
  V1 → the letter area (V4/LO). LGN cells under a word fire about 3× more
  than background. After supervised learning, the letter area recognises
  which of 10 letters sits at the fovea correctly 58% of the time (chance
  is 10%).
- **The web UI.** You can create a brain, type into the chat and watch
  every area live in the microscope. There are cell cards, chemistry,
  pathways, and a view of what the eye sees and where the hand presses.
- **Working-memory control blocks (E1), as separate small networks.**
  These follow the control scheme of Spaun (Eliasmith 2012), rebuilt from
  spiking populations. They are not yet inside the brain. Results:
  - a memory cell (Brunel & Wang 2001) holds one word: 8/8 trials;
  - the thalamic gate (MD under SNr) passes or blocks a write as it
    should: 5 conditions, 8/8 trials, 3 seeds;
  - a position counter steps through positions 1 → 5 without error in
    23/24 trials (3 seeds);
  - writing a 3-word phrase into "position × word" memory, with the
    channel chosen by spiking basal ganglia: 23/24.

### Does not work yet

- **There is no conversation.** You can type a phrase and the brain will
  look at it, but the hand types noise. The motor thalamus (VA) is held
  up by a fitted current and drives the hand map in bursts, so random keys
  are pressed even on an empty screen. This is a known stub (ISSUES 23).
  In one check (working preset, seed 1, with the flags "hand E→E by
  distance", "dendrite coincidence detector" and "switch off VA current")
  the reply to "привет" ("hi") was empty: no noise, but also no answer.
- **The eye does not read along the line.** The gaze stays where the text
  starts and sees the first letters. The letter area recognises letters
  only at the fovea, and moving the gaze letter by letter was never
  learned.
- **Recalling a stored phrase in order ("repeat") works in only 3–4 of 8
  trials.** The cause has been found: the counter's reset weakens the
  first position (docs/SPAUN_TRANSFER.md, step 5b).
- **Letters are not position-invariant.** The letter area loses a letter
  that moves ±3 px away from the fovea.
- **Meaning is not learned.** Mapping a word to its meaning is at chance
  (0.12). No plasticity rule for human cortex has been fitted yet.
- **Premotor → hand is not shown to work.** Driving L5 of the hand map
  presses the right key (16/18). A command arriving from L2/3 or from the
  premotor area does not reach the key.
- Many numbers are still marked `[сверить]` ("to be checked against data").
  Every stub is listed in docs/ISSUES.md under "Костыли и заглушки"
  ("Crutches and stubs").

## Running

Requires an NVIDIA GPU (6 GB is enough), Python 3.12 and Node.js (to
build the UI).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
./run-web.sh            # http://localhost:8000
```

In the web UI, create a new brain from the **working** preset
(`brain/presets.py`). It is the same brain used in the experiments: fitted
visual pathway and learned letters. Its files ship with the repository in
`data/presets/`. Brains are saved to `~/.brain/brains/` (`BRAIN_HOME`).

When creating a brain you can also set the seed and experimental flags
(`brain/flags.py`). Each flag shows how well it is debugged. Wiring flags
are fixed at creation; runtime flags can be toggled in the Menu tab of a
running brain. The menu also shows what the current brain was built with
(preset, seed, flags), read from its saved config, so a run can be
repeated.

Why the minimum brain is ~320 000 neurons: every cortical area is a column
of at least 20 000 cells. In a smaller column almost all input is the
average current standing in for missing neighbours, and the network
fires in pathological bursts (`min_column_neurons` in `brain/config.py`).

The E1 blocks run on the CPU:

```bash
cd tools
PYTHONPATH=. ../.venv/bin/python e1_counter.py     # position counter
PYTHONPATH=. ../.venv/bin/python e1_phrase.py      # write a phrase
PYTHONPATH=. ../.venv/bin/python e1_recall.py      # recall it in order (not working yet)
```

## How the brain meets the world

The chat is like a messenger: the world draws text on a screen, the brain
reads it with its eye and types the answer with its hand on a keyboard.
Where to look, what it means and which key to press is decided by the
network. The code is only the body and the world: retina, keyboard,
screen, the teacher's sweet/bitter taste as reward. Voice goes in through
the ear (cochlea).

```
                SCREEN (world)                      VOICE (world)
                    │                                    │
              retina ON/OFF                       cochlea, auditory nerve
                    │                                    │
                   LGN ◄── reticular nucleus          cochlear nucleus (CN)
                    │                                    │      └► VNLL ┐
          V1: simple (L4) → complex (L2/3)        inferior colliculus ◄─┘
                    │                                    │
          letter area (V4 / LO)                     MGN ◄── reticular
                    │                                    │
                    └──────────►  MEANING CORTEX ◄──── auditory cortex (A1)
                                 (association)  ◄────► speech motor (Broca)
                                 ▲    ▲    ▲
          PFC and goal ◄─────────┘    │    └────────► temporal pole (categories)
          (MD thalamus)               │
                                hippocampus: EC → DG → CA3 ↺ → CA1
                    │
    meaning → letters → Exner's area (premotor) → hand motor cortex → KEYS
                         ▲          │                     ▲
                         │          └► basal ganglia ─────┤ (action choice:
                         │             striatum D1/D2, GPe,│  "press / wait")
                         │             STN, GPi ⊣ VA/VL ───┘
                         │
               cerebellum: mossy fibres → granule → Purkinje ⊣ nuclei → VLp
               (learns from error: inferior olive — climbing fibres)

    Gaze: peripheral vision and V1 → superior colliculus (⊣ GPi) → saccade
```

Chemistry and states:

```
  VTA (dopamine) ◄─ "sweet"; lateral habenula ⊣ VTA ◄─ "bitter", GPi
  locus coeruleus (noradrenaline), raphe (serotonin), nucleus basalis
  (acetylcholine), lateral hypothalamus (orexin)
  sleep: VLPO ⊣ wake nuclei; adenosine builds up while awake; SCN clock → DMH
  amygdala: meaning → basolateral → central ⊣ LC, VTA, nucleus basalis
```

Every cortical area is a layered column (L2/3, L4, L5, L6). Input from the
thalamus and lower areas goes to L4, links to neighbours to L2/3, output to
nuclei and muscles from L5, feedback to the thalamus from L6. Substances
act through receptors: release → concentration → receptor occupancy →
current in the cells.

## Files

- `brain/anatomy.py` — areas, pathways, nuclei, receptors and human data,
  with references to the studies.
- `brain/config.py` — cell, column, plasticity, chemistry and body
  parameters (sources next to the numbers).
- `brain/engine.py` — the simulator; `brain/csrc/events.cu`,
  `brain/kernels.py` — CUDA kernels.
- `brain/mind.py` — the whole brain: build, save, load.
- `brain/body.py`, `brain/world.py`, `brain/session.py` — body, world and
  conversation with the world.
- `brain/presets.py`, `data/presets/` — ready-made brains.
- `brain/server.py`, `web/` — the web UI.
- `tools/` — test benches and experiments (`e1_*.py` — working-memory
  control blocks).

Code comments and docs are in Russian; numbers taken from studies carry a
DOI or URL.
