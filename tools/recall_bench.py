import sys; sys.path.insert(0, ".")
from pathlib import Path
from brain.mind import Mind
from brain.teach import load_facts, same_word
m = Mind.load(sys.argv[1]); L = m.language
facts = [f for f in load_facts(Path("teach-data.txt")) if not f.held_out]
print("эпизодов:", len(L.hippocampus), "пример достраивания:", L.hippocampus.recall(L.tokenize("столица франции?"), key=L.word))
for mv in [float(x) for x in sys.argv[2:]]:
    m.cfg.episode_recall_mv = mv
    hits, ex = 0, []
    for f in facts:
        said = [t for t, _ in L.generate(f.question) if t[0].isalnum()]
        hit = any(same_word(w, f.answer.split()[-1]) for w in said[:3]); hits += hit
        ex.append(f"{f.question}→{' '.join(said[:4])}")
    print(f"подсказка {mv:4.0f} мВ: {hits}/20   ", " | ".join(ex[:4]))
