import sys; sys.path.insert(0, ".")
from brain.mind import Mind
m = Mind.load(sys.argv[1]); L = m.language
eps = [" ".join(e.tokens) for e in L.hippocampus.episodes if "?" in e.tokens][:6]
print("эпизоды с ?:", *eps, sep="\n  ")
print("? знаком:", L.familiar("?"), "реплики-эпизоды всего", len(L.hippocampus))
print(L.trace("столица франции?", top=5)[-1800:])
