"""
Варианты гиппокампа без правки кода: правит hippocampus_areas/pathways и запускает стенд.

    .venv/bin/python data/tools/variant.py "ca3.inhibition=6,entorhinal-ca1.w=2:2,ca1-association.k=100" data/tools/hippo_lab.py 20000 [key=value ...]

Ключи: <зона>.inhibition / .kind / .background / .k_ee / .feedback / .target / любое поле AreaSpec;
<откуда>-<куда>.w=низ:верх / .k / .iw (вес торможения).
"""
import runpy, sys
import brain.config as C
spec = dict(kv.split("=", 1) for kv in sys.argv[1].split(",") if kv)
orig_a, orig_p = C.hippocampus_areas, C.hippocampus_pathways
def areas():
    out = orig_a()
    for a in out:
        if f"{a.name}.inhibition" in spec: a.inhibition = float(spec[f"{a.name}.inhibition"])
        if f"{a.name}.kind" in spec: a.kind = spec[f"{a.name}.kind"]
        if f"{a.name}.background" in spec: a.background = float(spec[f"{a.name}.background"])
        if f"{a.name}.k_ee" in spec: a.k_ee = int(spec[f"{a.name}.k_ee"])
        if f"{a.name}.feedback" in spec: a.feedback = float(spec[f"{a.name}.feedback"])
        if f"{a.name}.target" in spec: a.target_activity = float(spec[f"{a.name}.target"])
        # Любое другое поле зоны: <зона>.<поле>=значение (тип — по текущему).
        for key, value in spec.items():
            area, _, field = key.partition(".")
            if area == a.name and field not in ("inhibition", "kind", "background", "k_ee", "feedback", "target"):
                old = getattr(a, field)
                setattr(a, field, value == "1" if isinstance(old, bool) or old is None else type(old)(value))
    return out
def paths():
    out = orig_p()
    for p in out:
        key = f"{p.src}-{p.dst}"
        if key + ".w" in spec:
            lo, hi = map(float, spec[key + ".w"].split(":")); p.weight = (lo, hi)
        if key + ".k" in spec: p.k = int(spec[key + ".k"])
        if key + ".iw" in spec: p.inhibitory_weight = float(spec[key + ".iw"])
    return out
C.hippocampus_areas, C.hippocampus_pathways = areas, paths
import brain.mind as M
M.hippocampus_areas, M.hippocampus_pathways = areas, paths
sys.argv = [sys.argv[2]] + sys.argv[3:]
runpy.run_path(sys.argv[0], run_name="__main__")
