# -*- coding: utf-8 -*-
"""Выгрузка параметров сети Lindahl & Hellgren Kotaleski 2016 их же кодом
(core/network/default_params.py, класс Inhibition), NEST и тяжёлые пакеты — заглушки.
Запуск: docker run --rm -v ~/.brain/ext/bgmodel/python:/bg:ro -v $PWD:/w -w /w lindahl-py2 python dump.py [Класс]"""
import sys, types, json, imp

class Stub(types.ModuleType):
    __path__ = []
    def __getattr__(self, k):
        if k.startswith('__'):
            raise AttributeError(k)
        return Any()
class Any(object):
    def __call__(self, *a, **k): return Any()
    def __getattr__(self, k): return Any()
    def __getitem__(self, k): return Any()
    def __iter__(self): return iter([])

class Finder(object):
    STUB = ('nest', 'matplotlib', 'pylab', 'scipy', 'mpi4py', 'NeuroTools', 'dotenv', 'psycopg2', 'core.my_nest',
            'core.parallelization', 'mpl_toolkits', 'h5py', 'NeuroTools', 'core.my_axes', 'core.my_signals', 'core.my_population', 'core.data_to_disk', 'core.network.structure', 'core.network.engine', 'core.network.manager')
    def find_module(self, name, path=None):
        return self if any(name == s or name.startswith(s + '.') for s in self.STUB) else None
    def load_module(self, name):
        m = sys.modules.setdefault(name, Stub(name))
        m.__loader__ = self
        return m
sys.meta_path.insert(0, Finder())
sys.path.insert(0, '/bgm/python')

REC = {'AMPA_1': 1, 'NMDA_1': 2, 'GABAA_1': 3, 'GABAA_2': 4, 'GABAA_3': 5, 'AMPA_2': 6}
import core.my_nest as nest
nest.GetDefaults = lambda m: {'receptor_types': dict(REC) if m == 'izhik_cond_exp' else
                              {k: v for k, v in REC.items() if k != 'GABAA_3'}}
nest.get_default_module_paths = lambda *a: ('', '')
nest.install_module = lambda *a, **k: None
import core.directories as dr
from core.network import default_params as dp

cls = sys.argv[1] if len(sys.argv) > 1 else 'Inhibition'
par = getattr(dp, cls)()
d = par.dic

def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (int, float, str, bool)) or x is None:
        return x
    try:
        return float(x)
    except Exception:
        return repr(x)
json.dump(clean(d), open('params_%s.json' % cls, 'w'), indent=1, sort_keys=True)
print('ok', sorted(d.keys()))
