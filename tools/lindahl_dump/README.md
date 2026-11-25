Выгрузка параметров BG Lindahl & Hellgren Kotaleski 2016 кодом авторов (для tools/bg_lindahl.py).

    git -C ~/.brain/ext/bgmodel fetch --depth 1 origin tag 1.0.0
    mkdir -p ~/.brain/lindahl/bgm10 && git -C ~/.brain/ext/bgmodel archive 1.0.0 python | tar -x -C ~/.brain/lindahl/bgm10
    echo BGMODEL_HOME_DATA=/tmp/bgdata > ~/.brain/lindahl/bgm10/.env
    docker build -q -t lindahl-py2 tools/lindahl_dump
    cd tools/lindahl_dump && for a in "Beta 0" "Slow_wave2 1"; do
      docker run --rm -e BGMODEL_HOME_DATA=/tmp/bgdata -e BGMODEL_HOME_MODULE=/tmp \
        -v ~/.brain/lindahl/bgm10:/bgm -v $PWD:/w -w /w lindahl-py2 python dump_fig1.py $a 10000; done

dump.py — класс параметров как есть (по умолчанию Inhibition); dump_fig1.py — сеть рисунка 1 статьи:
Beta|Slow_wave2(other=Inhibition, общее + M*_low + tata_dop 0.8 + возмущение eNeuro_fig_01_and_02_pert[i]),
i = 0 — активация (кора 20 Гц ±10%), 1 — медленные волны (1 Гц ±10%). NEST, matplotlib, scipy, mpi4py,
NeuroTools — заглушки (для словаря параметров не нужны); receptor_types — из enum SpikeSynapseTypes
module-130701 (AMPA_1=1 … AMPA_2=6).
