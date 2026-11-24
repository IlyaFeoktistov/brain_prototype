"""
Шаг 4а (docs/CHAT_PLAN.md): блок E1 целиком (tools/e1_recall.py, без изменений
внутри) в словесном мозге — слова E1 берёт из смысловой коры, ответ отдаёт в
неё же; экзамен «повтори» видит только популяции слов коры.

Связка кусками по 5 мс (мозг — GPU, E1 — шаги 0.1 мс):
  - вход E1 (счёт спайков слов wc, у E1 NWORD = 20 источников на слово) —
    спайки 20 клеток популяции слова L2/3 смысловой коры (скрининг
    tools/word_port.py) за прошлый кусок, разложенные по шагам E1 случайно;
  - выход E1 (слой ответа, 40 клеток на слово) → кора: рецепторы порта слов
    слова-ответа стреляют words_hz × (частота ответа / 40 Гц) в следующий
    кусок [заглушка пути E1 → кора, ISSUES];
  - учитель — как в E1: расписание (окна слов, сдвиги счётчика, «повтори»)
    и состояние задачи «повторяю» (ISSUES 28–29); слова фразы и «повтори»
    звучат через порт слов — содержимое E1 видит только через кору.

Экзамен — расписание E1 (запись: 3 слота по 1200 мс, слово 200 мс; два
вывода по 4 окна 300 мс); «сказано» — популяции коры по порогам скрининга
(chat_exam.WordBody.said). Меры: фраза целиком (оба вывода), позиции, стоп;
утечка — ответ E1 и чужие слова коры во время записи; повторная запись —
содержимое ячеек слов E1 до первого и после второго вывода.

Мозг перед экзаменом — как в опыте B: разгон 2.5 с, адаптация с гомеостазом
окнами по 1 с (adapt), затем скрининг слов; пластичность — как у chat_exam.

    PYTHONPATH=tools:. .venv/bin/python tools/e1_in_brain.py [seed=1] [trials=4] [adapt=30] [screen=4]
"""

from __future__ import annotations

import sys
import time

import torch

import chat_exam
import word_port
from brain.body import Words
from e1_bg import BG, LIF, NSRC, NWORD
from e1_counter import relay_input, sources
from e1_gate import NR, Relay
from e1_slot import C_EXT, C_SEL, DT, P, R_RESET, R_SEL, Slot

T = 1200.0
NANS = 40
LMAX = 3
READS = 4
REPEATS = 2
CHUNK_MS = 5.0
S_WIN = 300.0        # окно S в периоде записи, мс: вспышка коры ~100 мс от начала слова + запас
SEL_SD = 1.0         # клетки слова для реле MD — избирательность > 1 разброса (tools/word_drive.py: 162 клетки, 909/364 в с)


class E1Net:
    """Сеть e1_recall.main для одного испытания (B = 1), шаг — как в её цикле."""

    def __init__(self, length: int, g_ans: float = 8.0, s_mode: str = "task", msn: str = "lif", B: int = 1, dev: str = "cpu"):
        self.dev = dev
        # Пачка B проб: у каждой своё состояние (все части E1 — по строке пачки), расписание
        # общее; условия по строкам — pos_to_bg [B, 1], s_enable [B], r_enable [B] (замер ворот).
        # Вход стриатума записи (docs/PBWM_TRANSFER.md): "sum" — позиция + сумма всех
        # слов (E1 как был); "task" — позиция + состояние задачи S «запоминаю» от
        # учителя (как CtrlInput S в sir2), окно ворот — фаза 0–300 мс каждого периода
        # записи (такт); "off" — контроль: S не включается никогда.
        self.s_mode = s_mode
        # Замер ворот (tools/e1_gate_check.py): pos_to_bg 0 — позиция не идёт в стриатум;
        # r_mode "off" — сигнал задачи чтения R не включается.
        self.pos_to_bg, self.r_mode = 1.0, "task"
        self.s_enable = torch.ones(B, dtype=torch.bool, device=dev)
        self.r_enable = torch.ones(B, dtype=torch.bool, device=dev)
        self.s_on = torch.zeros(B, dtype=torch.bool, device=dev)
        self.B, self.length, self.g_ans = B, length, g_ans
        self.cur, self.nxt = Slot(B, dev), Slot(B, dev)
        self.g1, self.g2 = Relay(B, dev, 16.0, 1.0, 4.0), Relay(B, dev, 16.0, 1.0, 4.0)
        self.src_cur, self.src_nxt = sources(self.cur, NSRC), sources(self.nxt, NSRC)
        self.bg_w, self.bg_r = BG(B, dev, msn=msn), BG(B, dev, msn=msn)
        self.md_w, self.md_r = Relay(B * P, dev, 16.0, 1.0, 4.0), Relay(B * P, dev, 16.0, 1.0, 4.0)
        self.ws = Slot(B * P, dev)
        self.src_ws = sources(self.ws, NSRC)
        self.ans = LIF(B, NANS, dev, -70.0)
        cur = self.cur
        # По умолчанию e1_recall (шаг 5в): «повтори» — тормозной сброс, сдвиги — залп.
        self.reset_e = None
        self.reset_e_adv = torch.full((B, cur.ne), C_EXT * R_RESET * 1e-3, device=dev)
        self.reset_i = torch.full((B, cur.ni), C_EXT * R_RESET * 1e-3, device=dev)
        self.start = (cur.lab[None, :] == 0).float().expand(B, cur.ne) * (C_SEL * R_SEL * 1e-3)
        self.closed, self.opened = torch.ones(B, dtype=torch.bool, device=dev), torch.zeros(B, dtype=torch.bool, device=dev)
        self.periods = [(400.0 + i * T, "w", i) for i in range(LMAX)]
        self.rec_starts = []
        t_end = 400.0 + LMAX * T
        for r in range(REPEATS):
            rs = t_end + 300.0
            self.rec_starts.append(rs)
            self.periods += [(rs + 400.0 + k * T, "r", (r, k)) for k in range(READS)]
            t_end = rs + 400.0 + READS * T
        self.total = t_end + 200.0
        self.pi = 0

    def held(self) -> torch.Tensor:
        """Ячейки слов: какая группа слов сильнее в каждой позиции (спайки шага, [позиция, слово])."""

        return (self.ws.spk_e @ self.ws.onehot)[:, :P].view(self.B, P, P)

    def step(self, t: float, wc: torch.Tensor) -> torch.Tensor:
        """Шаг DT: wc [B, P] — спайки источников слов. Возвращает спайки ответа по словам [B, P]."""

        B = self.B
        while self.pi + 1 < len(self.periods) and t >= self.periods[self.pi + 1][0]:
            self.pi += 1
        p0, kind, idx = self.periods[self.pi]
        ph = t - p0 if t >= p0 else -1.0
        in_period = 0 <= ph < T
        win_r = any(rs + 150.0 <= t < rs + 400.0 + READS * T for rs in self.rec_starts)
        task_on = self.r_enable & (self.r_mode == "task") & win_r
        self.task_on = task_on
        tc = torch.poisson(torch.where(task_on, 40.0, 2.0) * NWORD * 1e-3 * DT)
        de_c = di_c = de_n = di_n = None
        if 100 <= t < 150 or any(rs + 100 <= t < rs + 150 for rs in self.rec_starts):
            de_c = self.start
        if any(rs <= t < rs + 50 for rs in self.rec_starts):
            de_c, di_c, de_n, di_n = self.reset_e, self.reset_i, self.reset_e, self.reset_i
        if in_period and 300 <= ph < 350:
            de_n, di_n = self.reset_e_adv, self.reset_i
        if in_period and 700 <= ph < 750:
            de_c, di_c = self.reset_e_adv, self.reset_i
        cur, nxt = self.cur, self.nxt
        self.g1.step_counts(relay_input(cur.spk_e, self.src_cur, 1), self.opened if in_period and 400 <= ph < 600 else self.closed)
        self.g2.step_counts(relay_input(nxt.spk_e, self.src_nxt, 0), self.opened if in_period and 800 <= ph < 1000 else self.closed)
        pos = cur.spk_e[:, self.src_cur].sum(2) * self.pos_to_bg
        if self.s_mode == "sum":
            self.bg_w.step(pos, wc.sum(1))
        else:
            s_on = self.s_enable & (self.s_mode == "task") & (kind == "w" and 0 <= ph < S_WIN)
            self.s_on = s_on
            self.bg_w.step(pos, torch.poisson(torch.where(s_on, 40.0, 2.0) * NWORD * 1e-3 * DT))
        self.bg_r.step(pos, tc)
        ctx_w = wc.repeat_interleave(NR, 1)[:, None, :].expand(B, P, P * NR).reshape(B * P, P * NR)
        self.md_w.step_counts(ctx_w, None, self.bg_w.snr.per_channel().reshape(B * P))
        ctx_r = self.ws.spk_e[:, self.src_ws].sum(2).repeat_interleave(NR, 1)
        self.md_r.step_counts(ctx_r, None, self.bg_r.snr.per_channel().reshape(B * P))
        to_ans = self.md_r.spk.view(B, P, P, NR).sum((1, 3))
        self.ans.step(self.g_ans * to_ans.repeat_interleave(NANS, 1), torch.zeros(B, P * NANS, device=self.dev))
        nxt.step(de_n, di_n, add_e=self.g1.to_slot(nxt.lab))
        cur.step(de_c, di_c, add_e=self.g2.to_slot(cur.lab))
        self.ws.step(add_e=self.md_w.to_slot(self.ws.lab))
        return self.ans.spk.view(B, P, NANS).sum(2)


def build(seed: int, adapt: int, screen: int):
    torch.set_num_threads(2)
    mind, b = word_port.build({"seed": seed, "size": "small"})
    b.cfg.event_overflow = "warn"
    port = Words.of(b)
    vocab = list(chat_exam.WORDS) + ["повтори", "?", chat_exam.UNKNOWN] + [w for p in chat_exam.PAIRS for w in p]
    if len(vocab) != b.cfg.words_count:
        raise ValueError(f"словарь {len(vocab)} слов, а порт на {b.cfg.words_count}")
    b.cfg.homeostasis = True
    b.run(2500.0, port.hear(None, 2500.0))
    for _ in range(adapt):
        b.homeostasis_update(b.observe(1000.0, port.hear(None, 1000.0)))
    b.cfg.homeostasis = False
    plastic = b.plasticity
    b.plasticity = False
    pops, thr = word_port.screen(b, port, screen, seed)
    b.plasticity = plastic
    L23 = b.areas["association"].layers["L23"]["E"]
    conv = [torch.nonzero(word_port.LAST_SEL[i] > SEL_SD).squeeze(1) + L23.start for i in range(len(vocab))]
    return b, port, vocab, pops, thr, conv


def main(opts: dict) -> None:
    seed = int(opts.get("seed", 1))
    torch.manual_seed(seed)
    trials = int(opts.get("trials", 4))
    trace = opts.get("trace")
    t0 = time.time()
    b, port, vocab, pops, thr, conv = build(seed, int(opts.get("adapt", 30)), int(opts.get("screen", 4)))
    index = {w: i for i, w in enumerate(vocab)}
    cells = {w: pops[i] for i, w in enumerate(vocab)}
    thr_w = {w: float(thr[i]) for i, w in enumerate(vocab)}
    print(f"зерно {seed}: мозг готов за {time.time() - t0:.0f} с; пороги «сказано» слов фразы: "
          + " ".join(f"{w} {thr_w[w]:.1f}" for w in chat_exam.WORDS), flush=True)

    def said(r) -> str:
        over = [w for w in vocab if float(r[cells[w]].mean()) > thr_w[w] and w not in ("повтори", "?")]
        return over[0] if len(over) == 1 else ("" if not over else "?")

    texts = chat_exam.phrases(seed, trials)
    for s_mode in opts.get("s", "task").split(","):
        # режим входа стриатума записи: s_mode
        # Вход E1: "task"/"off" — схождение (все клетки слова > SEL_SD), "sum" — 20 клеток, как было.
        src_sets = ([conv[index[w]].to(b.device) for w in chat_exam.WORDS] if s_mode != "sum"
                    else [pops[index[w]][:NWORD].to(b.device) for w in chat_exam.WORDS])
        print(f"режим входа стриатума записи: {s_mode}; источники E1 по словам: " + " ".join(f"{w} {len(x)}" for w, x in zip(chat_exam.WORDS, src_sets)), flush=True)
        results, leaks_e1, leaks_ctx, rewrites = [], 0, 0, 0
        steps_chunk = int(round(CHUNK_MS / DT))
        for text in texts:
            net = E1Net(len(text), s_mode=s_mode)
            # Что звучит через порт: (начало, конец, слово).
            heard = [(400.0 + i * T, 400.0 + i * T + 200.0, w) for i, w in enumerate(text)]
            heard += [(rs, rs + 200.0, "повтори") for rs in net.rec_starts]
            ans_rate = torch.zeros(P)
            read = {}
            write_e1 = torch.zeros(LMAX, P)
            diag_src = torch.zeros(LMAX, P)          # Гц источников E1 по словам, 0–200 мс слова
            diag_msn = torch.zeros(LMAX, P)          # стриатум записи E1 по каналам, 0–300 мс слова, Гц
            diag_read = torch.zeros(REPEATS, READS, 3, P)   # стриатум чтения, реле чтения (по словам), ответ E1
            write_ctx = [torch.zeros(b.n, device=b.device) for _ in range(LMAX)]
            held_before = torch.zeros(P, P)
            msn_w_chunks = []                         # стриатум записи по каналам за кусок, Гц (попытки записи)
            held_after = torch.zeros(P, P)
            n_chunks = int(net.total // CHUNK_MS)
            # Трассы по кускам 5 мс (trace=файл.pt) — для картинок tools/e1_in_brain_plot.py.
            tr = {k: [] for k in ("src", "ans", "pops", "held", "msn", "rel", "areas")}
            area_names = [a for a in b.areas if b.areas[a].E is not None]
            held_acc = torch.zeros(P, P)
            for c in range(n_chunks):
                t = c * CHUNK_MS
                kind_now = next((kd for p0, kd, _ in reversed(net.periods) if t >= p0), "w")
                w_now = next((w for a, z, w in heard if a <= t < z), None)
                raster = port.hear(index[w_now] if w_now else None, CHUNK_MS)
                # Ответ E1 прошлого куска → рецепторы слова-ответа.
                p = (b.cfg.words_hz * (ans_rate / 40.0).clamp(max=1.0)) * b.cfg.dt_ms / 1000.0
                for j in range(P):
                    if float(p[j]) > 0:
                        a = port.offset + index[chat_exam.WORDS[j]] * port.per
                        raster[:, a: a + port.per] = torch.maximum(raster[:, a: a + port.per],
                                                                   (torch.rand(raster.shape[0], port.per, device=raster.device) < p[j]).float())
                r = b.observe(CHUNK_MS, raster)
                # Вход E1: спайки 20 клеток популяции слова за кусок, по шагам E1 случайно.
                counts = torch.stack([(r[x] * CHUNK_MS / 1000.0).round().sum() for x in src_sets]).long().cpu()   # [P]
                wc_seq = torch.zeros(steps_chunk, P)
                for j in range(P):
                    if int(counts[j]):
                        wc_seq[torch.randint(0, steps_chunk, (int(counts[j]),)), j] += 1
                src_hz = torch.stack([r[x].mean() for x in src_sets]).cpu()               # [P] Гц на клетку
                msn_acc = torch.zeros(P)
                rel_acc = torch.zeros(P)
                spikes = torch.zeros(P)
                before_win = net.rec_starts[0] - 300.0 <= t < net.rec_starts[0]
                after_win = net.total - 500.0 <= t < net.total - 200.0
                for k in range(steps_chunk):
                    spikes += net.step(t + k * DT, wc_seq[k][None, :])[0]
                    msn_acc += net.bg_w.msn.per_channel()[0] if kind_now == "w" else net.bg_r.msn.per_channel()[0]
                    rel_acc += net.md_r.spk.view(P, P, NR).sum((0, 2))
                    if before_win:
                        held_before += net.held()[0]
                    if after_win:
                        held_after += net.held()[0]
                    if trace:
                        held_acc += net.held()[0]
                ans_rate = spikes / NANS / (CHUNK_MS / 1000.0)
                msn_hz = msn_acc / 40 / steps_chunk * (1000.0 / DT)
                msn_w_chunks.append((t, kind_now, msn_hz.clone()))
                rel_hz = rel_acc / (P * NR) / steps_chunk * (1000.0 / DT)
                if trace:
                    tr["src"].append(src_hz.clone())
                    tr["ans"].append(ans_rate.clone())
                    tr["pops"].append(torch.stack([r[cells[w]].mean() for w in vocab]).cpu())
                    tr["held"].append(held_acc.clone())
                    held_acc.zero_()
                    tr["msn"].append(msn_hz.clone())
                    tr["rel"].append(rel_hz.clone())
                    tr["areas"].append(torch.stack([r[b.areas[a].E.slice].mean() for a in area_names]).cpu())
                # Меры по окнам.
                for (p0, kind, idx) in net.periods:
                    ph = t - p0
                    if kind == "w" and 0 <= ph < 200 and idx < len(text):
                        diag_src[idx] += src_hz * CHUNK_MS / 200.0
                    if kind == "w" and 0 <= ph < 300 and idx < len(text):
                        diag_msn[idx] += msn_hz * CHUNK_MS / 300.0
                    if kind == "r" and 0 <= ph < 300:
                        diag_read[idx[0], idx[1], 0] += msn_hz * CHUNK_MS / 300.0
                        diag_read[idx[0], idx[1], 1] += rel_hz * CHUNK_MS / 300.0
                        diag_read[idx[0], idx[1], 2] += ans_rate * CHUNK_MS / 300.0
                    if kind == "w" and 0 <= ph < 600 and idx < len(text):
                        write_e1[idx] += ans_rate * CHUNK_MS / 600.0
                        write_ctx[idx] += r[: b.n] * CHUNK_MS / 600.0
                    if kind == "r" and 0 <= ph < 300:
                        read.setdefault(idx, torch.zeros(b.n, device=r.device))
                        read[idx] += r[: b.n] * CHUNK_MS / 300.0
            if trace:
                torch.save({"text": text, "vocab": vocab, "words": list(chat_exam.WORDS), "area_names": area_names,
                            "thr": [thr_w[w] for w in vocab], "periods": net.periods, "rec_starts": net.rec_starts,
                            "heard": heard, "chunk_ms": CHUNK_MS,
                            **{k: torch.stack(v) for k, v in tr.items()}}, f"{trace}.{s_mode}.{text}.pt")
            runs = []
            for rr in range(REPEATS):
                line = ""
                for k in range(READS):
                    s = said(read[(rr, k)])
                    line += s if s and s in chat_exam.WORDS else ("·" if s == "" else "?")
                runs.append(line)
            results.append(runs)
            leak_e1 = int((write_e1.max(1).values[: len(text)] > 10.0).sum())
            # Кора при записи: выше порога слово, которого на этой позиции не было.
            leak_ctx = sum(1 for i in range(len(text)) if any(
                float(write_ctx[i][cells[w]].mean()) > thr_w[w] for w in chat_exam.WORDS if w != text[i]))
            before = held_before[: len(text)].argmax(1)
            after = held_after[: len(text)].argmax(1)
            rewrite = int((before != after).sum())
            leaks_e1 += leak_e1
            leaks_ctx += leak_ctx
            rewrites += rewrite
            stored = "".join(chat_exam.WORDS[int(x)] for x in before)
            # Попытки записи: эпизоды стриатума записи канала i (> 20 Гц подряд) в периоде записи i;
            # ячейка i до вывода: доля сильнейшего слова (0.2 — шум, 1 — одно слово).
            for i in range(LMAX):
                p0 = net.periods[i][0]
                on = [float(m[i]) > 20.0 for (tt, kd, m) in msn_w_chunks if p0 <= tt < p0 + T and kd == "w"]
                episodes = sum(1 for j in range(len(on)) if on[j] and (j == 0 or not on[j - 1]))
                h = held_before[i]
                share = float(h.max() / h.sum().clamp(min=1e-9))
                word = text[i] if i < len(text) else "—"
                print(f"    период записи {i + 1} (слово {word}): попыток записи {episodes}; ячейка {i + 1} до вывода — "
                      f"«{chat_exam.WORDS[int(h.argmax())]}», доля {share:.2f}")
            f = lambda v: " ".join(f"{float(x):5.1f}" for x in v)
            for i in range(len(text)):
                print(f"    запись слова {i + 1} «{text[i]}»: источники E1 из коры, Гц (А Б В Г Д) {f(diag_src[i])}; "
                      f"стриатум записи по каналам {f(diag_msn[i])}")
            for k in range(min(READS, len(text) + 1)):
                print(f"    чтение позиции {k + 1} (вывод 1): стриатум чтения {f(diag_read[0, k, 0])}; "
                      f"реле чтения по словам {f(diag_read[0, k, 1])}; ответ E1 {f(diag_read[0, k, 2])}")
            print(f"  {text:3s} → {' | '.join(runs)}   (в ячейках до вывода: {stored}; после: "
                  f"{''.join(chat_exam.WORDS[int(x)] for x in after)}; утечка E1 {leak_e1}, кора {leak_ctx}) "
                  f"{time.time() - t0:.0f} с", flush=True)
        n_words = sum(len(t) for t in texts)
        chat_exam.report_repeat(f"E1 в мозге, зерно {seed}, вход стриатума {s_mode}", [(texts, results)], None)
        print(f"  утечка при записи: ответ E1 > 10 Гц в {leaks_e1}/{n_words} окнах, чужое слово в коре — {leaks_ctx}/{n_words}; "
              f"повторная запись (ячейка поменяла слово за вывод) — {rewrites}/{n_words}; {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main(dict(kv.split("=", 1) for kv in sys.argv[1:]))
