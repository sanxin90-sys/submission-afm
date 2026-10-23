"""Etapa 4b: incerteza por bootstrap em blocos (janelas de 5 dias), como no Artigo 1,
mas reamostrando janelas inteiras em vez de pontos isolados.
Para cada membro b: reamostra as janelas de ajuste, reajusta a base linear, treina uma rede
(semente b) e integra as janelas de teste. Saídas por conjunto e modelo:
RMSE e viés da média do conjunto, faixa 5-95 % do RMSE dos membros, cobertura empírica
do intervalo de 90 % e fração de membros em que o modelo bate a base do mesmo membro."""
import os, numpy as np, pandas as pd
import fit_dry_model as F
import fit_baseline as BL
import fit_nn_correction as NN

DATA = F.DATA
NB = int(os.environ.get('NB', 30))
rng = np.random.default_rng(2026)


def trajectories(base, nets, cols, T, R, Q, M):
    a, tau, Teq = base
    n, L = T.shape
    S = np.empty((n, L)); S[:, 0] = T[:, 0]
    sel = [{'T': 0, 'R': 1, 'q': 2}[c] for c in cols]

    def f(y, r, q):
        corr = np.mean([p(np.c_[y, r, q][:, sel]) for p in nets], axis=0) if nets else 0.0
        return a * r - (y - Teq) / tau + corr
    for j in range(1, L):
        x, r, q = S[:, j - 1], R[:, j], Q[:, j]
        k1 = f(x, r, q); k2 = f(x + 0.5 * F.DT * k1, r, q); k3 = f(x + 0.5 * F.DT * k2, r, q)
        k4 = f(x + F.DT * k3, r, q)
        S[:, j] = np.clip(x + F.DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4), 250, 330)
    return S


def main():
    rows = []
    for site in BL.SITES:
        df = pd.read_csv(os.path.join(DATA, f'{site}_30min.csv'), index_col=0, parse_dates=True)
        for est in df['estacao'].dropna().unique():
            ws = NN.windows_q(df[df['estacao'] == est])
            if sum(len(w[0]) for w in ws) < 10 * 48:
                continue
            tr, va, te = F.split(ws)
            if F.skip(site):
                for b in range(NB):
                    rng.integers(0, len(tr), len(tr))
                continue
            Ttr, Rtr, Qtr, Mtr = NN.packq(tr)
            p_full = BL.fit(Ttr, Rtr, Mtr)
            Tte, Rte, Qte, Mte = NN.packq(te)
            m = Mte.copy(); m[:, 0] = False
            obs = Tte[m]
            sims = {'base': [], 'NN(T,R)': [], 'NN(T,R,q)': []}
            for b in range(NB):
                k = rng.integers(0, len(tr), len(tr))
                trb = [tr[i] for i in k]
                Tb, Rb, Qb, Mb = NN.packq(trb)
                p = BL.fit(Tb, Rb, Mb, p0=p_full)
                base = (np.exp(p[0]), np.exp(p[1]), p[2])
                sims['base'].append(trajectories(base, [], ['T', 'R'], Tte, Rte, Qte, Mte)[m])
                Xtr, ytr = NN.samples(trb, base); Xva, yva = NN.samples(va, base)
                for name, cols in NN.VARIANTS.items():
                    sel = [{'T': 0, 'R': 1, 'q': 2}[c] for c in cols]
                    pr, _, _ = NN.train(Xtr[:, sel], ytr, Xva[:, sel], yva, seed=1000 + b)
                    sims[name].append(trajectories(base, [pr], cols, Tte, Rte, Qte, Mte)[m])
                print(site, est, 'membro', b + 1, flush=True)
            rmse_b = {k: np.sqrt(np.mean((np.array(v) - obs) ** 2, axis=1)) for k, v in sims.items()}
            for name, v in sims.items():
                v = np.array(v)
                mean = v.mean(0); lo, hi = np.percentile(v, [5, 95], axis=0)
                rows.append(dict(sitio=site, estacao=est, modelo=name, membros=NB,
                                 rmse_media_conjunto_K=np.sqrt(np.mean((mean - obs) ** 2)),
                                 vies_media_conjunto_K=np.mean(mean - obs),
                                 rmse_membros_p05_K=np.percentile(rmse_b[name], 5),
                                 rmse_membros_p95_K=np.percentile(rmse_b[name], 95),
                                 cobertura_90=np.mean((obs >= lo) & (obs <= hi)),
                                 largura_media_IC90_K=np.mean(hi - lo),
                                 frac_membros_melhor_que_base=np.mean(rmse_b[name] < rmse_b['base'])
                                 if name != 'base' else np.nan))
            F.save_merged(rows, os.path.join(DATA, 'bootstrap_nn_resultados.csv'), '%.4g')


if __name__ == '__main__':
    main()
