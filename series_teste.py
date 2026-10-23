"""Séries simuladas nas janelas de teste, para a figura de observado x simulado.
Mesmas janelas, divisão e linha de base de fit_nn_correction.py (exige q válido), com:
- base linear (reajustada como em fit_nn_correction.py)
- base + NN(T,R): média de N_SEEDS redes (sementes 0..N_SEEDS-1, como em fit_nn_correction.py)
- base + NN(T,R) treinada na trajetória (fit_ude.py, pesos em dados/ude_modelos/), coluna T_base_UDE
- modelo seco do Artigo 1 (a R - g T^4) com os parâmetros ajustados no protocolo do Artigo 2
  (artigo1_aplicado.csv, colunas a_T4_A1 e g_T4_A1, em unidades de 10^6 s)
- faixa de 90 % da base: quantis 5 % e 95 % de (obs - sim) nas janelas de validação
Saídas: dados/series_teste.csv (formato longo) e dados/series_teste_janelas.csv (RMSE por janela)."""
import os, numpy as np, pandas as pd
from multiprocessing import Pool
import fit_dry_model as F
import fit_baseline as BL
import fit_nn_correction as NN
import artigo1_aplicado as A

DATA = F.DATA
N_SEEDS = int(os.environ.get('N_SEEDS', 5))
NPROC = int(os.environ.get('NPROC', 2))


def simulate_series(f, T, R, Q):
    n, L = T.shape
    S = np.empty((n, L)); S[:, 0] = T[:, 0]
    for j in range(1, L):
        x, r, q = S[:, j - 1], R[:, j], Q[:, j]
        k1 = f(x, r, q); k2 = f(x + 0.5 * F.DT * k1, r, q); k3 = f(x + 0.5 * F.DT * k2, r, q)
        k4 = f(x + F.DT * k3, r, q)
        S[:, j] = np.clip(x + F.DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4), 250, 330)
    return S


def run(job):
    site, est = job
    import torch
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)       # redes treinadas na derivada: float32, como em fit_nn_correction.py
    df = pd.read_csv(os.path.join(DATA, f'{site}_30min.csv'), index_col=0, parse_dates=True)
    ws = NN.windows_q(df[df['estacao'] == est])
    tr, va, te = F.split(ws)
    Ttr, Rtr, Qtr, Mtr = NN.packq(tr)
    p = BL.fit(Ttr, Rtr, Mtr)
    a, tau, Teq = np.exp(p[0]), np.exp(p[1]), p[2]
    Xtr, ytr = NN.samples(tr, (a, tau, Teq)); Xva, yva = NN.samples(va, (a, tau, Teq))
    nets = [NN.train(Xtr[:, :2], ytr, Xva[:, :2], yva, seed=s)[0] for s in range(N_SEEDS)]
    t4 = pd.read_csv(os.path.join(DATA, 'artigo1_aplicado.csv'))
    t4 = t4[(t4['sitio'] == site) & (t4['estacao'] == est)].iloc[0]
    a4, g4 = t4['a_T4_A1'] / A.U, t4['g_T4_A1'] / A.U
    lin = lambda y, r, q: a * r - (y - Teq) / tau
    nn = lambda y, r, q: lin(y, r, q) + np.mean([pr(np.c_[y, r]) for pr in nets], axis=0)
    dry = lambda y, r, q: a4 * r - g4 * y ** 4
    Tva, Rva, Qva, Mva = NN.packq(va)
    mv = Mva.copy(); mv[:, 0] = False
    ev = (Tva - simulate_series(lin, Tva, Rva, Qva))[mv]
    q05, q95 = np.percentile(ev, [5, 95])
    T, R, Q, M = NN.packq(te)
    sims = {nm: simulate_series(f, T, R, Q) for nm, f in [('T_base', lin), ('T_base_NN', nn), ('T_seco_A1', dry)]}
    import glob, fit_ude as U
    torch.set_default_dtype(torch.float64)       # redes treinadas na trajetória: float64
    umods = []
    for fn in sorted(glob.glob(os.path.join(U.MODEL_DIR, f'{site}_{est}_semente*.pt'))):
        dd = torch.load(fn, weights_only=False)
        mo = U.UDE(*dd['scale']); mo.load_state_dict(dd['state']); mo.eval(); umods.append(mo)
        assert np.allclose(dd['base'], (a, tau, Teq))
    if umods:
        with torch.no_grad():
            Tt = torch.as_tensor(np.where(M, T, 0.0))
            sims['T_base_UDE'] = U.integrate((a, tau, Teq), umods, Tt[:, 0], torch.as_tensor(R), clip=True).numpy()
    rows, wrows = [], []
    m0 = M.copy(); m0[:, 0] = False
    for i, w in enumerate(te):
        n = len(w[0])
        t = pd.date_range(w[3], periods=n, freq='30min')
        d = pd.DataFrame({'sitio': site, 'estacao': est, 'janela': i, 'fim_intervalo': t, 'T_obs': T[i, :n],
                          'Rn': R[i, :n], **{k: v[i, :n] for k, v in sims.items()}})
        d['base_p05'], d['base_p95'] = d['T_base'] + q05, d['T_base'] + q95
        rows.append(d)
        mm = m0[i, :n]
        wr = dict(sitio=site, estacao=est, janela=i, inicio=t[0], fim=t[-1], pontos=n)
        for k, v in sims.items():
            wr['rmse_' + k] = np.sqrt(np.mean((v[i, :n][mm] - T[i, :n][mm]) ** 2))
        wrows.append(wr)
    tot = {k: np.sqrt(np.mean((v - T)[m0] ** 2)) for k, v in sims.items()}
    print(site, est, {k: round(v, 3) for k, v in tot.items()}, 'q05/q95', round(q05, 2), round(q95, 2), flush=True)
    return pd.concat(rows), pd.DataFrame(wrows)


def main():
    jobs = [('jaru', 'chuvosa'), ('jaru', 'seca'), ('cuieiras', 'seca'), ('cuieiras', 'chuvosa'),
            ('sinop', 'seca'), ('mujui', 'terrestre'), ('mujui', 'inundada')]
    with Pool(NPROC) as pool:
        res = pool.map(run, jobs, chunksize=1)
    pd.concat([r[0] for r in res]).to_csv(os.path.join(DATA, 'series_teste.csv'), index=False, float_format='%.5g')
    pd.concat([r[1] for r in res]).to_csv(os.path.join(DATA, 'series_teste_janelas.csv'), index=False,
                                          float_format='%.4g')


if __name__ == '__main__':
    main()
