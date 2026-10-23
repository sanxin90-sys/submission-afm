"""Revisão (parecer dos colegas): simulações contínuas, sem reinício.
Cada trecho contínuo de cada conjunto principal (forçante Rn) é integrado de uma vez, do primeiro ao último ponto
(até 93 dias), a partir da temperatura observada no início do trecho, por RK4 com passo de 30 min e o Rn do
próprio intervalo. Modelos, com os parâmetros já ajustados nas janelas de ajuste:
- linha de base (linha_base_parametros.csv);
- modelo seco (a*R - g*T^4), reajustado aqui com o protocolo de fit_dry_model.py nas mesmas janelas;
- linha de base + RN treinada na derivada (fit_nn_correction.py, cinco sementes, refeitas aqui);
- linha de base + RN treinada na trajetória (fit_ude.py, pesos em dados/ude_modelos/).
O erro é medido nos mesmos pontos de teste da avaliação por janelas (sem o primeiro ponto de cada janela), para
comparar o RMSE com e sem reinício; também é dado o RMSE em todos os pontos após as primeiras 24 h de cada trecho.
Saída: dados/simulacao_continua.csv."""
import os, glob, numpy as np, pandas as pd, torch
from multiprocessing import Pool
import fit_dry_model as F
import fit_baseline as BL
import fit_nn_correction as NN

DATA = F.DATA
NPROC = int(os.environ.get('NPROC', 2))
N_SEEDS = int(os.environ.get('N_SEEDS', 5))
ONLY = os.environ.get('ONLY_SETS')   # ex.: "jaru:seca,sinop:seca"


def rk4(f, T0, R):
    S = np.empty(len(R)); S[0] = x = T0
    for j in range(1, len(R)):
        r = R[j]
        k1 = f(x, r); k2 = f(x + 0.5 * F.DT * k1, r); k3 = f(x + 0.5 * F.DT * k2, r); k4 = f(x + F.DT * k3, r)
        x = min(max(x + F.DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4), 250.0), 330.0)
        S[j] = x
    return S


def load_ude(site, est):
    import fit_ude as U
    torch.set_default_dtype(torch.float64)       # redes treinadas na trajetória: float64
    models, base = [], None
    for fn in sorted(glob.glob(os.path.join(U.MODEL_DIR, f'{site}_{est}_semente*.pt'))):
        d = torch.load(fn, weights_only=False)
        m = U.UDE(*d['scale']); m.load_state_dict(d['state']); m.eval()
        models.append(m); base = d['base']
    return base, models


def run(job):
    site, est = job
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)       # redes treinadas na derivada: float32, como em fit_nn_correction.py
    df = pd.read_csv(os.path.join(DATA, f'{site}_30min.csv'), index_col=0, parse_dates=True)
    sub = df[df['estacao'] == est].copy()
    sub['ok_TR'] = sub['T_C'].notna() & sub['Rn'].notna()
    ws = F.windows(sub)
    tr, va, te = F.split(ws)
    # pontos de teste (sem o primeiro de cada janela), por instante
    test_t = set()
    for T, R, t0 in te:
        test_t.update(pd.date_range(t0, periods=len(T), freq='30min')[1:])
    # parâmetros
    lb = pd.read_csv(os.path.join(DATA, 'linha_base_parametros.csv'))
    lb = lb[(lb['sitio'] == site) & (lb['estacao'] == est) & (lb['forcante'] == 'Rn')].iloc[0]
    a, tau, Teq = lb['a'], lb['tau_h'], lb['Teq_C'] + 273.15
    Ttr, Rtr, Mtr = F.pack(tr, 'atual')
    a4, g4, _, _ = F.fit(Ttr, Rtr, Mtr)
    # RN na derivada (como fit_nn_correction.py: janelas com q válido, base reajustada nelas)
    wq = NN.windows_q(sub)
    trq, vaq, teq = F.split(wq)
    Tq, Rq, Qq, Mq = NN.packq(trq)
    pq = BL.fit(Tq, Rq, Mq)
    bq = (np.exp(pq[0]), np.exp(pq[1]), pq[2])
    Xtr, ytr = NN.samples(trq, bq); Xva, yva = NN.samples(vaq, bq)
    nets = [NN.train(Xtr[:, :2], ytr, Xva[:, :2], yva, seed=s)[0] for s in range(N_SEEDS)]
    ub, umods = load_ude(site, est)
    models = {
        'base': lambda y, r: a * r - (y - Teq) / tau,
        'seco': lambda y, r: a4 * r - g4 * y ** 4,
        'base_RN_derivada': lambda y, r: bq[0] * r - (y - bq[2]) / bq[1]
        + np.mean([p(np.array([[y, r]]))[0] for p in nets]),
    }
    if umods:
        def f_ude(y, r):
            with torch.no_grad():
                yt, rt = torch.tensor([y], dtype=torch.float64), torch.tensor([r], dtype=torch.float64)
                c = torch.stack([m(yt, rt) for m in umods]).mean().item()
            return ub[0] * r - (y - ub[2]) / ub[1] + c
        models['base_RN_trajetoria'] = f_ude
    segs = F.windows_full(sub)
    rows = []
    for nm, f in models.items():
        e_test_c, e_after24 = [], []
        e_test_w = []
        maxdias = 0.0
        for T, R, t0 in segs:
            t = pd.date_range(t0, periods=len(T), freq='30min')
            S = rk4(f, T[0], R)
            mt = np.array([x in test_t for x in t])
            e_test_c.append((S - T)[mt]); e_after24.append((S - T)[48:])
            if mt.any():
                maxdias = max(maxdias, (np.nonzero(mt)[0].max()) / 48)
            # mesma série, com reinício a cada janela de 5 dias (avaliação por janelas)
            Sw = np.empty(len(T))
            for k in range(0, len(T), F.WIN):
                Sw[k:k + F.WIN] = rk4(f, T[k], R[k:k + F.WIN])
            e_test_w.append((Sw - T)[mt])
        ec, ew, ea = np.concatenate(e_test_c), np.concatenate(e_test_w), np.concatenate(e_after24)
        rows.append(dict(sitio=site, estacao=est, modelo=nm, trechos=len(segs),
                         maior_trecho_dias=round(max(len(s[0]) for s in segs) / 48, 1),
                         max_dias_ate_ponto_de_teste=round(maxdias, 1), pontos_teste=len(ec),
                         rmse_teste_janelas_K=np.sqrt(np.mean(ew ** 2)), vies_teste_janelas_K=np.mean(ew),
                         rmse_teste_continua_K=np.sqrt(np.mean(ec ** 2)), vies_teste_continua_K=np.mean(ec),
                         rmse_continua_apos_24h_K=np.sqrt(np.mean(ea ** 2)), vies_continua_apos_24h_K=np.mean(ea)))
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    return rows


def main():
    par = pd.read_csv(os.path.join(DATA, 'linha_base_parametros.csv'))
    jobs = [tuple(x) for x in par.loc[par['forcante'] == 'Rn', ['sitio', 'estacao']].values]
    if ONLY:
        jobs = [j for j in jobs if f'{j[0]}:{j[1]}' in ONLY.split(',')]
    with Pool(NPROC) as pool:
        res = pool.map(run, jobs, chunksize=1)
    out = os.path.join(DATA, os.environ.get('SIM_OUT', 'simulacao_continua.csv'))
    pd.DataFrame([r for rs in res for r in rs]).to_csv(out, index=False, float_format='%.5g')


if __name__ == '__main__':
    main()
