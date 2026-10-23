"""Etapa 3 (versão adotada): linha de base de relaxação linear
    dT/dt = a*R - (T - Teq)/tau
Mesmo protocolo de fit_dry_model.py (tiro múltiplo em janelas de 5 dias, divisão
cronológica 70/15/15, R do intervalo atual, bootstrap em blocos B = 200).
Sinop também roda com Rsd como forçante (sensibilidade; única opção na estação chuvosa)."""
import os, sys, numpy as np, pandas as pd
from scipy.optimize import minimize
import fit_dry_model as F

DATA = F.DATA
B = 200
rng = np.random.default_rng(7)
SITES = ['jaru', 'sinop', 'cuieiras', 'mujui']


def simulate_lin(a, tau, Teq, T, R, M):
    n, L = T.shape
    S = np.empty((n, L)); S[:, 0] = T[:, 0]
    for j in range(1, L):
        x, r = S[:, j - 1], R[:, j]
        f = lambda y: a * r - (y - Teq) / tau
        k1 = f(x); k2 = f(x + 0.5 * F.DT * k1); k3 = f(x + 0.5 * F.DT * k2); k4 = f(x + F.DT * k3)
        S[:, j] = x + F.DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return S


def resid(p, T, R, M):
    S = simulate_lin(np.exp(p[0]), np.exp(p[1]), p[2], T, R, M)
    m = M.copy(); m[:, 0] = False
    return (S - T)[m]


def loss(p, T, R, M):
    e = resid(p, T, R, M)
    return np.mean(e ** 2) if np.all(np.isfinite(e)) else 1e6


def fit(T, R, M, p0=None):
    if p0 is None:
        Tm, Rm = np.nanmean(T[M]), np.mean(R[M])
        best = None
        for tau0 in [1, 3, 10, 30]:
            a0 = 4.0 / (np.percentile(R[M], 99) * tau0)   # amplitude diurna ~4 K
            r = minimize(loss, [np.log(a0), np.log(tau0), Tm - a0 * Rm * tau0], args=(T, R, M),
                         method='Nelder-Mead', options=dict(maxiter=4000, xatol=1e-6, fatol=1e-9))
            best = r if best is None or r.fun < best.fun else best
        p0 = best.x
    r = minimize(loss, p0, args=(T, R, M), method='L-BFGS-B')
    return r.x if r.fun <= loss(p0, T, R, M) else np.asarray(p0)


def windows_with(df, rcol):
    d = df.copy()
    d['Rn'] = d[rcol]
    d['ok_TR'] = d['T_C'].notna() & d['Rn'].notna()
    return F.windows(d)


def main():
    rows = []
    for site in SITES:
        df = pd.read_csv(os.path.join(DATA, f'{site}_30min.csv'), index_col=0, parse_dates=True)
        forc = ['Rn'] + (['Rsd'] if site == 'sinop' else [])
        for rcol in forc:
            for est in df['estacao'].dropna().unique():
                ws = windows_with(df[df['estacao'] == est], rcol)
                if sum(len(w[0]) for w in ws) < 10 * 48:
                    print(site, est, rcol, 'sem dados suficientes'); continue
                tr, va, te = F.split(ws)
                if F.skip(site):
                    for _ in range(B):
                        rng.integers(0, len(tr), len(tr))
                    continue
                Ttr, Rtr, Mtr = F.pack(tr, 'atual'); Tte, Rte, Mte = F.pack(te, 'atual')
                p = fit(Ttr, Rtr, Mtr)
                etr, ete = resid(p, Ttr, Rtr, Mtr), resid(p, Tte, Rte, Mte)
                bs = []
                for _ in range(B):
                    k = rng.integers(0, len(tr), len(tr))
                    Tb, Rb, Mb = F.pack([tr[i] for i in k], 'atual')
                    bs.append(fit(Tb, Rb, Mb, p0=p))
                bs = np.array(bs)
                par = np.c_[np.exp(bs[:, 0]), np.exp(bs[:, 1]), bs[:, 2] - 273.15]
                C = np.corrcoef(par.T)
                row = dict(sitio=site, estacao=est, forcante=rcol, dias_ajuste=round(Mtr.sum() / 48, 1),
                           dias_teste=round(Mte.sum() / 48, 1),
                           a=np.exp(p[0]), tau_h=np.exp(p[1]), Teq_C=p[2] - 273.15,
                           rmse_ajuste_K=np.sqrt(np.mean(etr ** 2)), rmse_teste_K=np.sqrt(np.mean(ete ** 2)),
                           vies_teste_K=np.mean(ete),
                           corr_a_tau=C[0, 1], corr_a_Teq=C[0, 2], corr_tau_Teq=C[1, 2])
                for j, nm in enumerate(['a', 'tau_h', 'Teq_C']):
                    row[nm + '_p05'], row[nm + '_p95'] = np.percentile(par[:, j], [5, 95])
                rows.append(row)
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    F.save_merged(rows, os.path.join(DATA, 'linha_base_parametros.csv'), '%.6g')


if __name__ == '__main__':
    main()
