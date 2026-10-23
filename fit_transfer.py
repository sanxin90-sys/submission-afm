"""Etapa 6 (H3): transferência da linha de base linear entre sítios e estações.
Para cada par fonte -> alvo, simula as janelas de teste do alvo com os parâmetros
ajustados na fonte (linha_base_parametros.csv, forçante Rn). Duas variantes:
- completa: a, tau e Teq da fonte
- dinamica: a e tau da fonte, Teq reajustado nas janelas de ajuste do alvo
  (calibração local de um só número, a temperatura de equilíbrio)
- so_tau: só tau da fonte; a e Teq reajustados no alvo (a depende da calibração do saldo-radiômetro,
  que muda entre sítios; tau e Teq não mudam com um erro multiplicativo de Rn)
Mesmo protocolo de fit_baseline.py (janelas de 5 dias, divisão cronológica 70/15/15)."""
import os, numpy as np, pandas as pd
from scipy.optimize import minimize_scalar, minimize
import fit_dry_model as F
import fit_baseline as L

DATA = F.DATA


def rmse(a, tau, Teq, T, R, M):
    e = L.resid([np.log(a), np.log(tau), Teq], T, R, M)
    return np.sqrt(np.mean(e ** 2)), np.mean(e)


def main():
    par = pd.read_csv(os.path.join(DATA, 'linha_base_parametros.csv'))
    par = par[par['forcante'] == 'Rn'].reset_index(drop=True)
    sets = {}
    for site in par['sitio'].unique():
        df = pd.read_csv(os.path.join(DATA, f'{site}_30min.csv'), index_col=0, parse_dates=True)
        for est in par.loc[par['sitio'] == site, 'estacao']:
            tr, va, te = F.split(L.windows_with(df[df['estacao'] == est], 'Rn'))
            sets[(site, est)] = (F.pack(tr, 'atual'), F.pack(te, 'atual'))
    rows = []
    for _, s in par.iterrows():
        for _, t in par.iterrows():
            (Ttr, Rtr, Mtr), (Tte, Rte, Mte) = sets[(t['sitio'], t['estacao'])]
            a, tau = s['a'], s['tau_h']
            r_full, b_full = rmse(a, tau, s['Teq_C'] + 273.15, Tte, Rte, Mte)
            opt = minimize_scalar(lambda q: rmse(a, tau, q, Ttr, Rtr, Mtr)[0],
                                  bounds=(273.15 + 10, 273.15 + 35), method='bounded')
            r_dyn, b_dyn = rmse(a, tau, opt.x, Tte, Rte, Mte)
            o2 = minimize(lambda u: L.loss([u[0], np.log(tau), u[1]], Ttr, Rtr, Mtr),
                          [np.log(t['a']), t['Teq_C'] + 273.15], method='Nelder-Mead',
                          options=dict(xatol=1e-6, fatol=1e-9, maxiter=2000))
            r_tau, b_tau = rmse(np.exp(o2.x[0]), tau, o2.x[1], Tte, Rte, Mte)
            rows.append(dict(fonte=f"{s['sitio']} {s['estacao']}", alvo=f"{t['sitio']} {t['estacao']}",
                             rmse_completa_K=r_full, vies_completa_K=b_full,
                             rmse_dinamica_K=r_dyn, vies_dinamica_K=b_dyn,
                             Teq_reajustado_C=opt.x - 273.15, rmse_so_tau_K=r_tau, vies_so_tau_K=b_tau,
                             a_so_tau=np.exp(o2.x[0]), Teq_so_tau_C=o2.x[1] - 273.15,
                             rmse_local_K=t['rmse_teste_K']))
            print(rows[-1]['fonte'], '->', rows[-1]['alvo'],
                  f"completa {r_full:.2f} dinamica {r_dyn:.2f} so_tau {r_tau:.2f} local {t['rmse_teste_K']:.2f}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(DATA, 'transferencia_base.csv'), index=False, float_format='%.6g')


if __name__ == '__main__':
    main()
