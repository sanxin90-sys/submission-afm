"""Revisão (painel simulado): tau separado para o dia e a noite.
    dT/dt = a*R - (T - Teq)/tau(t),  tau(t) = tau_d se R > 0 (dia) e tau_n se R <= 0 (noite)
Mesmo protocolo da linha de base (fit_baseline.py): janelas de 5 dias reiniciadas na observação, RK4 com passo de
30 min, divisão cronológica 70/15/15 e bootstrap em blocos de janelas (B réplicas). a e Teq são comuns ao dia e à
noite. O ajuste parte do tau único e de dois chutes assimétricos (tau/2 e 2*tau), e fica com o menor erro.
Três grupos de séries:
- 'principal': os sete conjuntos principais (forçante Rn);
- 'jaru_ano': Jaru por ano e altura, estações seca e chuvosa (mesmas séries de estacoes_anos.py, inclusive as
  marcadas com problema de sensor, que ficam identificadas na coluna obs);
- 'perfil': os perfis verticais de alturas.py (mesmos instantes válidos em todas as alturas).
O tau noturno não depende do aquecimento radiativo do sensor durante o dia, que entra só em tau_d e em a; por isso
o perfil de tau_n testa se o mínimo de tau na copa é artefato do sensor.
Saída: dados/tau_dia_noite.csv."""
import os, numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.optimize import minimize
import fit_dry_model as F
import fit_baseline as L

DATA = F.DATA
B = int(os.environ.get('B', 100))
NPROC = int(os.environ.get('NPROC', 4))
GRUPOS = os.environ.get('GRUPOS', 'principal,jaru_ano,perfil').split(',')


def simulate2(a, td, tn, Teq, T, R):
    n, L_ = T.shape
    S = np.empty((n, L_)); S[:, 0] = T[:, 0]
    for j in range(1, L_):
        x, r = S[:, j - 1], R[:, j]
        tau = np.where(r > 0, td, tn)
        f = lambda y: a * r - (y - Teq) / tau
        k1 = f(x); k2 = f(x + 0.5 * F.DT * k1); k3 = f(x + 0.5 * F.DT * k2); k4 = f(x + F.DT * k3)
        S[:, j] = x + F.DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return S


def resid2(p, T, R, M):
    S = simulate2(np.exp(p[0]), np.exp(p[1]), np.exp(p[2]), p[3], T, R)
    m = M.copy(); m[:, 0] = False
    return (S - T)[m]


def loss2(p, T, R, M):
    e = resid2(p, T, R, M)
    return np.mean(e ** 2) if np.all(np.isfinite(e)) else 1e6


def fit2(T, R, M, p1=None, p0=None):
    """p1: ajuste de tau único (log a, log tau, Teq); p0: ponto de partida já de dois tau (bootstrap)."""
    if p0 is None:
        la, lt, Teq = p1
        best = None
        for dd, dn in [(0, 0), (-np.log(2), np.log(2)), (np.log(2), -np.log(2))]:
            r = minimize(loss2, [la, lt + dd, lt + dn, Teq], args=(T, R, M), method='Nelder-Mead',
                         options=dict(maxiter=6000, xatol=1e-6, fatol=1e-9))
            best = r if best is None or r.fun < best.fun else best
        p0 = best.x
    r = minimize(loss2, p0, args=(T, R, M), method='L-BFGS-B')
    return r.x if r.fun <= loss2(p0, T, R, M) else np.asarray(p0)


def run(job):
    grupo, site, alt, ano, est, z_h, obs, ws, seed = job
    rng = np.random.default_rng(seed)
    tr, va, te = F.split(ws)
    Ttr, Rtr, Mtr = F.pack(tr, 'atual'); Tte, Rte, Mte = F.pack(te, 'atual')
    p1 = L.fit(Ttr, Rtr, Mtr)
    p2 = fit2(Ttr, Rtr, Mtr, p1=p1)
    e1, e2 = L.resid(p1, Tte, Rte, Mte), resid2(p2, Tte, Rte, Mte)
    bs = []
    for _ in range(B):
        k = rng.integers(0, len(tr), len(tr))
        bs.append(fit2(*F.pack([tr[i] for i in k], 'atual'), p0=p2))
    bs = np.array(bs)
    td, tn = np.exp(bs[:, 1]), np.exp(bs[:, 2])
    dia = Rtr[Mtr] > 0
    row = dict(grupo=grupo, sitio=site, altura=alt, ano=ano, estacao=est, z_sobre_h=z_h, obs=obs,
               dias_em_janelas=round(sum(len(w[0]) for w in ws) / 48, 1), frac_dia=dia.mean(),
               tau_h=np.exp(p1[1]), a=np.exp(p2[0]), tau_dia_h=np.exp(p2[1]), tau_noite_h=np.exp(p2[2]),
               Teq_C=p2[3] - 273.15,
               rmse_ajuste_1tau_K=np.sqrt(L.loss(p1, Ttr, Rtr, Mtr)), rmse_ajuste_2tau_K=np.sqrt(loss2(p2, Ttr, Rtr, Mtr)),
               rmse_teste_1tau_K=np.sqrt(np.mean(e1 ** 2)), rmse_teste_2tau_K=np.sqrt(np.mean(e2 ** 2)), B=B)
    row['tau_dia_p05'], row['tau_dia_p95'] = np.percentile(td, [5, 95])
    row['tau_noite_p05'], row['tau_noite_p95'] = np.percentile(tn, [5, 95])
    row['razao_noite_dia'] = row['tau_noite_h'] / row['tau_dia_h']
    row['razao_p05'], row['razao_p95'] = np.percentile(tn / td, [5, 95])
    print(grupo, site, alt, ano, est, f"tau={row['tau_h']:.2f} dia={row['tau_dia_h']:.2f} "
          f"noite={row['tau_noite_h']:.2f} h; teste {row['rmse_teste_1tau_K']:.2f}->{row['rmse_teste_2tau_K']:.2f} K",
          flush=True)
    return row


def jobs_principal(seed):
    out = []
    for site in ['jaru', 'sinop', 'cuieiras', 'mujui']:
        df = pd.read_csv(os.path.join(DATA, f'{site}_30min.csv'), index_col=0, parse_dates=True)
        for est in df['estacao'].dropna().unique():
            ws = L.windows_with(df[df['estacao'] == est], 'Rn')
            if sum(len(w[0]) for w in ws) < 10 * 48:
                continue
            out.append(('principal', site, '', '', est, np.nan, '', ws, seed)); seed += 1
    return out, seed


def jobs_jaru(seed):
    import estacoes_anos as EA
    out = []
    for site, alt, g in EA.series():
        if site != 'jaru':
            continue
        g = g[g['estacao4'].isin(['seca', 'chuvosa'])]
        for (ano, est), sub in g.groupby(['ano_estacao', 'estacao4'], sort=False):
            sub = sub.sort_index().reindex(pd.date_range(sub.index.min(), sub.index.max(), freq='30min'))
            sub['ok_TR'] = sub['ok_TR'].fillna(False).astype(bool)
            ws = F.windows(sub)
            if sum(len(w[0]) for w in ws) < EA.MIN_DIAS * 48:
                continue
            obs = 'sensor' if (alt, int(ano)) in [('62m', 2009), ('22m', 2008)] else ''
            out.append(('jaru_ano', site, alt, ano, est, np.nan, obs, ws, seed)); seed += 1
    return out, seed


def jobs_perfil(seed):
    import alturas as AL
    out = []
    for site in ['cuieiras', 'sinop', 'mujui']:
        base = pd.read_csv(os.path.join(DATA, f'{site}_30min.csv'), index_col=0, parse_dates=True)
        prof = {'cuieiras': AL.perfil_cuieiras, 'sinop': AL.perfil_sinop}.get(site, lambda: AL.perfil_mujui(base))()
        prof = prof.reindex(base.index)
        for est in [e for e in base['estacao'].dropna().unique() if e]:
            e = (base['estacao'] == est) & base['Rn'].notna()
            zs = [z for z in prof.columns if (e & prof[z].notna()).sum() >= 20 * 48]
            m = e & prof[zs].notna().all(axis=1)
            for z in zs:
                sub = pd.DataFrame({'T_C': prof[z], 'Rn': base['Rn']})[base['estacao'] == est].copy()
                sub['T_K'] = sub['T_C'] + 273.15
                sub['ok_TR'] = m[base['estacao'] == est].values
                ws = F.windows(sub)
                if sum(len(w[0]) for w in ws) < 10 * 48:
                    continue
                out.append(('perfil', site, f'{float(z):g}m', '', est, float(z) / AL.DOSSEL[site], '', ws, seed))
                seed += 1
    return out, seed


def main():
    jobs, seed = [], 900
    for g, fn in [('principal', jobs_principal), ('jaru_ano', jobs_jaru), ('perfil', jobs_perfil)]:
        if g in GRUPOS:
            js, seed = fn(seed); jobs += js
    print(len(jobs), 'ajustes', flush=True)
    with Pool(NPROC) as pool:
        rows = pool.map(run, jobs, chunksize=1)
    out = os.path.join(DATA, 'tau_dia_noite.csv')
    pd.DataFrame(rows).to_csv(out, index=False, float_format='%.5g')
    print('salvo', out)


if __name__ == '__main__':
    main()
