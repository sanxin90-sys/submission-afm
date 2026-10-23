"""Revisão (parecer dos colegas): temperatura de equilíbrio variável no tempo.
    dT/dt = a*R - (T - Teq(t))/tau
Teq(t) sai do balanço médio das 24 h anteriores (média móvel de baixa frequência da temperatura):
    Teq = T24 - a*tau*R24,
em que T24 e R24 são as médias da temperatura e do saldo de radiação observados nas 48 meias horas anteriores.
Não há parâmetro novo: a e tau são ajustados como na linha de base, e Teq0 (constante) só é usado quando não há
24 h anteriores no mesmo trecho contínuo (início de trecho).
Variantes:
- constante: a linha de base (Teq fixo na estação);
- janela (prognóstica): Teq fixado no início de cada janela pelas 24 h anteriores a ela e mantido por 5 dias; só usa
  informação disponível antes da previsão;
- contínua (diagnóstica): Teq atualizado a cada passo pelas 24 h anteriores observadas; usa a temperatura observada
  durante a janela, então mede quanto do erro vem do fundo de baixa frequência, e não é uma previsão.
Mesmas janelas, divisão 70/15/15, forçante e função objetivo de fit_baseline.py.
Saídas: dados/teq_variavel.csv (conjuntos principais, com bootstrap em blocos de a e tau nas variantes janela e
contínua) e
dados/teq_variavel_anos.csv (cada sítio, altura, ano e estação de estacoes_anos.py, sem bootstrap)."""
import os, numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.optimize import minimize
import fit_dry_model as F
import fit_baseline as L

DATA = F.DATA
B = int(os.environ.get('B', 100))
NPROC = int(os.environ.get('NPROC', 4))
HMAX = 240          # até 5 dias de pontos anteriores guardados por janela


def windows_ctx(df):
    """Como F.windows, mas guarda até 48 pontos válidos anteriores a cada janela, no mesmo trecho contínuo."""
    ok = df['ok_TR'].values
    idx = df.index
    brk = np.r_[True, (ok[1:] != ok[:-1]) | (np.diff(idx.values) != np.timedelta64(30, 'm'))]
    sid = np.cumsum(brk)
    out = []
    for s in np.unique(sid[ok]):
        m = (sid == s) & ok
        T, R, t = df['T_K'].values[m], df['Rn'].values[m], idx[m]
        if len(T) < F.MIN_WIN:
            continue
        for k in range(0, len(T), F.WIN):
            if len(T) - k >= F.MIN_WIN:
                k0 = max(0, k - HMAX)
                out.append((T[k:k + F.WIN], R[k:k + F.WIN], t[k], T[k0:k], R[k0:k]))
    return out


def pack_ctx(ws, H=48):
    """T, R, M (como F.pack, R atual) e as médias dos H pontos anteriores: por janela (Tb, Rb) e por passo (Tc, Rc)."""
    T, R, M = F.pack([(w[0], w[1], w[2]) for w in ws], 'atual')
    n, Lw = T.shape
    Tb, Rb = np.full(n, np.nan), np.full(n, np.nan)
    Tc, Rc = np.full((n, Lw), np.nan), np.full((n, Lw), np.nan)
    for i, (Tw, Rw, _, Tp, Rp) in enumerate(ws):
        if len(Tp) >= H:
            Tb[i], Rb[i] = Tp[-H:].mean(), Rp[-H:].mean()
        ET, ER = np.r_[Tp, Tw], np.r_[Rp, Rw]
        cT, cR = np.r_[0.0, np.cumsum(ET)], np.r_[0.0, np.cumsum(ER)]
        p = len(Tp)
        for j in range(1, len(Tw)):
            e = p + j          # passo j-1 -> j usa as observações de e-H a e-1 (até o início do passo)
            if e - H >= 0:
                Tc[i, j] = (cT[e] - cT[e - H]) / H
                Rc[i, j] = (cR[e] - cR[e - H]) / H
    return dict(T=T, R=R, M=M, Tb=Tb, Rb=Rb, Tc=Tc, Rc=Rc)


def teq_array(var, a, tau, Teq0, D):
    n, Lw = D['T'].shape
    if var == 'constante':
        return np.full((n, Lw), Teq0)
    if var == 'janela':
        tb = D['Tb'] - a * tau * D['Rb']
        return np.repeat(np.where(np.isfinite(tb), tb, Teq0)[:, None], Lw, axis=1)
    tc = D['Tc'] - a * tau * D['Rc']
    return np.where(np.isfinite(tc), tc, Teq0)


def simulate(a, tau, Te, T, R):
    n, Lw = T.shape
    S = np.empty((n, Lw)); S[:, 0] = T[:, 0]
    for j in range(1, Lw):
        x, r, te = S[:, j - 1], R[:, j], Te[:, j]
        f = lambda y: a * r - (y - te) / tau
        k1 = f(x); k2 = f(x + 0.5 * F.DT * k1); k3 = f(x + 0.5 * F.DT * k2); k4 = f(x + F.DT * k3)
        S[:, j] = x + F.DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return S


def resid(var, p, D):
    a, tau = np.exp(p[0]), np.exp(p[1])
    S = simulate(a, tau, teq_array(var, a, tau, p[2], D), D['T'], D['R'])
    m = D['M'].copy(); m[:, 0] = False
    return (S - D['T'])[m]


def loss(p, var, D):
    e = resid(var, p, D)
    return np.mean(e ** 2) if np.all(np.isfinite(e)) else 1e6


def fit(var, D, p0):
    r = minimize(loss, p0, args=(var, D), method='Nelder-Mead',
                 options=dict(maxiter=4000, xatol=1e-6, fatol=1e-9))
    r2 = minimize(loss, r.x, args=(var, D), method='L-BFGS-B')
    return r2.x if r2.fun <= r.fun else r.x


def evaluate(sub, tag, boot=0, seed=0):
    """Ajusta as três variantes no conjunto (DataFrame com T_K, Rn, ok_TR) e devolve uma linha por variante."""
    ws = windows_ctx(sub)
    ref = F.windows(sub)
    assert len(ws) == len(ref) and all(np.array_equal(a[0], b[0]) for a, b in zip(ws, ref))
    tr, va, te = F.split(ws)
    Dtr, Dte = pack_ctx(tr), pack_ctx(te)
    pc = L.fit(Dtr['T'], Dtr['R'], Dtr['M'])          # linha de base (constante), como em fit_baseline.py
    rows = []
    frac_tr = np.mean(np.isfinite(Dtr['Tb'])); frac_te = np.mean(np.isfinite(Dte['Tb']))
    for var in ['constante', 'janela', 'continua']:
        p = pc if var == 'constante' else fit(var, Dtr, pc)
        etr, ete = resid(var, p, Dtr), resid(var, p, Dte)
        row = dict(tag, variante=var, a=np.exp(p[0]), tau_h=np.exp(p[1]), Teq0_C=p[2] - 273.15,
                   rmse_ajuste_K=np.sqrt(np.mean(etr ** 2)), rmse_teste_K=np.sqrt(np.mean(ete ** 2)),
                   vies_teste_K=np.mean(ete), janelas_ajuste=len(tr), janelas_teste=len(te),
                   frac_janelas_ajuste_com_24h=frac_tr, frac_janelas_teste_com_24h=frac_te)
        if var != 'constante':
            Te = teq_array(var, np.exp(p[0]), np.exp(p[1]), p[2], Dte)
            row['Teq_teste_medio_C'] = np.nanmean(Te[Dte['M']]) - 273.15
        if boot and var != 'constante':
            rng = np.random.default_rng(seed)
            bs = []
            for _ in range(boot):
                k = rng.integers(0, len(tr), len(tr))
                bs.append(fit(var, pack_ctx([tr[i] for i in k]), p))
            bs = np.array(bs)
            row['tau_h_p05'], row['tau_h_p95'] = np.percentile(np.exp(bs[:, 1]), [5, 95])
            row['a_p05'], row['a_p95'] = np.percentile(np.exp(bs[:, 0]), [5, 95])
        rows.append(row)
    return rows


def run_main(job):
    site, est = job
    df = pd.read_csv(os.path.join(DATA, f'{site}_30min.csv'), index_col=0, parse_dates=True)
    sub = df[df['estacao'] == est].copy()
    sub['ok_TR'] = sub['T_C'].notna() & sub['Rn'].notna()
    rows = evaluate(sub, dict(sitio=site, estacao=est), boot=B, seed=7)
    for r in rows:
        print(site, est, r['variante'], f"tau={r['tau_h']:.2f} h RMSE={r['rmse_teste_K']:.2f} "
              f"viés={r['vies_teste_K']:+.2f}", flush=True)
    return rows


def run_year(job):
    site, alt, ano, est, sub = job
    ws = F.windows(sub)
    if sum(len(w[0]) for w in ws) / 48 < 20:      # MIN_DIAS de estacoes_anos.py
        return []
    rows = evaluate(sub, dict(sitio=site, altura=alt, ano=ano, estacao=est))
    print(site, alt, ano, est, ' '.join(f"{r['variante']}: {r['rmse_teste_K']:.2f}/{r['vies_teste_K']:+.2f}"
                                        for r in rows), flush=True)
    return rows


def main():
    import estacoes_anos as EA
    par = pd.read_csv(os.path.join(DATA, 'linha_base_parametros.csv'))
    jobs = [tuple(x) for x in par.loc[par['forcante'] == 'Rn', ['sitio', 'estacao']].values]
    with Pool(NPROC) as pool:
        res = pool.map(run_main, jobs, chunksize=1)
    pd.DataFrame([r for rs in res for r in rs]).to_csv(os.path.join(DATA, 'teq_variavel.csv'), index=False,
                                                       float_format='%.5g')
    yjobs = []
    for site, alt, g in EA.series():
        g = g[g['estacao4'].notna()]
        for (ano, est), sub in g.groupby(['ano_estacao', 'estacao4'], sort=False):
            sub = sub.sort_index()
            sub = sub.reindex(pd.date_range(sub.index.min(), sub.index.max(), freq='30min'))
            sub['ok_TR'] = sub['ok_TR'].fillna(False).astype(bool)
            yjobs.append((site, alt, ano, est, sub))
    with Pool(NPROC) as pool:
        res = pool.map(run_year, yjobs, chunksize=1)
    pd.DataFrame([r for rs in res for r in rs]).to_csv(os.path.join(DATA, 'teq_variavel_anos.csv'), index=False,
                                                       float_format='%.5g')


if __name__ == '__main__':
    main()
