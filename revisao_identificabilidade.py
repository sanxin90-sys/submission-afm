"""Revisão (pareceres de outubro de 2026): identificabilidade de τ, protocolo de divisão, sensibilidade ao comprimento
das janelas, cobertura do intervalo de predição e associação entre τ e a umidade relativa controlada pelo sítio.

(1) Perfil da função objetivo em τ. Com τ fixo, a Equação 4 integrada por RK4 (passo de 30 min, R do próprio
    intervalo) é linear na condição inicial, em α e em Teq: S = X0 + α X1 + Teq X2. Para cada τ de uma grade
    logarítmica, α e Teq ótimos saem de mínimos quadrados exatos, e o RMSE de ajuste resultante é o perfil.
(2) Bootstrap em blocos com os mesmos sorteios de fit_baseline.py (semente 7, mesma ordem de sítios, forçantes e
    estações), resolvido pelo perfil (mínimo global na grade, refinado por parábola em log τ): forma da
    distribuição de τ e de α (assimetria, número de modas) e conferência dos percentis da Tabela 3.
(3) Janelas de 3 e 7 dias em vez de 5, com a mesma divisão cronológica 70/15/15.
(4) Protocolo: intervalo de tempo entre a última janela de uma partição e a primeira da seguinte; número de
    janelas e de pontos de teste; autocorrelação dos resíduos de teste; cobertura do intervalo de predição de 90 %
    com intervalo de confiança por bootstrap de dias inteiros do teste.
(5) τ mensal × umidade relativa: Spearman total, por série, com postos centrados em cada série (sítio e altura) e
    com anomalias em relação ao mesmo mês do ano em Jaru a 62 m (controle da sazonalidade).
Saídas: dados/perfil_tau.csv (perfis), dados/revisao_identificabilidade.csv (um registro por conjunto),
dados/janelas_sensibilidade.csv e dados/tau_umidade_controle.csv."""
import os, numpy as np, pandas as pd
from scipy.stats import spearmanr, skew, gaussian_kde
import fit_dry_model as F
import fit_baseline as L

DATA = F.DATA
GRID = np.exp(np.linspace(np.log(0.5), np.log(60.0), 400))
MAIN = [('jaru', 'chuvosa'), ('jaru', 'seca'), ('cuieiras', 'chuvosa'), ('cuieiras', 'seca'),
        ('sinop', 'seca'), ('mujui', 'terrestre'), ('mujui', 'inundada')]


def design(ws, tau):
    """Somas por janela para o ajuste linear de (α, Teq) com τ fixo."""
    T, R, M = F.pack(ws, 'atual')
    Tz = T.copy(); Tz[:, 0] = 0.0
    X0 = L.simulate_lin(0.0, tau, 0.0, T, R, M)
    X1 = L.simulate_lin(1.0, tau, 0.0, Tz, R, M)
    X2 = L.simulate_lin(0.0, tau, 1.0, Tz, R, M)
    m = M.copy(); m[:, 0] = False
    y = np.where(m, T - X0, 0.0); X1 = np.where(m, X1, 0.0); X2 = np.where(m, X2, 0.0)
    G = np.empty((len(ws), 2, 2))
    G[:, 0, 0] = (X1 * X1).sum(1); G[:, 0, 1] = G[:, 1, 0] = (X1 * X2).sum(1); G[:, 1, 1] = (X2 * X2).sum(1)
    b = np.c_[(X1 * y).sum(1), (X2 * y).sum(1)]
    return G, b, (y * y).sum(1), m.sum(1)


def tabelas(ws):
    out = [design(ws, t) for t in GRID]
    return (np.stack([o[0] for o in out]), np.stack([o[1] for o in out]), np.stack([o[2] for o in out]),
            out[0][3])


def perfil(tab, k=None):
    """SSE mínimo em (α, Teq) para cada τ da grade, com pesos k por janela (contagens do bootstrap)."""
    G, b, c, n = tab
    k = np.ones(G.shape[1]) if k is None else k
    Gs = np.einsum('w,twij->tij', k, G); bs = np.einsum('w,twi->ti', k, b); cs = np.einsum('w,tw->t', k, c)
    x = np.linalg.solve(Gs, bs[..., None])[..., 0]
    sse = cs - np.einsum('ti,ti->t', bs, x)
    return sse, x, (k * n).sum()


def minimo(sse, x):
    i = int(np.argmin(sse))
    i = min(max(i, 1), len(GRID) - 2)
    lt = np.log(GRID[i - 1:i + 2]); s = sse[i - 1:i + 2]
    den = s[0] - 2 * s[1] + s[2]
    d = 0.5 * (s[0] - s[2]) / den if den > 0 else 0.0
    lt0 = lt[1] + d * (lt[2] - lt[1])
    w = d if d >= 0 else -d
    j = i + (1 if d >= 0 else -1)
    xs = (1 - w) * x[i] + w * x[j]
    return float(np.exp(lt0)), xs


def n_modas(v):
    if np.std(v) == 0:
        return 1
    g = np.linspace(v.min(), v.max(), 400)
    d = gaussian_kde(v)(g)
    pk = (d[1:-1] > d[:-2]) & (d[1:-1] > d[2:]) & (d[1:-1] > 0.1 * d.max())
    return int(pk.sum())


def bootstrap_sorteios():
    """Repete a sequência de sorteios de fit_baseline.main()."""
    rng = np.random.default_rng(7)
    out = {}
    for site in L.SITES:
        df = pd.read_csv(os.path.join(DATA, f'{site}_30min.csv'), index_col=0, parse_dates=True)
        for rcol in ['Rn'] + (['Rsd'] if site == 'sinop' else []):
            for est in df['estacao'].dropna().unique():
                ws = L.windows_with(df[df['estacao'] == est], rcol)
                if sum(len(w[0]) for w in ws) < 10 * 48:
                    continue
                tr, va, te = F.split(ws)
                out[(site, est, rcol)] = [rng.integers(0, len(tr), len(tr)) for _ in range(L.B)]
    return out


def fim(w):
    return w[2] + pd.Timedelta(minutes=30) * (len(w[0]) - 1)


def principal():
    par = pd.read_csv(os.path.join(DATA, 'linha_base_parametros.csv'))
    par = par[par['forcante'] == 'Rn'].set_index(['sitio', 'estacao'])
    ic = pd.read_csv(os.path.join(DATA, 'intervalo_predicao_base.csv')).set_index(['sitio', 'estacao'])
    sorteios = bootstrap_sorteios()
    rng = np.random.default_rng(11)
    linhas, perfis = [], []
    for site, est in MAIN:
        df = pd.read_csv(os.path.join(DATA, f'{site}_30min.csv'), index_col=0, parse_dates=True)
        ws = L.windows_with(df[df['estacao'] == est], 'Rn')
        tr, va, te = F.split(ws)
        tab = tabelas(tr)
        sse, x, n = perfil(tab)
        tau, (a, Teq) = minimo(sse, x)
        rm = np.sqrt(sse / n); rmin = np.sqrt(sse.min() / n)
        for t, r, (aa, tq) in zip(GRID, rm, x):
            perfis.append(dict(sitio=site, estacao=est, tau_h=t, rmse_ajuste_K=r, a=aa, Teq_C=tq - 273.15))
        # curvatura do perfil: aumento relativo do RMSE com τ dividido e multiplicado por 2
        f = lambda tt: np.sqrt(np.interp(np.log(tt), np.log(GRID), sse) / n)
        faixa = GRID[rm <= 1.01 * rmin]
        # mínimos locais do perfil
        loc = int(((sse[1:-1] < sse[:-2]) & (sse[1:-1] < sse[2:])).sum())
        # bootstrap pelo perfil
        bt, ba = [], []
        for k in sorteios[(site, est, 'Rn')]:
            cnt = np.bincount(k, minlength=len(tr)).astype(float)
            s_b, x_b, _ = perfil(tab, cnt)
            t_b, (a_b, _) = minimo(s_b, x_b)
            bt.append(t_b); ba.append(a_b)
        bt, ba = np.array(bt), np.array(ba)
        p05, p50, p95 = np.percentile(bt, [5, 50, 95])
        # protocolo: intervalos entre partições, janelas e pontos de teste
        gap_tv = (va[0][2] - fim(tr[-1])).total_seconds() / 3600 if va else np.nan
        gap_vt = (te[0][2] - fim(va[-1])).total_seconds() / 3600 if va and te else np.nan
        p = [np.log(a), np.log(tau), Teq]
        Tt, Rt, Mt = F.pack(te, 'atual')
        e = L.resid(p, Tt, Rt, Mt)                       # sim - obs, sem o 1º ponto de cada janela
        S = L.simulate_lin(a, tau, Teq, Tt, Rt, Mt); m = Mt.copy(); m[:, 0] = False
        E = np.where(m, S - Tt, np.nan)
        ac1 = np.nanmean([np.corrcoef(r[np.isfinite(r)][:-1], r[np.isfinite(r)][1:])[0, 1] for r in E])
        ac48 = np.nanmean([np.corrcoef(r[np.isfinite(r)][:-48], r[np.isfinite(r)][48:])[0, 1] for r in E
                           if np.isfinite(r).sum() > 96])
        # cobertura do IP 90 % com os parâmetros da Tabela 3 e bootstrap de dias inteiros do teste
        q = ic.loc[(site, est)]
        pt = [np.log(par.loc[(site, est), 'a']), np.log(par.loc[(site, est), 'tau_h']), par.loc[(site, est), 'Teq_C'] + 273.15]
        St = L.simulate_lin(*np.r_[np.exp(pt[:2]), pt[2]], Tt, Rt, Mt)
        ob = np.where(m, Tt - St, np.nan)              # obs - sim
        dentro = (ob >= q['q05_K']) & (ob <= q['q95_K'])
        dias = []
        for r_, d_ in zip(ob, dentro):
            ok = np.isfinite(r_)
            for j in range(0, ok.sum(), 48):
                dias.append(d_[ok][j:j + 48])
        cov = np.mean(np.concatenate(dias))
        cb = [np.mean(np.concatenate([dias[i] for i in rng.integers(0, len(dias), len(dias))])) for _ in range(2000)]
        linhas.append(dict(sitio=site, estacao=est, janelas_ajuste=len(tr), janelas_validacao=len(va),
                           janelas_teste=len(te), pontos_teste=int(m.sum()),
                           inicio_validacao=va[0][2] if va else None, inicio_teste=te[0][2],
                           intervalo_ajuste_validacao_h=gap_tv, intervalo_validacao_teste_h=gap_vt,
                           tau_perfil_h=tau, a_perfil=a, Teq_perfil_C=Teq - 273.15,
                           tau_tabela3_h=par.loc[(site, est), 'tau_h'],
                           rmse_min_K=rmin, aumento_rmse_tau_metade=f(tau / 2) / rmin - 1,
                           aumento_rmse_tau_dobro=f(2 * tau) / rmin - 1,
                           tau_1pct_min_h=faixa.min(), tau_1pct_max_h=faixa.max(), minimos_locais_perfil=loc,
                           boot_tau_p05=p05, boot_tau_p50=p50, boot_tau_p95=p95,
                           tab3_tau_p05=par.loc[(site, est), 'tau_h_p05'], tab3_tau_p95=par.loc[(site, est), 'tau_h_p95'],
                           boot_tau_assimetria=skew(bt), boot_tau_razao_caudas=(p95 - p50) / (p50 - p05),
                           boot_tau_modas=n_modas(bt), boot_a_assimetria=skew(ba), boot_a_modas=n_modas(ba),
                           corr_log_a_tau=np.corrcoef(np.log(ba), np.log(bt))[0, 1],
                           autocorr_residuo_30min=ac1, autocorr_residuo_24h=ac48,
                           cobertura_IP90=cov, cobertura_p05=np.percentile(cb, 5), cobertura_p95=np.percentile(cb, 95),
                           dias_teste_completos=len(dias)))
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in linhas[-1].items()}, flush=True)
    pd.DataFrame(perfis).to_csv(os.path.join(DATA, 'perfil_tau.csv'), index=False, float_format='%.5g')
    pd.DataFrame(linhas).to_csv(os.path.join(DATA, 'revisao_identificabilidade.csv'), index=False, float_format='%.4g')


def janelas():
    linhas = []
    w0 = F.WIN
    try:
        for dias in [3, 5, 7]:
            F.WIN = dias * 48
            for site, est in MAIN:
                df = pd.read_csv(os.path.join(DATA, f'{site}_30min.csv'), index_col=0, parse_dates=True)
                tr, va, te = F.split(L.windows_with(df[df['estacao'] == est], 'Rn'))
                sse, x, n = perfil(tabelas(tr))
                tau, (a, Teq) = minimo(sse, x)
                e = L.resid([np.log(a), np.log(tau), Teq], *F.pack(te, 'atual'))
                linhas.append(dict(dias_janela=dias, sitio=site, estacao=est, tau_h=tau, a=a, Teq_C=Teq - 273.15,
                                   rmse_ajuste_K=np.sqrt(sse.min() / n), rmse_teste_K=np.sqrt(np.mean(e ** 2)),
                                   janelas_ajuste=len(tr), janelas_teste=len(te)))
                print(linhas[-1], flush=True)
    finally:
        F.WIN = w0
    d = pd.DataFrame(linhas)
    ref = d[d['dias_janela'] == 5].set_index(['sitio', 'estacao'])['tau_h']
    d['tau_rel_5d'] = [r['tau_h'] / ref[(r['sitio'], r['estacao'])] - 1 for _, r in d.iterrows()]
    d.to_csv(os.path.join(DATA, 'janelas_sensibilidade.csv'), index=False, float_format='%.4g')


def umidade():
    m = pd.read_csv(os.path.join(DATA, 'meses_parametros.csv'), dtype={'ano_estacao': str})
    m['mes_num'] = pd.PeriodIndex(m['mes'], freq='M').month
    m['ano'] = pd.PeriodIndex(m['mes'], freq='M').year
    susp = ((m['sitio'] == 'jaru') & (((m['altura'] == '62m') & (m['ano'] == 2009)) |
                                      ((m['altura'] == '22m') & (m['ano'] == 2008))))
    m = m[~susp].copy()
    m['serie'] = m['sitio'] + ' ' + m['altura'].astype(str)
    out = []
    r, p = spearmanr(m['UR_media'], m['tau_h'])
    out.append(dict(analise='todos os meses', n=len(m), rho=r, p=p))
    for s, g in m.groupby('serie'):
        if len(g) >= 5:
            r, p = spearmanr(g['UR_media'], g['tau_h'])
            out.append(dict(analise=f'série {s}', n=len(g), rho=r, p=p))
    # postos centrados em cada série: retira a diferença média entre sítios e alturas
    m['rk_ur'] = m.groupby('serie')['UR_media'].rank(pct=True) - 0.5
    m['rk_tau'] = m.groupby('serie')['tau_h'].rank(pct=True) - 0.5
    r = np.corrcoef(m['rk_ur'], m['rk_tau'])[0, 1]
    # permutação dentro de cada série
    rng = np.random.default_rng(3)
    null = []
    for _ in range(5000):
        pr = m.groupby('serie')['rk_tau'].transform(lambda v: rng.permutation(v.values))
        null.append(np.corrcoef(m['rk_ur'], pr)[0, 1])
    out.append(dict(analise='postos centrados por série (permutação dentro da série)', n=len(m), rho=r,
                    p=(np.sum(np.abs(null) >= abs(r)) + 1) / (len(null) + 1)))
    # sazonalidade: anomalias em relação ao mesmo mês do ano, Jaru a 62 m
    j = m[m['serie'] == 'jaru 62m'].copy()
    cnt = j.groupby('mes_num')['tau_h'].transform('count')
    j = j[cnt >= 2].copy()
    j['an_tau'] = j['tau_h'] - j.groupby('mes_num')['tau_h'].transform('mean')
    j['an_ur'] = j['UR_media'] - j.groupby('mes_num')['UR_media'].transform('mean')
    r, p = spearmanr(j['an_ur'], j['an_tau'])
    out.append(dict(analise='Jaru 62 m, anomalias em relação ao mesmo mês do ano', n=len(j), rho=r, p=p))
    d = pd.DataFrame(out)
    print(d.to_string())
    d.to_csv(os.path.join(DATA, 'tau_umidade_controle.csv'), index=False, float_format='%.4g')


if __name__ == '__main__':
    import sys
    etapas = sys.argv[1:] or ['principal', 'janelas', 'umidade']
    for e in etapas:
        globals()[e]()
