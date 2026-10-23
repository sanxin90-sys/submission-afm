"""Revisão para a Agricultural and Forest Meteorology (22/10/2026): τ × partição do balanço de energia no Cuieiras.
O Cuieiras (K34) é o único sítio com fluxos turbulentos (H e LE, covariância de vórtices) no projeto. Testa se o τ da
linha de base acompanha a partição de Rn entre H e LE, como supõe a interpretação τ = C/λ (Seção 3.4 do manuscrito).

(1) τ por janela: cada janela de 5 dias (todas as janelas válidas de T e Rn, sem divisão ajuste/teste) é ajustada
    sozinha pelo perfil exato de revisao_identificabilidade.py (α e Teq por mínimos quadrados para cada τ da grade).
    Janelas com o mínimo na borda da grade ou α ≤ 0 saem. Para cada janela: fração evaporativa diurna
    EF = ΣLE/Σ(H+LE), razão de Bowen ΣH/ΣLE, H/Rn e LE/Rn (somas nos instantes com Rn > 10 W/m² e H, LE e Rn
    válidos), fechamento (H+LE)/Rn, UR, Rn e T médios. Exige fluxos válidos em pelo menos 70 % dos instantes diurnos.
    Spearman de τ com cada variável (as janelas são consecutivas e não independentes).
(2) τ agrupado por tercis de EF (e, para comparação, de UR): α, Teq e τ ajustados juntos nas janelas de cada tercil
    pelo perfil, com IC 90 % por bootstrap das janelas do tercil (B = 500, semente 11).
(3) Partição por estação (calendário de estacoes_anos.py) nos mesmos dias, lado a lado com o τ sazonal de
    estacoes_anos_parametros.csv, e por mês, com o τ de meses_parametros.csv.
Saídas: dados/fluxos_cuieiras_janelas.csv, dados/fluxos_cuieiras_resumo.csv."""
import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
import fit_dry_model as F
import revisao_identificabilidade as RI
from estacoes_anos import qc_rn, CAL

DATA = F.DATA
B = 500
rng = np.random.default_rng(11)
RN_DIA = 10.0
COB_MIN = 0.7


def particao(d):
    """Somas diurnas da partição de Rn em d (DataFrame a 30 min)."""
    m = (d['Rn'] > RN_DIA) & d['H'].notna() & d['LE'].notna()
    n_dia = (d['Rn'] > RN_DIA).sum()
    H, LE, Rn = d.loc[m, 'H'].sum(), d.loc[m, 'LE'].sum(), d.loc[m, 'Rn'].sum()
    return dict(cob_fluxo=m.sum() / n_dia if n_dia else np.nan, EF=LE / (H + LE), bowen=H / LE,
                H_Rn=H / Rn, LE_Rn=LE / Rn, fechamento=(H + LE) / Rn)


def tau_janela(tab, w):
    G, b, c, n = tab
    sse, x, _ = RI.perfil((G[:, w:w + 1], b[:, w:w + 1], c[:, w:w + 1], n[w:w + 1]))
    i = int(np.argmin(sse))
    if i in (0, len(RI.GRID) - 1):
        return np.nan, np.nan
    tau, xs = RI.minimo(sse, x)
    return tau, xs[0]


def tau_grupo(tab, idx):
    G, b, c, n = tab
    sub = (G[:, idx], b[:, idx], c[:, idx], n[idx])
    tau = RI.minimo(*RI.perfil(sub)[:2])[0]
    bs = []
    for _ in range(B):
        k = np.bincount(rng.integers(0, len(idx), len(idx)), minlength=len(idx)).astype(float)
        sse, x, _ = RI.perfil(sub, k)
        bs.append(RI.minimo(sse, x)[0])
    return tau, *np.percentile(bs, [5, 95])


def main():
    d = qc_rn(pd.read_csv(os.path.join(DATA, 'cuieiras_30min.csv'), index_col=0, parse_dates=True))
    t0 = d.index - pd.Timedelta('1ns')
    d['estacao4'] = pd.Series(t0.month, index=d.index).map(CAL['cuieiras'])
    d['ano'] = t0.year
    ws = F.windows(d)
    tab = RI.tabelas(ws)

    rows = []
    for w, (T, R, ini) in enumerate(ws):
        fim = ini + pd.Timedelta(minutes=30 * (len(T) - 1))
        dw = d.loc[ini:fim]
        tau, a = tau_janela(tab, w)
        row = dict(inicio=ini, dias=len(T) / 48, estacao=dw['estacao4'].mode().iat[0],
                   tau_h=tau, a=a, T_media_C=dw['T_C'].mean(), Rn_media=dw['Rn'].mean(), UR_media=dw['RH'].mean())
        row.update(particao(dw))
        rows.append(row)
    J = pd.DataFrame(rows)
    J['valida'] = J['tau_h'].notna() & (J['a'] > 0) & (J['cob_fluxo'] >= COB_MIN)
    J.to_csv(os.path.join(DATA, 'fluxos_cuieiras_janelas.csv'), index=False, float_format='%.5g')
    V = J[J['valida']].reset_index(drop=True)
    idx_val = np.flatnonzero(J['valida'].values)
    print(f'janelas: {len(J)}; válidas: {len(V)}')

    res = []
    for v in ['EF', 'bowen', 'H_Rn', 'LE_Rn', 'fechamento', 'UR_media', 'Rn_media', 'T_media_C']:
        r, p = spearmanr(V['tau_h'], V[v])
        res.append(dict(analise='janelas_spearman', variavel=v, n=len(V), rho=r, p=p))
    res.append(dict(analise='janelas_tau', variavel='tau_h', n=len(V), mediana=V['tau_h'].median(),
                    p25=V['tau_h'].quantile(.25), p75=V['tau_h'].quantile(.75)))

    for v in ['EF', 'UR_media']:
        q = pd.qcut(V[v], 3, labels=['baixo', 'medio', 'alto'])
        for g in ['baixo', 'medio', 'alto']:
            sel = idx_val[(q == g).values]
            tau, lo, hi = tau_grupo(tab, sel)
            res.append(dict(analise=f'tercil_{v}', variavel=g, n=len(sel), tau_h=tau, tau_p05=lo, tau_p95=hi,
                            EF=V.loc[(q == g).values, 'EF'].mean(), UR=V.loc[(q == g).values, 'UR_media'].mean(),
                            v_min=V.loc[(q == g).values, v].min(), v_max=V.loc[(q == g).values, v].max()))

    E = pd.read_csv(os.path.join(DATA, 'estacoes_anos_parametros.csv'))
    E = E[(E['sitio'] == 'cuieiras') & E['tau_h'].notna()]
    for _, e in E.iterrows():
        dd = d[(d.index > e['inicio']) & (d.index <= e['fim']) & d['ok_TR']]
        row = dict(analise='estacao', variavel=f"{e['estacao']}_{e['ano']}", tau_h=e['tau_h'],
                   tau_p05=e['tau_h_p05'], tau_p95=e['tau_h_p95'], UR=e['UR_media'])
        row.update(particao(dd))
        res.append(row)

    Mo = pd.read_csv(os.path.join(DATA, 'meses_parametros.csv'))
    Mo = Mo[Mo['sitio'] == 'cuieiras']
    mm = []
    for _, e in Mo.iterrows():
        per = pd.Period(e['mes'], 'M')
        dd = d[(t0.to_period('M') == per) & d['ok_TR']]
        row = dict(analise='mes', variavel=e['mes'], tau_h=e['tau_h'], UR=e['UR_media'])
        row.update(particao(dd))
        mm.append(row)
    res += mm
    mm = pd.DataFrame(mm)
    for v in ['EF', 'bowen', 'H_Rn', 'LE_Rn', 'UR']:
        r, p = spearmanr(mm['tau_h'], mm[v])
        res.append(dict(analise='meses_spearman', variavel=v, n=len(mm), rho=r, p=p))

    R = pd.DataFrame(res)
    R.to_csv(os.path.join(DATA, 'fluxos_cuieiras_resumo.csv'), index=False, float_format='%.6g')
    with pd.option_context('display.width', 200, 'display.max_columns', 30):
        print(R)


if __name__ == '__main__':
    main()
