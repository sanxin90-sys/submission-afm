"""Análise sazonal por ano, com as estações de transição.
Calendários (meses de cada estação):
- Jaru, como no Artigo 1 (Webler et al., 2007): chuvosa jan-mar; transição chuvosa-seca abr-jun;
  seca jul-set; transição seca-chuvosa out-dez.
- Sinop, como em De Paulo et al. (2015): chuvosa dez-fev; transição chuvosa-seca mar-mai;
  seca jun-ago; transição seca-chuvosa set-nov.
- Cuieiras (Amazônia central, conferido com a chuva da torre): chuvosa jan-abr; transição mai-jun;
  seca jul-out; transição nov-dez.
- Mujuí, fases do pulso de inundação pelo nível do rio: terrestre (23/11/2009 a 28/02/2010),
  cheia (lâmina no sítio > 0, 01/05 a 21/07/2010) e pós-cheia (22/07 a 31/08/2010).
Para cada sítio, altura, ano e estação com pelo menos MIN_DIAS dias em janelas: base linear com o
protocolo do Artigo 2 (janelas de 5 dias, divisão cronológica 70/15/15, bootstrap em blocos).
Cada mês com pelo menos 15 dias também é ajustado com todas as suas janelas (sem divisão), partindo
dos parâmetros da estação, para desenhar o ciclo anual.
QC extra de Rn: dias com Rn máximo abaixo de 50 W/m2 (sensor parado) saem.
Saídas: dados/estacoes_anos_parametros.csv e dados/meses_parametros.csv."""
import os, numpy as np, pandas as pd
from multiprocessing import Pool
import fit_dry_model as F
import fit_baseline as L

DATA = F.DATA
B = int(os.environ.get('B', 100))
MIN_DIAS = 20
NPROC = int(os.environ.get('NPROC', 2))
CAL = {
    'jaru': {1: 'chuvosa', 2: 'chuvosa', 3: 'chuvosa', 4: 'transicao_cs', 5: 'transicao_cs', 6: 'transicao_cs',
             7: 'seca', 8: 'seca', 9: 'seca', 10: 'transicao_sc', 11: 'transicao_sc', 12: 'transicao_sc'},
    'sinop': {12: 'chuvosa', 1: 'chuvosa', 2: 'chuvosa', 3: 'transicao_cs', 4: 'transicao_cs', 5: 'transicao_cs',
              6: 'seca', 7: 'seca', 8: 'seca', 9: 'transicao_sc', 10: 'transicao_sc', 11: 'transicao_sc'},
    'cuieiras': {1: 'chuvosa', 2: 'chuvosa', 3: 'chuvosa', 4: 'chuvosa', 5: 'transicao_cs', 6: 'transicao_cs',
                 7: 'seca', 8: 'seca', 9: 'seca', 10: 'seca', 11: 'transicao_sc', 12: 'transicao_sc'},
}
FLAGS = {('jaru', '62m', 2009): 'T62 de 2009 de 2 a 4 K abaixo dos outros anos (sensor suspeito)',
         ('jaru', '22m', 2008): 'T22 de 2008 com amplitude maior que a de 2005-2006 (sensor ou posição diferente)'}


def qc_rn(d):
    d = d.copy()
    dmax = d['Rn'].groupby((d.index - pd.Timedelta('1ns')).normalize()).transform('max')
    d.loc[dmax < 50, 'Rn'] = np.nan
    d['ok_TR'] = d['T_C'].notna() & d['Rn'].notna()
    return d


def series():
    """Lista de (sítio, altura, DataFrame com colunas estacao4 e ano_estacao)."""
    out = []
    j = pd.read_csv(os.path.join(DATA, 'jaru_anos_30min.csv'), index_col=0, parse_dates=True)
    for (ano, alt), g in j.groupby(['ano', 'altura']):
        g = qc_rn(g)
        g['estacao4'] = (g.index - pd.Timedelta('1ns')).month.map(CAL['jaru'])
        g['ano_estacao'] = str(ano)
        out.append(('jaru', alt, g))
    for site, alt in [('sinop', '40m'), ('cuieiras', '51m')]:
        g = qc_rn(pd.read_csv(os.path.join(DATA, f'{site}_30min.csv'), index_col=0, parse_dates=True))
        mes = (g.index - pd.Timedelta('1ns')).month
        ano = (g.index - pd.Timedelta('1ns')).year
        g['estacao4'] = pd.Series(mes, index=g.index).map(CAL[site])
        # chuvosa de Sinop (dez-fev) leva o ano de janeiro
        g['ano_estacao'] = np.where((site == 'sinop') & (mes == 12), ano + 1, ano).astype(str)
        out.append((site, alt, g))
    g = qc_rn(pd.read_csv(os.path.join(DATA, 'mujui_30min.csv'), index_col=0, parse_dates=True))
    g['estacao4'] = g['fase_hidro'].replace('', np.nan)
    g['ano_estacao'] = np.where(g['estacao4'] == 'terrestre', '2009/2010', '2010')
    out.append(('mujui', '12m', g))
    return out


def fit_season(job):
    site, alt, ano, est, sub, seed = job
    rng = np.random.default_rng(seed)
    ws = F.windows(sub)
    dias = sum(len(w[0]) for w in ws) / 48
    row = dict(sitio=site, altura=alt, ano=ano, estacao=est, inicio=str(sub.index.min().date()),
               fim=str(sub.index.max().date()), dias_validos=round(sub['ok_TR'].sum() / 48, 1),
               dias_em_janelas=round(dias, 1), obs=FLAGS.get((site, alt, int(ano) if ano.isdigit() else 0), ''),
               T_media_C=sub['T_C'].mean(), Rn_media=sub['Rn'].mean(),
               UR_media=sub['RH'].mean() if 'RH' in sub else np.nan)
    if dias < MIN_DIAS:
        return row, None
    tr, va, te = F.split(ws)
    Ttr, Rtr, Mtr = F.pack(tr, 'atual'); Tte, Rte, Mte = F.pack(te, 'atual')
    p = L.fit(Ttr, Rtr, Mtr)
    ete = L.resid(p, Tte, Rte, Mte)
    bs = []
    for _ in range(B):
        k = rng.integers(0, len(tr), len(tr))
        bs.append(L.fit(*F.pack([tr[i] for i in k], 'atual'), p0=p))
    bs = np.array(bs)
    par = np.c_[np.exp(bs[:, 0]), np.exp(bs[:, 1]), bs[:, 2] - 273.15]
    row.update(a=np.exp(p[0]), tau_h=np.exp(p[1]), Teq_C=p[2] - 273.15,
               rmse_ajuste_K=np.sqrt(np.mean(L.resid(p, Ttr, Rtr, Mtr) ** 2)),
               rmse_teste_K=np.sqrt(np.mean(ete ** 2)), vies_teste_K=np.mean(ete), B=B)
    for j, nm in enumerate(['a', 'tau_h', 'Teq_C']):
        row[nm + '_p05'], row[nm + '_p95'] = np.percentile(par[:, j], [5, 95])
    # meses da estação, ajustados com todas as janelas do mês a partir de p
    months = []
    for per, s in sub.groupby(sub.index.to_period('M')):
        wm = F.windows(s)
        dm = sum(len(w[0]) for w in wm) / 48
        if dm < 15:
            continue
        T, R, M = F.pack(wm, 'atual')
        pm = L.fit(T, R, M, p0=p)
        months.append(dict(sitio=site, altura=alt, ano_estacao=ano, estacao=est, mes=str(per),
                           dias_em_janelas=round(dm, 1), a=np.exp(pm[0]), tau_h=np.exp(pm[1]),
                           Teq_C=pm[2] - 273.15, rmse_K=np.sqrt(np.mean(L.resid(pm, T, R, M) ** 2)),
                           T_media_C=s['T_C'].mean(), Rn_media=s['Rn'].mean(),
                           UR_media=s['RH'].mean() if 'RH' in s else np.nan))
    print(site, alt, ano, est, f"{dias:.0f} d, tau={row['tau_h']:.2f} h, Teq={row['Teq_C']:.1f} C, "
          f"RMSE={row['rmse_teste_K']:.2f} K", flush=True)
    return row, months


def main():
    jobs, seed = [], 100
    for site, alt, g in series():
        g = g[g['estacao4'].notna()]
        for (ano, est), sub in g.groupby(['ano_estacao', 'estacao4'], sort=False):
            sub = sub.sort_index()
            sub = sub.reindex(pd.date_range(sub.index.min(), sub.index.max(), freq='30min'))
            sub['ok_TR'] = sub['ok_TR'].fillna(False).astype(bool)
            jobs.append((site, alt, ano, est, sub, seed)); seed += 1
    with Pool(NPROC) as pool:
        res = pool.map(fit_season, jobs, chunksize=1)
    rows = [r for r, _ in res if r['dias_validos'] > 0]
    months = [m for _, ms in res if ms for m in ms]
    pd.DataFrame(rows).to_csv(os.path.join(DATA, 'estacoes_anos_parametros.csv'), index=False, float_format='%.5g')
    pd.DataFrame(months).to_csv(os.path.join(DATA, 'meses_parametros.csv'), index=False, float_format='%.5g')


if __name__ == '__main__':
    main()
