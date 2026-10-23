"""Jaru, comparação entre anos (2004 a 2010) com a linha de base linear e o protocolo fixo.
A aba DADOS OK tem um bloco de colunas por ano; o tempo (Dia juliano e Hora HHMM) vem do
bloco de 2004. Nos anos não bissextos, a linha do dia 366 fica vazia.
Altura da temperatura: 62 m em 2004, 2008, 2009 e 2010; 22 m em 2005, 2006 e 2008.
2007 fica fora: o pico médio de T22 cai entre 4 h e 24 h conforme o mês (sensor com defeito).
Estações: chuvosa = jan a mar; seca = jul a set (as mesmas de Jaru 2008).
Saídas: dados/jaru_anos_30min.csv e dados/jaru_anos_parametros.csv."""
import os, numpy as np, pandas as pd, openpyxl
import fit_dry_model as F
import fit_baseline as L
from preprocess import num, flatline, T_RANGE, RN_RANGE, RH_RANGE, FLAT_H, SRC

DATA = F.DATA
B = 100
rng = np.random.default_rng(11)
# ano: (coluna Ano, coluna Rn, {altura: coluna T}, coluna UR)
BLOCOS = {2004: (0, 5, {'62m': 6}, 7), 2005: (11, 18, {'22m': 20}, 19), 2006: (23, 26, {'22m': 28}, 27),
          2008: (44, 47, {'62m': 50, '22m': 49}, 48), 2009: (55, 56, {'62m': 58}, 57),
          2010: (63, 64, {'62m': 66}, 65)}
SEASONS = {'chuvosa': (1, 3), 'seca': (7, 9)}


def interp_run(s, n):
    """Trechos de >= n pontos com primeira diferença constante e não nula (interpolação linear)."""
    d1 = s.diff(); lin = (d1.diff().abs() < 1e-6) & (d1.abs() > 1e-6)
    run = lin.groupby((~lin).cumsum()).transform('sum') * lin
    return run >= n


def longest_run(ok):
    ok = ok.astype(int).values
    best = cur = 0
    for v in ok:
        cur = cur + 1 if v else 0; best = max(best, cur)
    return best / 48


def load():
    wb = openpyxl.load_workbook(os.path.join(SRC, 'Dados Rebio Jaru 2006 a 2008.xlsx'), read_only=True)
    A = pd.DataFrame(list(wb['DADOS OK'].iter_rows(min_row=2, values_only=True)))
    dia, hora = np.asarray(num(A[3]), float), np.asarray(num(A[4]), float)
    out = []
    for ano, (ca, cr, cts, cu) in BLOCOS.items():
        t = (pd.Timestamp(f'{ano}-01-01') + pd.to_timedelta(dia - 1, unit='D')
             + pd.to_timedelta((hora // 100) * 60 + hora % 100, unit='min'))
        for alt, ct in cts.items():
            d = pd.DataFrame({'T_C': np.asarray(num(A[ct]), float), 'Rn': np.asarray(num(A[cr]), float),
                              'RH': np.asarray(num(A[cu]), float), 'ano_bloco': np.asarray(num(A[ca]), float)}, index=t)
            d = d[d['ano_bloco'].notna() & d.index.notna() & (d.index.year == ano)]
            d = d[~d.index.duplicated()]
            d = d.reindex(pd.date_range(f'{ano}-01-01 00:30', f'{ano + 1}-01-01 00:00', freq='30min'))
            badT = ~d['T_C'].between(*T_RANGE) | flatline(d['T_C'], FLAT_H * 2) | interp_run(d['T_C'], FLAT_H * 2)
            badR = ~d['Rn'].between(*RN_RANGE) | interp_run(d['Rn'], FLAT_H * 2)
            d.loc[badT, 'T_C'] = np.nan; d.loc[badR, 'Rn'] = np.nan
            d.loc[~d['RH'].between(*RH_RANGE), 'RH'] = np.nan
            d['T_K'] = d['T_C'] + 273.15
            d['ok_TR'] = d['T_C'].notna() & d['Rn'].notna()
            d['ano'], d['altura'] = ano, alt
            d['estacao'] = None
            for est, (m0, m1) in SEASONS.items():
                d.loc[d.index.month.isin(range(m0, m1 + 1)), 'estacao'] = est
            out.append(d.drop(columns='ano_bloco'))
    return out


def main():
    series = load()
    pd.concat(series).to_csv(os.path.join(DATA, 'jaru_anos_30min.csv'), index_label='fim_intervalo',
                             float_format='%.4g')
    rows = []
    for d in series:
        ano, alt = d['ano'].iloc[0], d['altura'].iloc[0]
        for est in SEASONS:
            sub = d[d['estacao'] == est]
            ws = F.windows(sub)
            dias = sum(len(w[0]) for w in ws) / 48
            row = dict(ano=ano, altura=alt, estacao=est, dias_validos=round(sub['ok_TR'].sum() / 48, 1),
                       dias_em_janelas=round(dias, 1), maior_trecho_dias=round(longest_run(sub['ok_TR']), 1))
            if dias < 30:
                print(ano, alt, est, f'só {dias:.1f} dias em janelas; fica fora', flush=True)
                rows.append(row); continue
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
                       rmse_teste_K=np.sqrt(np.mean(ete ** 2)), vies_teste_K=np.mean(ete),
                       T_media_C=sub['T_C'].mean(), Rn_media=sub['Rn'].mean(), UR_media=sub['RH'].mean(),
                       amplitude_diaria_K=sub['T_C'].resample('D').agg(lambda x: x.max() - x.min()).mean())
            for j, nm in enumerate(['a', 'tau_h', 'Teq_C']):
                row[nm + '_p05'], row[nm + '_p95'] = np.percentile(par[:, j], [5, 95])
            rows.append(row)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(DATA, 'jaru_anos_parametros.csv'), index=False, float_format='%.5g')


if __name__ == '__main__':
    main()
