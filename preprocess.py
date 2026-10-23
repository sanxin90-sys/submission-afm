"""Pré-processamento único do Artigo 2 (Mujuí e Cuieiras).

Gera séries a 30 min (rótulo no FIM do intervalo, como em Jaru) com
sinalizadores de qualidade, rótulo de estação/fase e segmentos contínuos.
Sem preenchimento de lacunas. Uso: python3 preprocess.py [pasta_saida]
"""
import os, sys, glob, warnings
import numpy as np, pandas as pd, xlrd
warnings.filterwarnings('ignore')

SRC = os.environ.get('RAW_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'raw'))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
MIN_SEG_H = 24          # segmento mínimo útil (horas)
FLAT_H = 2              # T idêntica por >= 2 h => sensor travado
T_RANGE = (15.0, 40.0)  # °C
RH_RANGE = (5.0, 100.5) # %
RN_RANGE = (-200.0, 1200.0)  # W/m2
P_MUJUI = 1010.0        # hPa, constante (sem barômetro em Mujuí; ~60 m de altitude)

SEASONS = {
    # (sítio): lista de (rótulo, início, fim) — transições ficam sem rótulo
    # inundada = lâmina d'água no sítio > 0 (nível do rio > 33,43 m), de 01/05 a 21/07/2010;
    # de 22/07 em diante o sítio está seco (fase_hidro = 'pos_cheia', fora dos ajustes)
    'mujui': [('terrestre', '2009-11-23', '2010-02-28 23:59'),
              ('inundada',  '2010-05-01', '2010-07-21 23:59')],
    # Sinop (floresta de transição): seca jun-ago e chuvosa dez-fev, como em De Paulo et al. (2015);
    # abr-mai/2001 e set-nov/2003 são transições e ficam sem rótulo aqui (ver estacoes_anos.py)
    'sinop': [('seca',    '2002-06-01', '2002-06-30 23:59'),
              ('seca',    '2003-08-01', '2003-08-31 23:59'),
              ('chuvosa', '2003-12-01', '2003-12-31 23:59')],
    'jaru': [('chuvosa', '2008-01-01', '2008-03-31 23:59'),
             ('seca',    '2008-07-01', '2008-09-30 23:59')],
    'cuieiras': [('seca',     '2003-08-15', '2003-10-31 23:59'),
                 ('chuvosa',  '2004-01-01', '2004-04-30 23:59'),
                 ('seca',     '2004-07-01', '2004-10-31 23:59')],
}
EVENTS = {'mujui': [('friagem', '2010-07-17', '2010-07-18 23:59')]}
# Nível do rio em Mamirauá (diário, mai a jul/2010), enviado pelo usuário; cota do terreno do sítio = 33,43 m
NIVEL_MUJUI = os.path.join(SRC, 'NIVEL_AGUA_MEDIO.xlsx')
COTA_SITIO = 33.43


def num(s):
    return pd.to_numeric(pd.Series(s).replace(['', ' '], np.nan), errors='coerce').values


def q_specific(T_c, RH, p_hpa):
    """Umidade específica (kg/kg), Tetens/Bolton."""
    es = 6.112 * np.exp(17.67 * T_c / (T_c + 243.5))
    e = RH / 100.0 * es
    return 0.622 * e / (p_hpa - 0.378 * e)


def flatline(s, n):
    same = s.diff().eq(0)
    run = same.groupby((~same).cumsum()).cumsum()
    # marca o run inteiro quando ele atinge n repetições
    grp = (~same).cumsum()
    mx = run.groupby(grp).transform('max')
    return mx >= n


# ---------------------------------------------------------------- Mujuí
def load_mujui():
    cols = ['Data', 'Hora', 'Ttr1_11', 'Ttr1_5', 'Ttr1_2', 'Ttr75_6', 'Ttr75_2',
            'Ttr125_5', 'Ttr125_15', 'Tar5', 'Tar12', 'UR5', 'UR12', 'Rn20']
    frames = []
    for f in sorted(glob.glob(os.path.join(SRC, '0*_10 min.xls'))):
        sh = xlrd.open_workbook(f).sheet_by_index(0)
        rows = [sh.row_values(r)[:14] for r in range(3, sh.nrows)]
        df = pd.DataFrame(rows, columns=cols)
        frames.append(df.apply(num))
    d = pd.concat(frames).dropna(subset=['Data'])
    t = pd.to_datetime('1899-12-30') + pd.to_timedelta(d['Data'] + d['Hora'] % 1, unit='D')
    d.index = t.dt.round('min')
    d = d[~d.index.duplicated()].sort_index().drop(columns=['Data', 'Hora'])
    d = d.reindex(pd.date_range(d.index.min().floor('D'), d.index.max(), freq='10min'))
    # QC a 10 min
    for c in ['Tar5', 'Tar12']:
        bad = ~d[c].between(*T_RANGE) | flatline(d[c], FLAT_H * 6)
        d.loc[bad, c] = np.nan
    for c in ['UR5', 'UR12']:
        d.loc[~d[c].between(*RH_RANGE), c] = np.nan
    d.loc[~d['Rn20'].between(*RN_RANGE), 'Rn20'] = np.nan
    d[['UR5', 'UR12']] = d[['UR5', 'UR12']].clip(upper=100)
    # 10 -> 30 min: média em bloco, exige 3/3 válidos, rótulo no fim
    r = d.resample('30min', closed='right', label='right')
    m, n = r.mean(), r.count()
    m = m.where(n == 3)
    out = pd.DataFrame({
        'T_C': m['Tar12'], 'T5_C': m['Tar5'], 'Rn': m['Rn20'],
        'RH': m['UR12'], 'RH5': m['UR5'], 'P_hPa': P_MUJUI,
        'Ttr_75_2': m['Ttr75_2'], 'Ttr_75_6': m['Ttr75_6'],
    })
    out['q'] = q_specific(out['T_C'], out['RH'], out['P_hPa'])
    # nível diário do rio; o intervalo que termina às 00:00 pertence ao dia anterior
    import openpyxl
    with open(NIVEL_MUJUI, 'rb') as fh:
        ws = openpyxl.load_workbook(fh, data_only=True).active
        nv = pd.Series({pd.Timestamp(r[0]): r[1] for r in ws.iter_rows(min_row=2, values_only=True)
                        if r[0] is not None and r[1] is not None}, dtype=float)
    dia = (out.index - pd.Timedelta('1ns')).normalize()
    out['nivel_rio_m'] = nv.reindex(dia).values
    out['lamina_sitio_m'] = (out['nivel_rio_m'] - COTA_SITIO).clip(lower=0)
    out['fase_hidro'] = ''
    out.loc['2009-11-23':'2010-02-28 23:59', 'fase_hidro'] = 'terrestre'
    out.loc[out['lamina_sitio_m'] > 0, 'fase_hidro'] = 'cheia'
    out.loc['2010-07-22':'2010-08-31 23:59', 'fase_hidro'] = 'pos_cheia'
    return out


# ------------------------------------------------------------- Cuieiras
def load_sheets(fname, cols):
    wb = xlrd.open_workbook(os.path.join(SRC, fname), on_demand=True)
    parts = []
    for y in ['2003', '2004']:
        sh = wb.sheet_by_name(y)
        rows = [sh.row_values(r) for r in range(3, sh.nrows)]
        df = pd.DataFrame(rows)
        df = df[[0, 2] + list(cols)].rename(columns={0: 'Data', 2: 'Hora', **cols}).apply(num)
        parts.append(df)
    df = pd.concat(parts).dropna(subset=['Data', 'Hora'])
    h = df['Hora']
    t = (pd.to_datetime('1899-12-30') + pd.to_timedelta(df['Data'], unit='D')
         + pd.to_timedelta((h // 100) * 60 + h % 100, unit='min'))
    df.index = t
    return df[~df.index.duplicated()].drop(columns=['Data', 'Hora'])


def load_cuieiras():
    m = load_sheets('1. Variáveis Meteorológicas Cuieiras.xls',
                    {8: 'T51', 7: 'T42', 9: 'RH_aws', 10: 'RH_spcd', 12: 'PP', 17: 'P_hPa'})
    r = load_sheets('3. Irradiância Cuieiras.xls', {18: 'Rn_aws', 19: 'Rn_spcd'})
    f = load_sheets('4. Densidades de Fluxo de Energia Cuieiras.xls', {3: 'H', 4: 'LE', 5: 'G'})
    d = m.join(r, how='outer').join(f, how='outer')
    d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq='30min'))
    for c in ['T51', 'T42']:
        d.loc[~d[c].between(*T_RANGE) | flatline(d[c], FLAT_H * 2), c] = np.nan
    for c in ['RH_aws', 'RH_spcd']:
        d.loc[~d[c].between(*RH_RANGE), c] = np.nan
    for c in ['Rn_aws', 'Rn_spcd']:
        d.loc[~d[c].between(*RN_RANGE), c] = np.nan
    d.loc[~d['P_hPa'].between(950, 1050), 'P_hPa'] = np.nan
    out = pd.DataFrame({
        'T_C': d['T51'], 'T42_C': d['T42'],
        # Rn: sensor principal = AWS (DRN30, 44,6 m); SPCD31 (NR-LITE, 48 m) só em coluna à parte
        'Rn': d['Rn_aws'], 'Rn_spcd': d['Rn_spcd'],
        'RH': d['RH_aws'].fillna(d['RH_spcd']).clip(upper=100),
        'P_hPa': d['P_hPa'].fillna(d['P_hPa'].median()),
        'PP_mm': d['PP'], 'H': d['H'], 'LE': d['LE'], 'G': d['G'],
    })
    out['q'] = q_specific(out['T_C'], out['RH'], out['P_hPa'])
    return out


# ---------------------------------------------------------------- Jaru
def load_jaru():
    """Jaru 2008 (Artigo 1). O bloco de 2008 não tem colunas de tempo próprias:
    usa Dia juliano/Hora do bloco de 2004 (ambos bissextos, 17568 linhas).
    A coluna 'Dia' é o dia juliano. Uma linha (28/07/2008 12:30) traz Ano = 2010 por erro de digitação."""
    import openpyxl
    wb = openpyxl.load_workbook(os.path.join(SRC, 'Dados Rebio Jaru 2006 a 2008.xlsx'), read_only=True)
    rows = list(wb['DADOS OK'].iter_rows(min_row=2, values_only=True))
    a = pd.DataFrame([[r[2], r[3], r[4], r[44], r[47], r[50], r[48], r[52], r[53]] for r in rows],
                     columns=['mes', 'dia', 'hora', 'ano', 'Rn', 'T62', 'RH', 'P', 'PP']).apply(num)
    a = a.dropna(subset=['dia', 'hora'])
    h = a['hora']
    t = (pd.Timestamp('2008-01-01') + pd.to_timedelta(a['dia'] - 1, unit='D')
         + pd.to_timedelta((h // 100) * 60 + h % 100, unit='min'))
    a.index = t
    a = a[~a.index.duplicated()]
    a = a.reindex(pd.date_range(a.index.min(), a.index.max(), freq='30min'))
    a.loc[~a['T62'].between(*T_RANGE) | flatline(a['T62'], FLAT_H * 2), 'T62'] = np.nan
    a.loc[~a['RH'].between(*RH_RANGE), 'RH'] = np.nan
    a.loc[~a['Rn'].between(*RN_RANGE), 'Rn'] = np.nan
    a.loc[~a['P'].between(950, 1050), 'P'] = np.nan
    # T62 certificada a 10 min (pasta Inicio): média em bloco 3/3, rótulo no fim.
    # Usada para conferir a série de 30 min e completar T62 onde ela falta (dado medido, não interpolado).
    c = load_jaru_10min().reindex(a.index)
    both = a['T62'].notna() & c.notna()
    diff = (a['T62'] - c)[both]
    print(f'Jaru T62: 30 min vs 10 min certificada: n={both.sum()}, |dif| média={diff.abs().mean():.3f} K, '
          f'máx={diff.abs().max():.2f} K; pontos completados={int((a["T62"].isna() & c.notna()).sum())}')
    a['T_fonte'] = np.where(a['T62'].notna(), '30min', np.where(c.notna(), '10min_cert', ''))
    a['T62'] = a['T62'].fillna(c)
    out = pd.DataFrame({'T_C': a['T62'], 'Rn': a['Rn'], 'RH': a['RH'].clip(upper=100), 'T_fonte': a['T_fonte'],
                        'P_hPa': a['P'].fillna(a['P'].median()), 'PP_mm': a['PP']})
    out['q'] = q_specific(out['T_C'], out['RH'], out['P_hPa'])
    return out


def load_jaru_10min():
    sh = xlrd.open_workbook(os.path.join(SRC, 'Inicio', 'JAR_AWS_08_certificado.xls'), on_demand=True).sheet_by_name('Brutos')
    d = pd.DataFrame([sh.row_values(r)[:5] for r in range(1, sh.nrows)],
                     columns=['id', 'ano', 'dia', 'hora', 'T']).apply(num).dropna(subset=['dia', 'hora'])
    h = d['hora']
    d.index = (pd.Timestamp('2008-01-01') + pd.to_timedelta(d['dia'] - 1, unit='D')
               + pd.to_timedelta((h // 100) * 60 + h % 100, unit='min'))
    # A série certificada está em UTC; DADOS OK está em hora local (UTC-4): o melhor
    # alinhamento é -4 h (|dif| média 0,16 K contra 2,53 K sem deslocamento).
    d.index = d.index - pd.Timedelta(hours=4)
    t = d['T'][~d.index.duplicated()]
    t = t.reindex(pd.date_range(t.index.min().floor('D'), t.index.max(), freq='10min'))
    t[~t.between(*T_RANGE) | flatline(t, FLAT_H * 6)] = np.nan
    r = t.resample('30min', closed='right', label='right')
    return r.mean().where(r.count() == 3)


# ---------------------------------------------------------------- Sinop
SINOP_FILES = {'Apr01-Met.xls': 2001, 'May01-Met.xls': 2001, 'Cópia de Met-CO2 Jun-02.xls': 2002,
               'Met Aug-03.xls': 2003, 'Met Sep-03.xls': 2003, 'Met Nov-03.xls': 2003, 'Met Dec-03.xls': 2003}
P_SINOP = 965.0   # hPa, constante (423 m de altitude, sem barômetro nos arquivos)


def load_sinop():
    """Torre de Sinop (MT), 30 min. T a 40 m, Rn (NR-LITE) e Rsd (LI-200) acima do dossel (~28 m).
    Rn é descartado nos dias em que a razão diária Rn+/Rsd+ sai de [0,6; 1,05] (sensor com defeito
    em set-dez/2003). A razão muda entre anos (0,72 em 2002, 0,91 em 2003, 0,98 em 2001): o
    α de Sinop não é comparável entre anos nem com outros sítios."""
    parts = []
    for fn, yr in SINOP_FILES.items():
        sh = xlrd.open_workbook(os.path.join(SRC, 'Continuidade', fn), on_demand=True).sheet_by_index(0)
        hdr = [str(h).strip() for h in sh.row_values(0)]
        df = pd.DataFrame([sh.row_values(r) for r in range(1, sh.nrows)]).apply(num).replace(-6999, np.nan)
        col = lambda pref: hdr.index([h for h in hdr if h.startswith(pref)][0])
        rh = col('RH 41m') if any(h.startswith('RH 41m') for h in hdr) else hdr.index('RH')
        d = pd.DataFrame({'jd': df[col('Julian Day')], 'hora': df[col('Julian Day') + 1],
                          'T_C': df[col('T40')], 'T28_C': df[col('T28')], 'Rn': df[col('Rn (W')],
                          'Rsd': df[col('Rsd')], 'RH': df[rh], 'PP_mm': df[col('ppt')]}).dropna(subset=['jd', 'hora'])
        h = d['hora']
        d.index = (pd.Timestamp(f'{yr}-01-01') + pd.to_timedelta(d['jd'] - 1, unit='D')
                   + pd.to_timedelta((h // 100) * 60 + h % 100, unit='min'))
        d['arquivo'] = fn
        parts.append(d.drop(columns=['jd', 'hora']))
    d = pd.concat(parts).sort_index()
    d = d[~d.index.duplicated()]
    d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq='30min'))
    for c in ['T_C', 'T28_C']:
        d.loc[~d[c].between(*T_RANGE) | flatline(d[c], FLAT_H * 2), c] = np.nan
    d.loc[~d['RH'].between(*RH_RANGE), 'RH'] = np.nan
    for c in ['Rn', 'Rsd']:
        d.loc[~d[c].between(*RN_RANGE), c] = np.nan
    day = d.index.floor('D')
    ratio = (d['Rn'].clip(lower=0).groupby(day).transform('sum')
             / d['Rsd'].clip(lower=0).groupby(day).transform('sum'))
    d['Rn_razao_dia'] = ratio
    # razão fora de [0,6; 1,05], ou o período em que o sensor falhou de forma recorrente
    # (17/09 a 31/12/2003: razões de 0,02 a 18 e Rn noturno até -196 W/m2)
    bad = ~ratio.between(0.6, 1.05) | ((d.index >= '2003-09-17') & (d.index < '2004-01-01'))
    has = d['T_C'].notna().groupby(day).transform('any')
    print(f'Sinop: Rn descartado em {(bad & has).groupby(day).any().sum()} de {has.groupby(day).any().sum()} dias com dados')
    d.loc[bad, 'Rn'] = np.nan
    d['RH'] = d['RH'].clip(upper=100)
    d['P_hPa'] = P_SINOP
    d['q'] = q_specific(d['T_C'], d['RH'], d['P_hPa'])
    return d.drop(columns=['arquivo'])


# ------------------------------------------------------------- comuns
def label(df, site):
    df['estacao'] = ''
    for lab, a, b in SEASONS[site]:
        df.loc[a:b, 'estacao'] = lab
    df['evento'] = ''
    for lab, a, b in EVENTS.get(site, []):
        df.loc[a:b, 'evento'] = lab
    df['T_K'] = df['T_C'] + 273.15
    df['ok_TR'] = df['T_C'].notna() & df['Rn'].notna()
    df['ok_TRq'] = df['ok_TR'] & df['q'].notna()
    return df


def segments(df, site, okcol):
    rows = []
    for est, g in df[df['estacao'] != ''].groupby('estacao'):
        ok = g[okcol]
        # quebra também quando muda o ano/janela (índice não contíguo)
        brk = (ok != ok.shift()) | (g.index.to_series().diff() != pd.Timedelta('30min'))
        sid = brk.cumsum()
        for _, s in g[ok].groupby(sid[ok]):
            n = len(s)
            rows.append(dict(sitio=site, estacao=est, criterio=okcol, inicio=s.index[0],
                             fim=s.index[-1], n_pontos=n, dias=round(n / 48, 2),
                             util=n >= MIN_SEG_H * 2,
                             tem_evento=bool((s['evento'] != '').any())))
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    allseg, gaps = [], []
    for site, loader in [('jaru', load_jaru), ('sinop', load_sinop), ('mujui', load_mujui), ('cuieiras', load_cuieiras)]:
        df = label(loader(), site)
        df.index.name = 'fim_intervalo'
        df.to_csv(os.path.join(OUT, f'{site}_30min.csv'), float_format='%.6g')
        for c in ['ok_TR', 'ok_TRq']:
            allseg.append(segments(df, site, c))
        g = df.groupby(df.index.to_period('M')).agg(
            n=('T_C', 'size'), T_validos=('T_C', 'count'), Rn_validos=('Rn', 'count'),
            TR_validos=('ok_TR', 'sum'), TRq_validos=('ok_TRq', 'sum'))
        g['frac_TR'] = (g['TR_validos'] / g['n']).round(3)
        g.insert(0, 'sitio', site)
        gaps.append(g)
    seg = pd.concat(allseg).sort_values(['sitio', 'criterio', 'estacao', 'dias'], ascending=[True, True, True, False])
    seg.to_csv(os.path.join(OUT, 'segmentos.csv'), index=False)
    pd.concat(gaps).to_csv(os.path.join(OUT, 'cobertura_mensal.csv'))
    s = seg[(seg.criterio == 'ok_TR')]
    summ = s.groupby(['sitio', 'estacao']).apply(lambda x: pd.Series({
        'maior_seq_dias': x['dias'].max(),
        'segmentos_uteis': int(x['util'].sum()),
        'dias_em_segmentos_uteis': round(x.loc[x.util, 'dias'].sum(), 1),
        'dias_validos_total': round(x['dias'].sum(), 1)}))
    summ.to_csv(os.path.join(OUT, 'resumo_janelas.csv'))
    print(summ.to_string())


if __name__ == '__main__':
    main()
