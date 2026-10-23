"""Revisão (parecer dos colegas): tempo de relaxação em várias alturas, dentro e acima do dossel.
Perfis de temperatura do ar dos próprios sítios, todos com o mesmo saldo de radiação do conjunto principal:
- Cuieiras (K34; dossel de 30 a 40 m, Marques Filho et al., 2005): 5,2; 15,6; 28; 35,5; 42,5 e 51,1 m (AWS);
- Sinop (dossel de cerca de 28 m): 1, 4, 12, 20, 28, 30 e 40 m (termopares psicrométricos);
- Mujuí (dossel de cerca de 20 m): 5 e 12 m.
Para cada sítio e estação dos conjuntos principais, a linha de base é ajustada em cada altura com o protocolo de
fit_baseline.py (janelas de 5 dias, divisão 70/15/15, bootstrap em blocos com B réplicas), usando só os instantes
em que todas as alturas do sítio têm temperatura válida, para que as alturas sejam comparadas nos mesmos dias
(alturas com menos de 20 dias válidos na estação ficam fora: 5,2 m na chuvosa do Cuieiras).
Também: amplitude do ciclo diurno médio e atraso entre os picos médios de Rn e T (interpolação parabólica).
Saída: dados/alturas_parametros.csv."""
import os, numpy as np, pandas as pd, xlrd
from multiprocessing import Pool
import fit_dry_model as F
import fit_baseline as L
import preprocess as P

DATA = F.DATA
SRC = P.SRC
B = int(os.environ.get('B', 100))
NPROC = int(os.environ.get('NPROC', 1))
DOSSEL = {'cuieiras': 35.0, 'sinop': 28.0, 'mujui': 20.0}


def qc_t(s, n_flat):
    return s.where(s.between(*P.T_RANGE) & ~P.flatline(s, n_flat))


def perfil_cuieiras():
    cols = {3: 5.2, 4: 15.6, 5: 28.0, 6: 35.5, 7: 42.5, 8: 51.1}
    m = P.load_sheets('1. Variáveis Meteorológicas Cuieiras.xls', {k: f'T{v}' for k, v in cols.items()})
    m = m.reindex(pd.date_range(m.index.min(), m.index.max(), freq='30min'))
    return pd.DataFrame({z: qc_t(m[f'T{z}'], P.FLAT_H * 2) for z in cols.values()})


def perfil_sinop():
    parts = []
    for fn, yr in P.SINOP_FILES.items():
        sh = xlrd.open_workbook(os.path.join(SRC, 'Continuidade', fn), on_demand=True).sheet_by_index(0)
        hdr = [str(h).strip() for h in sh.row_values(0)]
        df = pd.DataFrame([sh.row_values(r) for r in range(1, sh.nrows)]).apply(P.num).replace(-6999, np.nan)
        first = lambda name: [i for i, h in enumerate(hdr) if h.split(' ')[0] == name][0]
        jd = [i for i, h in enumerate(hdr) if h.startswith('Julian Day')][0]
        d = pd.DataFrame({'jd': df[jd], 'hora': df[jd + 1]})
        for z in [1, 4, 12, 20, 28, 30, 40]:
            d[float(z)] = df[first(f'T{z}')]
        d = d.dropna(subset=['jd', 'hora'])
        h = d['hora']
        d.index = (pd.Timestamp(f'{yr}-01-01') + pd.to_timedelta(d['jd'] - 1, unit='D')
                   + pd.to_timedelta((h // 100) * 60 + h % 100, unit='min'))
        parts.append(d.drop(columns=['jd', 'hora']))
    d = pd.concat(parts).sort_index()
    d = d[~d.index.duplicated()]
    d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq='30min'))
    return pd.DataFrame({z: qc_t(d[z], P.FLAT_H * 2) for z in d.columns})


def perfil_mujui(base):
    return pd.DataFrame({5.0: base['T5_C'], 12.0: base['T_C']})


def diurnal(T, R, idx):
    """Amplitude (K) do ciclo diurno médio de T e atraso (h) do pico médio de T em relação ao de R."""
    hh = (idx.hour * 60 + idx.minute) // 30
    cT = pd.Series(T).groupby(hh).mean().reindex(range(48))
    cR = pd.Series(R).groupby(hh).mean().reindex(range(48))

    def pico(c):
        k = int(np.nanargmax(c.values)); y0, y1, y2 = c.values[(k - 1) % 48], c.values[k], c.values[(k + 1) % 48]
        den = y0 - 2 * y1 + y2
        return (k + (0.5 * (y0 - y2) / den if den != 0 else 0.0)) * 0.5
    return cT.max() - cT.min(), (pico(cT) - pico(cR)) % 24


def run(job):
    site, est, z, sub, seed = job
    rng = np.random.default_rng(seed)
    ws = F.windows(sub)
    tr, va, te = F.split(ws)
    Ttr, Rtr, Mtr = F.pack(tr, 'atual'); Tte, Rte, Mte = F.pack(te, 'atual')
    p = L.fit(Ttr, Rtr, Mtr)
    ete = L.resid(p, Tte, Rte, Mte)
    bs = []
    for _ in range(B):
        k = rng.integers(0, len(tr), len(tr))
        bs.append(L.fit(*F.pack([tr[i] for i in k], 'atual'), p0=p))
    bs = np.array(bs)
    ok = sub['ok_TR'].values
    amp, atraso = diurnal(sub['T_K'].values[ok], sub['Rn'].values[ok], sub.index[ok])
    row = dict(sitio=site, estacao=est, altura_m=z, dossel_m=DOSSEL[site], z_sobre_h=z / DOSSEL[site],
               dias_em_janelas=round(sum(len(w[0]) for w in ws) / 48, 1), a=np.exp(p[0]), tau_h=np.exp(p[1]),
               Teq_C=p[2] - 273.15, rmse_teste_K=np.sqrt(np.mean(ete ** 2)), vies_teste_K=np.mean(ete),
               amplitude_ciclo_medio_K=amp, atraso_pico_R_T_h=atraso, T_media_C=sub['T_C'][sub['ok_TR']].mean())
    row['tau_h_p05'], row['tau_h_p95'] = np.percentile(np.exp(bs[:, 1]), [5, 95])
    print(site, est, z, f"tau={row['tau_h']:.2f} [{row['tau_h_p05']:.2f}-{row['tau_h_p95']:.2f}] h, "
          f"atraso={atraso:.2f} h, RMSE={row['rmse_teste_K']:.2f} K", flush=True)
    return row


def main():
    jobs, seed = [], 500
    for site in ['cuieiras', 'sinop', 'mujui']:
        base = pd.read_csv(os.path.join(DATA, f'{site}_30min.csv'), index_col=0, parse_dates=True)
        prof = {'cuieiras': perfil_cuieiras, 'sinop': perfil_sinop}.get(site, lambda: perfil_mujui(base))()
        prof = prof.reindex(base.index)
        for est in [e for e in base['estacao'].dropna().unique() if e]:
            e = (base['estacao'] == est) & base['Rn'].notna()
            zs = [z for z in prof.columns if (e & prof[z].notna()).sum() >= 20 * 48]   # alturas com >= 20 dias
            m = e & prof[zs].notna().all(axis=1)
            for z in zs:
                sub = pd.DataFrame({'T_C': prof[z], 'Rn': base['Rn']})[base['estacao'] == est].copy()
                sub['T_K'] = sub['T_C'] + 273.15
                sub['ok_TR'] = m[base['estacao'] == est].values
                if sum(len(w[0]) for w in F.windows(sub)) < 10 * 48:
                    print(site, est, z, 'sem dados suficientes'); continue
                jobs.append((site, est, float(z), sub, seed)); seed += 1
    with Pool(NPROC) as pool:
        rows = pool.map(run, jobs, chunksize=1)
    pd.DataFrame(rows).to_csv(os.path.join(DATA, 'alturas_parametros.csv'), index=False, float_format='%.5g')


if __name__ == '__main__':
    main()
