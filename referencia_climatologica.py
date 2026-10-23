"""Referência ingênua: ciclo diurno climatológico.
Para cada conjunto (sítio x estação, forçante Rn), a temperatura prevista em cada instante das janelas de
teste é a média da T observada no mesmo horário (48 horários de 30 min) nas janelas de ajuste. Mesmas
janelas, divisão e máscara de fit_baseline.py (o primeiro ponto de cada janela fica fora).
Também calcula, no estilo do Artigo 1, o RMSE do ciclo médio da própria estação (dentro da amostra)
no maior segmento contínuo, para comparar com os RMSE publicados (1,40 K seca; 1,13 K chuvosa).
Saída: dados/referencia_climatologica.csv."""
import os, numpy as np, pandas as pd
import fit_dry_model as F
import fit_baseline as L

DATA = F.DATA


def slots(ws):
    """Horário (0..47) de cada ponto das janelas, no formato de F.pack."""
    n, Lw = len(ws), max(len(w[0]) for w in ws)
    S = np.zeros((n, Lw), int)
    for i, (T, R, t0) in enumerate(ws):
        h = pd.DatetimeIndex(pd.date_range(t0, periods=len(T), freq='30min'))
        S[i, :len(T)] = (h.hour * 2 + h.minute // 30).values
    return S


def main():
    par = pd.read_csv(os.path.join(DATA, 'linha_base_parametros.csv'))
    par = par[par['forcante'] == 'Rn']
    rows = []
    for _, s in par.iterrows():
        df = pd.read_csv(os.path.join(DATA, f"{s['sitio']}_30min.csv"), index_col=0, parse_dates=True)
        sub = df[df['estacao'] == s['estacao']]
        tr, va, te = F.split(L.windows_with(sub, 'Rn'))
        Ttr, _, Mtr = F.pack(tr, 'atual'); Str = slots(tr)
        clim = np.array([np.nanmean(Ttr[Mtr & (Str == k)]) for k in range(48)])
        Tte, _, Mte = F.pack(te, 'atual'); Ste = slots(te)
        m = Mte.copy(); m[:, 0] = False
        e = clim[Ste][m] - Tte[m]
        # estilo Artigo 1: ciclo médio do próprio segmento, dentro da amostra
        seg = max(F.windows_full(sub), key=lambda w: len(w[0]))
        Sg = slots([seg])[0]; Tg = seg[0]
        cg = np.array([Tg[Sg == k].mean() for k in range(48)])
        rows.append(dict(sitio=s['sitio'], estacao=s['estacao'], rmse_teste_clim_K=np.sqrt(np.mean(e ** 2)),
                         vies_teste_clim_K=np.mean(e), rmse_teste_base_K=s['rmse_teste_K'],
                         rmse_clim_segmento_dentro_K=np.sqrt(np.mean((cg[Sg] - Tg) ** 2)),
                         dias_segmento=round(len(Tg) / 48, 1)))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(DATA, 'referencia_climatologica.csv'), index=False, float_format='%.4g')


if __name__ == '__main__':
    main()
