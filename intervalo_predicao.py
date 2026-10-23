"""Etapa 4c: intervalo de predição da linha de base linear.
O bootstrap em blocos mede só a incerteza dos parâmetros e cobre pouco das observações.
Aqui o intervalo de 90 % soma à simulação os quantis 5 % e 95 % dos resíduos nas janelas
de validação (conformal simples, sem usar o teste); a cobertura é medida no teste."""
import os, numpy as np, pandas as pd
import fit_dry_model as F
import fit_baseline as L

DATA = F.DATA


def main():
    par = pd.read_csv(os.path.join(DATA, 'linha_base_parametros.csv'))
    par = par[par['forcante'] == 'Rn']
    rows = []
    for _, s in par.iterrows():
        df = pd.read_csv(os.path.join(DATA, f"{s['sitio']}_30min.csv"), index_col=0, parse_dates=True)
        tr, va, te = F.split(L.windows_with(df[df['estacao'] == s['estacao']], 'Rn'))
        p = [np.log(s['a']), np.log(s['tau_h']), s['Teq_C'] + 273.15]
        ev = -L.resid(p, *F.pack(va, 'atual'))    # obs - sim
        et = -L.resid(p, *F.pack(te, 'atual'))
        lo, hi = np.percentile(ev, [5, 95])
        rows.append(dict(sitio=s['sitio'], estacao=s['estacao'], dias_validacao=round(len(ev) / 48, 1),
                         q05_K=lo, q95_K=hi, largura_IC90_K=hi - lo,
                         cobertura_teste=np.mean((et >= lo) & (et <= hi))))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(DATA, 'intervalo_predicao_base.csv'), index=False, float_format='%.4g')


if __name__ == '__main__':
    main()
