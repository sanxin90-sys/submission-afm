"""Réplica do ajuste do modelo seco do Artigo 1 (Tabela I) nas estações completas de Jaru 2008.
Objetivo idêntico ao de Code_Dry/Code_Wet: integração RK4 única da estação inteira, R[i-1], MSE de todos
os pontos, tempo em 10^6 s. As poucas lacunas da série QC (1 ponto na seca, 10 valores na chuvosa) são
interpoladas só aqui, porque o notebook exige série contínua.
- 'artigo1': minimize(objective, palpite, bounds) = L-BFGS-B nos parâmetros brutos, como nos notebooks
- 'otimo': Nelder-Mead em log(a), log(g) a partir de três palpites (o ótimo do mesmo objetivo)
Saída: dados/replica_artigo1.csv."""
import os, numpy as np, pandas as pd
from scipy.optimize import minimize
import fit_dry_model as F
import artigo1_aplicado as A

DATA = F.DATA
PUB = {'seca': (1.1588, 2.05e-8, 3.66), 'chuvosa': (0.2710, 3.75e-9, 2.5693)}


def main():
    d = pd.read_csv(os.path.join(DATA, 'jaru_30min.csv'), index_col=0, parse_dates=True)
    rows = []
    for est, guess, k in [('seca', (1.2, 1e-8), 40.0), ('chuvosa', (0.2, 1e-8), 4.0)]:
        s = d[d['estacao'] == est].sort_index()
        T = s['T_K'].interpolate(limit_direction='both').values
        R = s['Rn'].interpolate(limit_direction='both').values
        Tm, Rm, T4 = T.mean(), R.mean(), (T ** 4).mean()
        base = dict(estacao=est, pontos=len(T), interpolados=int(s['T_K'].isna().sum() + s['Rn'].isna().sum()),
                    Rn_media=Rm, T_media_C=Tm - 273.15)
        a, g, rp = PUB[est]
        rows.append(dict(base, procedimento='publicado (Tabela I)', a_A1=a, g_A1=g, rmse_K=rp))
        x, rm, at, nf = A.fit_art1(T, R, guess, k)
        rows.append(dict(base, procedimento='artigo1 (L-BFGS-B bruto)', palpite=str(guess), a_A1=x[0], g_A1=x[1],
                         rmse_K=rm, no_limite=at, avaliacoes=nf))
        for g0 in [guess, (0.7, 1.2e-8), (0.3, 4e-9)]:
            r = minimize(lambda u: A.art1_obj(np.exp(u), T, R.tolist()), np.log(g0), method='Nelder-Mead',
                         options=dict(xatol=1e-8, fatol=1e-12, maxiter=3000))
            rows.append(dict(base, procedimento='otimo (Nelder-Mead em log)', palpite=str(g0), a_A1=np.exp(r.x[0]),
                             g_A1=np.exp(r.x[1]), rmse_K=np.sqrt(r.fun), avaliacoes=r.nfev))
        for row in rows[-5:]:
            row['tau_h'] = A.U / (4 * row['g_A1'] * Tm ** 3)
            row['aR_sobre_gT4'] = row['a_A1'] * Rm / (row['g_A1'] * T4)
            print({kk: (round(v, 5) if isinstance(v, float) else v) for kk, v in row.items()}, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(DATA, 'replica_artigo1.csv'), index=False, float_format='%.5g')


if __name__ == '__main__':
    main()
