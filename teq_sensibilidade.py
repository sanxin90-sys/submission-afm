"""Revisão (parecer dos colegas): sensibilidade do Teq variável (teq_variavel.py) à duração da média móvel.
Repete as variantes janela (prognóstica) e contínua (diagnóstica) nos conjuntos principais com médias das 24, 72 e
120 h anteriores, sem bootstrap. Com 72 e 120 h, há mais janelas sem o histórico completo, que usam o Teq0.
Saída: dados/teq_variavel_sensibilidade.csv."""
import os, pandas as pd
from multiprocessing import Pool
import teq_variavel as TV

DATA = TV.DATA
NPROC = int(os.environ.get('NPROC', 2))


def run(job):
    site, est, H = job
    TV.pack_ctx.__defaults__ = (H,)
    df = pd.read_csv(os.path.join(DATA, f'{site}_30min.csv'), index_col=0, parse_dates=True)
    sub = df[df['estacao'] == est].copy()
    sub['ok_TR'] = sub['T_C'].notna() & sub['Rn'].notna()
    rows = TV.evaluate(sub, dict(sitio=site, estacao=est, horas_media=H // 2))
    print(site, est, H // 2, 'h', ' '.join(f"{r['variante']}: {r['rmse_teste_K']:.2f}/{r['vies_teste_K']:+.2f}"
                                           for r in rows), flush=True)
    return rows


def main():
    par = pd.read_csv(os.path.join(DATA, 'linha_base_parametros.csv'))
    sets = [tuple(x) for x in par.loc[par['forcante'] == 'Rn', ['sitio', 'estacao']].values]
    jobs = [(s, e, H) for s, e in sets for H in (48, 144, 240)]
    with Pool(NPROC) as pool:
        res = pool.map(run, jobs, chunksize=1)
    pd.DataFrame([r for rs in res for r in rs]).to_csv(os.path.join(DATA, 'teq_variavel_sensibilidade.csv'),
                                                       index=False, float_format='%.5g')


if __name__ == '__main__':
    main()
