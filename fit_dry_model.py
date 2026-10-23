"""Etapa 3: ajuste do modelo seco dT/dt = a*R - g*T^4 com protocolo fixo.

Protocolo (igual para todos os conjuntos):
- tempo em horas, dt = 0,5 h; T em K; R em W/m2  =>  a [K h-1 (W m-2)-1], g [K-3 h-1]
- forçante: no passo t_{i-1} -> t_i usa R[i] (média do intervalo que termina em t_i);
  a convenção do Artigo 1 (R[i-1]) roda como sensibilidade
- tiro múltiplo: cada segmento contínuo (>= 24 h) é cortado em janelas de 5 dias,
  cada janela integrada por RK4 a partir da T observada no seu início
- divisão cronológica das janelas: 70 % ajuste, 15 % validação (reservada à rede), 15 % teste
- chute inicial por regressão linear da derivada observada em [R, -T^4];
  otimização em log(a), log(g) com limites de x10 em torno do chute (mesma largura sempre)
- incerteza: bootstrap em blocos (reamostra janelas de ajuste), B = 200
- referência estilo Artigo 1: uma integração única no maior segmento, ajuste no segmento todo
"""
import os, sys, numpy as np, pandas as pd
from scipy.optimize import minimize

DATA = os.path.normpath(os.environ.get('DATA_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')))
OUT = sys.argv[1] if len(sys.argv) > 1 else DATA
DT = 0.5
WIN = 5 * 48
MIN_WIN = 48
B = 200
rng = np.random.default_rng(42)
# ONLY_SITES=mujui refaz só esses sítios; os outros têm os sorteios do bootstrap repetidos sem ajuste,
# para que os números saiam idênticos aos de uma rodada completa
ONLY = set(os.environ['ONLY_SITES'].split(',')) if os.environ.get('ONLY_SITES') else None


def skip(site):
    return ONLY is not None and site not in ONLY


def save_merged(rows, path, float_format):
    new = pd.DataFrame(rows)
    if ONLY is not None and os.path.exists(path):
        old = pd.read_csv(path)
        new = pd.concat([old[~old['sitio'].isin(ONLY)], new], ignore_index=True)
    new.to_csv(path, index=False, float_format=float_format)


def windows(df):
    """Lista de (T, R, t0) por janela, em ordem cronológica."""
    ok = df['ok_TR'].values
    idx = df.index
    brk = np.r_[True, (ok[1:] != ok[:-1]) | (np.diff(idx.values) != np.timedelta64(30, 'm'))]
    sid = np.cumsum(brk)
    out = []
    for s in np.unique(sid[ok]):
        m = (sid == s) & ok
        T, R, t = df['T_K'].values[m], df['Rn'].values[m], idx[m]
        if len(T) < MIN_WIN:
            continue
        for k in range(0, len(T), WIN):
            if len(T) - k >= MIN_WIN:
                out.append((T[k:k + WIN], R[k:k + WIN], t[k]))
    return out


def pack(ws, lag):
    L = max(len(w[0]) for w in ws)
    n = len(ws)
    T = np.full((n, L), np.nan); R = np.zeros((n, L)); M = np.zeros((n, L), bool)
    for i, (t, r, _) in enumerate(ws):
        T[i, :len(t)] = t; M[i, :len(t)] = True
        R[i, :len(r)] = r if lag == 'atual' else np.r_[r[0], r[:-1]]
    return T, R, M


def simulate(a, g, T, R, M):
    n, L = T.shape
    S = np.empty((n, L)); S[:, 0] = T[:, 0]
    for j in range(1, L):
        x, r = S[:, j - 1], R[:, j]
        f = lambda y: a * r - g * y ** 4
        k1 = f(x); k2 = f(x + 0.5 * DT * k1); k3 = f(x + 0.5 * DT * k2); k4 = f(x + DT * k3)
        S[:, j] = x + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return S


def mse(a, g, T, R, M):
    S = simulate(a, g, T, R, M)
    m = M.copy(); m[:, 0] = False
    e = (S - T)[m]
    return np.mean(e ** 2) if np.all(np.isfinite(e)) else 1e6


def initial_guess(T, R, M):
    d = np.full_like(T, np.nan)
    d[:, 1:-1] = (T[:, 2:] - T[:, :-2]) / (2 * DT)
    m = np.isfinite(d) & M
    X = np.c_[R[m], -T[m] ** 4]
    a0, g0 = np.linalg.lstsq(X, d[m], rcond=None)[0]
    if a0 <= 0 or g0 <= 0:   # fallback físico: equilíbrio médio e tau = 12 h
        Tm = np.nanmean(T); g0 = 1 / (4 * Tm ** 3 * 12); a0 = g0 * Tm ** 4 / np.mean(R[M])
    return a0, g0


def fit(T, R, M, x0=None):
    a0, g0 = x0 or initial_guess(T, R, M)
    u0 = np.log([a0, g0])
    bnd = [(u0[0] - np.log(10), u0[0] + np.log(10)), (u0[1] - np.log(10), u0[1] + np.log(10))]
    res = minimize(lambda u: mse(*np.exp(u), T, R, M), u0, method='L-BFGS-B', bounds=bnd)
    a, g = np.exp(res.x)
    at_bound = any(abs(res.x[k] - b) < 1e-3 for k in range(2) for b in bnd[k])
    return a, g, (a0, g0), at_bound


def metrics(a, g, T, R, M):
    S = simulate(a, g, T, R, M)
    m = M.copy(); m[:, 0] = False
    e = (S - T)[m]
    return np.sqrt(np.mean(e ** 2)), np.mean(S[m] - T[m])


def split(ws):
    n = np.cumsum([len(w[0]) for w in ws]); tot = n[-1]
    tr = [w for w, c in zip(ws, n) if c <= 0.70 * tot + 1e-9] or ws[:1]
    va = [w for w, c in zip(ws, n) if 0.70 * tot < c <= 0.85 * tot]
    te = [w for w, c in zip(ws, n) if c > 0.85 * tot]
    return tr, va, te


def main():
    rows = []
    for site in ['jaru', 'cuieiras', 'mujui']:
        df = pd.read_csv(os.path.join(DATA, f'{site}_30min.csv'), index_col=0, parse_dates=True)
        for est in [e for e in df['estacao'].dropna().unique()]:
            sub = df[df['estacao'] == est]
            ws = windows(sub)
            tr, va, te = split(ws)
            Tm = np.nanmean(np.concatenate([w[0] for w in tr]))
            for lag in ['atual', 'anterior']:
                Ttr, Rtr, Mtr = pack(tr, lag); Tte, Rte, Mte = pack(te, lag)
                a, g, (a0, g0), atb = fit(Ttr, Rtr, Mtr)
                rtr, _ = metrics(a, g, Ttr, Rtr, Mtr)
                rte, bte = metrics(a, g, Tte, Rte, Mte)
                row = dict(sitio=site, estacao=est, forcante=lag, janelas_ajuste=len(tr),
                           janelas_teste=len(te), dias_ajuste=round(Mtr.sum() / 48, 1),
                           a=a, g=g, tau_h=1 / (4 * g * Tm ** 3), a0=a0, g0=g0, no_limite=atb,
                           rmse_ajuste_K=rtr, rmse_teste_K=rte, vies_teste_K=bte)
                if lag == 'atual':
                    bs = []
                    for _ in range(B):
                        k = rng.integers(0, len(tr), len(tr))
                        Tb, Rb, Mb = pack([tr[i] for i in k], lag)
                        ab, gb, _, _ = fit(Tb, Rb, Mb, x0=(a, g))
                        bs.append((ab, gb, 1 / (4 * gb * Tm ** 3)))
                    bs = np.array(bs)
                    for j, nm in enumerate(['a', 'g', 'tau_h']):
                        row[nm + '_p05'], row[nm + '_p95'] = np.percentile(bs[:, j], [5, 95])
                    row['corr_a_g'] = np.corrcoef(np.log(bs[:, 0]), np.log(bs[:, 1]))[0, 1]
                    # estilo Artigo 1: integração única no maior segmento
                    longest = max(windows_full(sub), key=lambda w: len(w[0]))
                    Tl, Rl, Ml = pack([longest], 'anterior')
                    al, gl, _, _ = fit(Tl, Rl, Ml)
                    row['rmse_estilo_art1_K'], _ = metrics(al, gl, Tl, Rl, Ml)
                    row['dias_maior_segmento'] = round(len(longest[0]) / 48, 1)
                rows.append(row)
                print(site, est, lag, f"a={a:.4g} g={g:.4g} tau={row['tau_h']:.2f}h "
                      f"rmse tr/te={rtr:.2f}/{rte:.2f} vies={bte:.2f}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, 'modelo_seco_parametros.csv'), index=False, float_format='%.6g')


def windows_full(df):
    global WIN
    w0 = WIN; WIN = 10 ** 9
    try:
        return windows(df)
    finally:
        WIN = w0


if __name__ == '__main__':
    main()
