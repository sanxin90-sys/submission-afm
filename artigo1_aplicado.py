"""Artigo 1 aplicado aos sítios do Artigo 2: palpites iniciais e modelo caixa-cinza original.

(A) Palpites iniciais do modelo seco dT/dt = a R - g T^4 (Artigo 1), por conjunto (sítio x estação, forçante Rn):
    - A1_seca: a0 = 1,2 e g0 = 1e-8 em unidades de 10^6 s, limites /40 e x40 (Code_Dry.ipynb)
    - A1_chuvosa: a0 = 0,2 e g0 = 1e-8, limites /4 e x4 (Code_Wet.ipynb; é o palpite da Tabela I do Artigo 1)
    - dados: regressão da derivada observada em [R, -T^4] (como fit_dry_model.initial_guess), limites /40 e x40
    Cada palpite roda em dois protocolos:
    - art1: como nos notebooks (integração única do maior segmento contínuo, R do passo anterior, L-BFGS-B
      nos parâmetros brutos com gradiente por diferença finita, tempo em 10^6 s); RMSE dentro da amostra
    - art2: janelas de 5 dias, divisão cronológica, L-BFGS-B em log(a) e log(g), R do passo atual; RMSE de teste
(A') Base linear dT/dt = a R - (T - Teq)/tau: o ponto final de cada partida do multi-start (tau0 = 1, 3, 10, 30 h)
    e de duas partidas tiradas dos palpites do Artigo 1 (a0 = a1/277,8; tau0 = 1/(4 g0 T^3)).
(B) Caixa-cinza do Artigo 1 (T^4 + NN(T,R)) no protocolo do Artigo 2: mesmas janelas e divisão de
    fit_nn_correction.py, N_SEEDS redes e média do conjunto; compara com a base linear e linear + NN(T,R).
(C) Protocolo completo do Artigo 1 no maior segmento: T^4 ajustado como em (A, art1, melhor dos dois palpites
    do Artigo 1), rede treinada na derivada com divisão aleatória (Code_Dry, random_state = 42) ou cronológica
    (Code_Wet), integração de Euler do segmento inteiro; RMSE dentro da amostra (métrica das Tabelas I e VI do
    Artigo 1) e, na divisão cronológica, também nos 15 % finais.
Saídas: dados/palpites_iniciais.csv, dados/multistart_linear.csv, dados/artigo1_aplicado.csv."""
import os, numpy as np, pandas as pd, torch
from multiprocessing import Pool
from scipy.optimize import minimize
import fit_dry_model as F
import fit_baseline as L
import fit_nn_correction as NN

DATA = F.DATA
U = 1e6 / 3600.0          # horas por unidade de tempo do Artigo 1 (10^6 s)
DT1 = F.DT / U            # passo de 30 min nessa unidade (0,0018)
A1 = {'A1_seca': (1.2, 1e-8, 40.0), 'A1_chuvosa': (0.2, 1e-8, 4.0)}
N_SEEDS = int(os.environ.get('N_SEEDS', 5))
NPROC = int(os.environ.get('NPROC', 1))


def guess_data(T, R):
    """Regressão da derivada central em [R, -T^4] (horas); devolve (a0, g0, usou_alternativa)."""
    d = (T[2:] - T[:-2]) / (2 * F.DT)
    X = np.c_[R[1:-1], -T[1:-1] ** 4]
    a0, g0 = np.linalg.lstsq(X, d, rcond=None)[0]
    if a0 > 0 and g0 > 0:
        return a0, g0, False
    Tm = np.mean(T); g0 = 1 / (4 * Tm ** 3 * 12); a0 = g0 * Tm ** 4 / np.mean(R)
    return a0, g0, True


def rk4_series(alpha, gamma, T0, R, dt):
    """Como Code_Dry.ipynb: RK4 a partir de T[0], com R[i-1] no passo i-1 -> i."""
    out = [T0]; x = T0
    for r in R[:-1]:
        ar = alpha * r
        k1 = dt * (ar - gamma * x ** 4); y = x + 0.5 * k1
        k2 = dt * (ar - gamma * y ** 4); y = x + 0.5 * k2
        k3 = dt * (ar - gamma * y ** 4); y = x + k3
        k4 = dt * (ar - gamma * y ** 4)
        x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        out.append(x)
    return np.array(out)


def art1_obj(p, T, Rl):
    try:
        S = rk4_series(float(p[0]), float(p[1]), float(T[0]), Rl, DT1)
    except OverflowError:
        return 1e6
    v = np.mean((T - S) ** 2)
    return v if np.isfinite(v) else 1e6


def fit_art1(T, R, guess, k):
    """minimize(objective, initial_guess, bounds=...) como no Artigo 1 (L-BFGS-B, parâmetros brutos)."""
    bnd = [(guess[0] / k, guess[0] * k), (guess[1] / k, guess[1] * k)]
    r = minimize(art1_obj, list(guess), args=(T, R.tolist()), bounds=bnd)
    at = any(np.isclose(r.x[j], b, rtol=1e-6, atol=0) for j in range(2) for b in bnd[j])
    return r.x, np.sqrt(r.fun), at, r.nfev


def fit_art2(T, R, M, guess_h, k):
    u0 = np.log(guess_h)
    bnd = [(u0[j] - np.log(k), u0[j] + np.log(k)) for j in range(2)]
    r = minimize(lambda u: F.mse(*np.exp(u), T, R, M), u0, method='L-BFGS-B', bounds=bnd)
    at = any(abs(r.x[j] - b) < 1e-3 for j in range(2) for b in bnd[j])
    return np.exp(r.x), at, r.nfev


def multistart(T, R, M):
    Tm, Rm = np.nanmean(T[M]), np.mean(R[M])
    R99 = np.percentile(R[M], 99)
    starts = [(f'tau0 = {t:g} h', 4.0 / (R99 * t), t) for t in [1, 3, 10, 30]]
    starts += [(nm, a1 / U, U / (4 * g1 * Tm ** 3)) for nm, (a1, g1, _) in A1.items()]
    rows = []
    for nm, a0, tau0 in starts:
        p0 = [np.log(a0), np.log(tau0), Tm - a0 * Rm * tau0]
        r = minimize(L.loss, p0, args=(T, R, M), method='Nelder-Mead',
                     options=dict(maxiter=4000, xatol=1e-6, fatol=1e-9))
        r2 = minimize(L.loss, r.x, args=(T, R, M), method='L-BFGS-B')
        pf, lf = (r2.x, r2.fun) if r2.fun <= r.fun else (r.x, r.fun)
        rows.append(dict(partida=nm, a0=a0, tau0_h=tau0, Teq0_C=p0[2] - 273.15,
                         tau_NM_h=np.exp(r.x[1]), Teq_NM_C=r.x[2] - 273.15, rmse_NM_K=np.sqrt(r.fun),
                         nfev_NM=r.nfev, a=np.exp(pf[0]), tau_h=np.exp(pf[1]), Teq_C=pf[2] - 273.15,
                         rmse_ajuste_K=np.sqrt(lf)))
    best = min(rows, key=lambda d: d['rmse_ajuste_K'])
    for d in rows:
        d['mesmo_otimo'] = bool(abs(d['tau_h'] / best['tau_h'] - 1) < 0.01 and abs(d['Teq_C'] - best['Teq_C']) < 0.05)
    return rows


def samples_t4(ws, a, g):
    X, y = [], []
    for T, R, q, _ in ws:
        dT = (T[2:] - T[:-2]) / (2 * F.DT)
        Rm = 0.5 * (R[1:-1] + R[2:]); Ti = T[1:-1]
        y.append(dT - (a * Rm - g * Ti ** 4)); X.append(np.c_[Ti, Rm])
    return np.vstack(X), np.concatenate(y)


def sim_t4_corr(a, g, nets, T, R, M):
    n, Lw = T.shape
    S = np.empty((n, Lw)); S[:, 0] = T[:, 0]

    def f(y, r):
        c = np.mean([p(np.c_[y, r]) for p in nets], axis=0) if nets else 0.0
        return a * r - g * y ** 4 + c
    for j in range(1, Lw):
        x, r = S[:, j - 1], R[:, j]
        k1 = f(x, r); k2 = f(x + 0.5 * F.DT * k1, r); k3 = f(x + 0.5 * F.DT * k2, r); k4 = f(x + F.DT * k3, r)
        S[:, j] = np.clip(x + F.DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4), 250, 330)
    m = M.copy(); m[:, 0] = False
    e = (S - T)[m]
    return np.sqrt(np.mean(e ** 2)), np.mean(e)


def split_art1(n, modo):
    if modo == 'aleatoria':   # sklearn train_test_split(test_size=0.3, random_state=42) duas vezes
        perm = np.random.RandomState(42).permutation(n)
        nt = int(np.ceil(0.3 * n)); tmp, tr = perm[:nt], perm[nt:]
        perm2 = np.random.RandomState(42).permutation(len(tmp))
        nt2 = int(np.ceil(0.5 * len(tmp)))
        return tr, tmp[perm2[nt2:]], tmp[perm2[:nt2]]
    ntr, nva = int(n * 0.70), int(n * 0.15)
    return np.arange(ntr), np.arange(ntr, ntr + nva), np.arange(ntr + nva, n)


def euler_corr(a, g, preds, T0, R):
    """Integração de Euler do Artigo 1 (R[i-1]); preds = lista de redes, correção = média."""
    out = np.empty(len(R)); x = out[0] = T0
    for i in range(1, len(R)):
        r = R[i - 1]
        c = np.mean([p(np.array([[x, r]]))[0] for p in preds]) if preds else 0.0
        x = min(max(x + F.DT * (a * r - g * x ** 4 + c), 200.0), 400.0)
        out[i] = x
    return out


def run(job):
    site, est = job
    torch.set_num_threads(1)
    df = pd.read_csv(os.path.join(DATA, f'{site}_30min.csv'), index_col=0, parse_dates=True)
    sub = df[df['estacao'] == est]
    tag = dict(sitio=site, estacao=est)
    # maior segmento contínuo (protocolo do Artigo 1)
    seg = max(F.windows_full(sub), key=lambda w: len(w[0]))
    Ts, Rs = seg[0], seg[1]
    Tm_s = Ts.mean()
    ag, gg, alt = guess_data(Ts, Rs)
    guesses = {nm: ((a1, g1), k) for nm, (a1, g1, k) in A1.items()}
    guesses['dados'] = ((ag * U, gg * U), 40.0)
    # janelas (protocolo do Artigo 2), como fit_baseline.py
    tr, va, te = F.split(L.windows_with(sub, 'Rn'))
    Ttr, Rtr, Mtr = F.pack(tr, 'atual'); Tte, Rte, Mte = F.pack(te, 'atual')
    Tm_w = np.nanmean(Ttr[Mtr])
    pal = []
    art1_fit = {}
    for nm, ((a1, g1), k) in guesses.items():
        x, rmse1, at1, nf1 = fit_art1(Ts, Rs, (a1, g1), k)
        art1_fit[nm] = (x, rmse1)
        a_h, g_h = x[0] / U, x[1] / U
        pal.append(dict(tag, protocolo='art1', palpite=nm, fator_limite=k, a0_A1=a1, g0_A1=g1,
                        palpite_alternativo=alt if nm == 'dados' else False,
                        a_A1=x[0], g_A1=x[1], a_h=a_h, g_h=g_h, tau_h=1 / (4 * g_h * Tm_s ** 3),
                        rmse_K=rmse1, tipo_rmse='dentro da amostra (segmento único)', no_limite=at1,
                        avaliacoes=nf1, dias=round(len(Ts) / 48, 1)))
        (a2, g2), at2, nf2 = fit_art2(Ttr, Rtr, Mtr, (a1 / U, g1 / U), k)
        r2, b2 = F.metrics(a2, g2, Tte, Rte, Mte)
        pal.append(dict(tag, protocolo='art2', palpite=nm, fator_limite=k, a0_A1=a1, g0_A1=g1,
                        palpite_alternativo=alt if nm == 'dados' else False,
                        a_A1=a2 * U, g_A1=g2 * U, a_h=a2, g_h=g2, tau_h=1 / (4 * g2 * Tm_w ** 3),
                        rmse_K=r2, vies_K=b2, tipo_rmse='teste (janelas)', no_limite=at2, avaliacoes=nf2,
                        dias=round(Mtr.sum() / 48, 1)))
        print(site, est, nm, f'art1 a={x[0]:.4g} g={x[1]:.3g} rmse={rmse1:.2f} lim={at1} | '
              f'art2 a={a2 * U:.4g} g={g2 * U:.3g} rmse={r2:.2f} lim={at2}', flush=True)
    ms = [dict(tag, **d) for d in multistart(Ttr, Rtr, Mtr)]
    print(site, est, 'multi-start:', [(d['partida'], round(d['tau_h'], 2), d['mesmo_otimo']) for d in ms], flush=True)

    # (B) T^4 + NN(T,R) nas janelas de fit_nn_correction.py
    out = dict(tag)
    wq = NN.windows_q(sub)
    trq, vaq, teq = F.split(wq)
    Tq, Rq, Qq, Mq = NN.packq(trq)
    a4, g4, _, at4 = F.fit(Tq, Rq, Mq)
    Tt, Rt, Qt, Mt = NN.packq(teq)
    r0, b0 = sim_t4_corr(a4, g4, [], Tt, Rt, Mt)
    Xtr, ytr = samples_t4(trq, a4, g4); Xva, yva = samples_t4(vaq, a4, g4); Xte, yte = samples_t4(teq, a4, g4)
    nets, eps = [], []
    for s in range(N_SEEDS):
        pr, ep, _ = NN.train(Xtr, ytr, Xva, yva, seed=s)
        nets.append(pr); eps.append(ep)
    yhat = np.mean([p(Xte) for p in nets], axis=0)
    r1, b1 = sim_t4_corr(a4, g4, nets, Tt, Rt, Mt)
    out.update(a_T4_A1=a4 * U, g_T4_A1=g4 * U, tau_T4_h=1 / (4 * g4 * np.nanmean(Tq[Mq]) ** 3), T4_no_limite=at4,
               rmse_teste_T4_K=r0, vies_teste_T4_K=b0, rmse_teste_T4_NN_K=r1, vies_teste_T4_NN_K=b1,
               R2_derivada_teste_T4_NN=1 - np.mean((yte - yhat) ** 2) / np.var(yte), epocas_T4_NN=np.mean(eps))
    print(site, est, f'(B) T4 {r0:.2f} -> T4+NN {r1:.2f} K', flush=True)

    # (C) protocolo completo do Artigo 1 no maior segmento
    best = min(['A1_seca', 'A1_chuvosa'], key=lambda nm: art1_fit[nm][1])
    a_h, g_h = art1_fit[best][0] / U
    out.update(dias_segmento=round(len(Ts) / 48, 1), palpite_art1=best, a_art1_A1=a_h * U, g_art1_A1=g_h * U,
               rmse_T4_rk4_K=art1_fit[best][1])
    S0 = euler_corr(a_h, g_h, [], Ts[0], Rs)
    out['rmse_T4_euler_K'] = np.sqrt(np.mean((Ts - S0) ** 2))
    d = np.empty_like(Ts)
    d[1:-1] = (Ts[2:] - Ts[:-2]) / (2 * F.DT); d[0] = (Ts[1] - Ts[0]) / F.DT; d[-1] = (Ts[-1] - Ts[-2]) / F.DT
    X = np.c_[Ts, Rs]; y = d - (a_h * Rs - g_h * Ts ** 4)
    for modo in ['aleatoria', 'cronologica']:
        itr, iva, ite = split_art1(len(Ts), modo)
        preds, r_seed = [], []
        for s in range(N_SEEDS):
            pr, _, _ = NN.train(X[itr], y[itr], X[iva], y[iva], seed=s)
            preds.append(pr)
            S = euler_corr(a_h, g_h, [pr], Ts[0], Rs)
            r_seed.append(np.sqrt(np.mean((Ts - S) ** 2)))
        Sm = euler_corr(a_h, g_h, preds, Ts[0], Rs)
        out[f'rmse_T4_NN_{modo}_media_K'] = np.mean(r_seed)
        out[f'rmse_T4_NN_{modo}_min_K'] = np.min(r_seed)
        out[f'rmse_T4_NN_{modo}_max_K'] = np.max(r_seed)
        out[f'rmse_T4_NN_{modo}_conjunto_K'] = np.sqrt(np.mean((Ts - Sm) ** 2))
        if modo == 'cronologica':
            out['rmse_T4_NN_cronologica_15finais_K'] = np.sqrt(np.mean((Ts[ite] - Sm[ite]) ** 2))
            out['rmse_T4_cronologica_15finais_K'] = np.sqrt(np.mean((Ts[ite] - S0[ite]) ** 2))
        print(site, est, f'(C) {modo}: T4 {out["rmse_T4_euler_K"]:.2f} -> T4+NN {np.mean(r_seed):.2f} '
              f'({np.min(r_seed):.2f}-{np.max(r_seed):.2f}) K', flush=True)
    return pal, ms, out


def main():
    par = pd.read_csv(os.path.join(DATA, 'linha_base_parametros.csv'))
    jobs = [tuple(x) for x in par.loc[par['forcante'] == 'Rn', ['sitio', 'estacao']].values]
    with Pool(NPROC) as pool:
        res = pool.map(run, jobs, chunksize=1)
    pd.DataFrame([r for p, _, _ in res for r in p]).to_csv(
        os.path.join(DATA, 'palpites_iniciais.csv'), index=False, float_format='%.6g')
    pd.DataFrame([r for _, m, _ in res for r in m]).to_csv(
        os.path.join(DATA, 'multistart_linear.csv'), index=False, float_format='%.6g')
    pd.DataFrame([o for _, _, o in res]).to_csv(
        os.path.join(DATA, 'artigo1_aplicado.csv'), index=False, float_format='%.5g')


if __name__ == '__main__':
    main()
