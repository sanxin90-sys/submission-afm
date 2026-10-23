"""Etapa 4: correção aditiva por rede neural sobre a linha de base linear
    dT/dt = a*R - (T - Teq)/tau + NN(x),  x = (T, R) ou (T, R, q)
Protocolo fixo para todos os conjuntos (igual ao do Artigo 1, exceto a divisão, que
aqui é cronológica em todos):
- alvo: resíduo da derivada observada (diferença central dentro de cada janela) em relação
  à linha de base; R e q no ponto = média dos dois intervalos vizinhos
- MLP [32, 16], LeakyReLU 0,01, Adam lr 1e-3, weight decay 1e-4, lote 128,
  ReduceLROnPlateau (0,5; paciência 5), parada antecipada (paciência 10), máx. 500 épocas
- normalização z-score com estatísticas do ajuste
- conjunto de N_SEEDS redes; a correção usada é a média
- avaliação: integração RK4 das janelas de teste (NN avaliada nos estados intermediários)
"""
import os, sys, numpy as np, pandas as pd, torch
import fit_dry_model as F
import fit_baseline as BL

torch.set_num_threads(int(os.environ.get('NN_THREADS', 2)))
DATA = F.DATA
N_SEEDS = int(os.environ.get('N_SEEDS', 5))
VARIANTS = {'NN(T,R)': ['T', 'R'], 'NN(T,R,q)': ['T', 'R', 'q']}


def windows_q(df):
    """Como F.windows, mas devolve (T, R, q, t0) e exige q válido."""
    d = df.copy()
    d['ok_TR'] = d['ok_TRq']
    ws = F.windows(d)
    out = []
    for T, R, t0 in ws:
        idx = d.index.get_loc(t0)
        q = d['q'].values[idx: idx + len(T)]
        out.append((T, R, q, t0))
    return out


def samples(ws, base):
    a, tau, Teq = base
    X, y = [], []
    for T, R, q, _ in ws:
        dT = (T[2:] - T[:-2]) / (2 * F.DT)
        Rm = 0.5 * (R[1:-1] + R[2:]); qm = 0.5 * (q[1:-1] + q[2:])
        Ti = T[1:-1]
        y.append(dT - (a * Rm - (Ti - Teq) / tau))
        X.append(np.c_[Ti, Rm, qm])
    return np.vstack(X), np.concatenate(y)


class MLP(torch.nn.Module):
    def __init__(self, nin):
        super().__init__()
        self.f = torch.nn.Sequential(torch.nn.Linear(nin, 32), torch.nn.LeakyReLU(0.01),
                                     torch.nn.Linear(32, 16), torch.nn.LeakyReLU(0.01),
                                     torch.nn.Linear(16, 1))

    def forward(self, x):
        return self.f(x)


def train(Xtr, ytr, Xva, yva, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    mx, sx = Xtr.mean(0), Xtr.std(0); my, sy = ytr.mean(), ytr.std()
    tX = lambda X: torch.tensor((X - mx) / sx, dtype=torch.float32)
    xt, yt = tX(Xtr), torch.tensor((ytr - my) / sy, dtype=torch.float32)[:, None]
    xv, yv = tX(Xva), torch.tensor((yva - my) / sy, dtype=torch.float32)[:, None]
    net = MLP(Xtr.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    best, best_state, wait, ep = np.inf, None, 0, 0
    for ep in range(500):
        net.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 128):
            b = perm[i:i + 128]
            opt.zero_grad(); l = torch.mean((net(xt[b]) - yt[b]) ** 2); l.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            lv = torch.mean((net(xv) - yv) ** 2).item()
        sch.step(lv)
        if lv < best - 1e-6:
            best, best_state, wait = lv, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= 10:
                break
    net.load_state_dict(best_state)
    net.eval()

    def predict(X):
        with torch.no_grad():
            return net(tX(X)).numpy()[:, 0] * sy + my
    return predict, ep + 1, best


def simulate_corr(base, nets, cols, T, R, Q, M):
    a, tau, Teq = base
    n, L = T.shape
    S = np.empty((n, L)); S[:, 0] = T[:, 0]
    sel = [{'T': 0, 'R': 1, 'q': 2}[c] for c in cols]

    def f(y, r, q):
        X = np.c_[y, r, q][:, sel]
        corr = np.mean([p(X) for p in nets], axis=0) if nets else 0.0
        return a * r - (y - Teq) / tau + corr
    for j in range(1, L):
        x, r, q = S[:, j - 1], R[:, j], Q[:, j]
        k1 = f(x, r, q); k2 = f(x + 0.5 * F.DT * k1, r, q); k3 = f(x + 0.5 * F.DT * k2, r, q)
        k4 = f(x + F.DT * k3, r, q)
        S[:, j] = np.clip(x + F.DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4), 250, 330)
    m = M.copy(); m[:, 0] = False
    e = (S - T)[m]
    return np.sqrt(np.mean(e ** 2)), np.mean(e)


def packq(ws):
    T, R, M = F.pack([(w[0], w[1], w[3]) for w in ws], 'atual')
    Q = np.zeros_like(T)
    for i, w in enumerate(ws):
        Q[i, :len(w[2])] = w[2]
    return T, R, Q, M


def main():
    rows = []
    for site in BL.SITES:
        if F.skip(site):
            continue
        df = pd.read_csv(os.path.join(DATA, f'{site}_30min.csv'), index_col=0, parse_dates=True)
        for est in df['estacao'].dropna().unique():
            ws = windows_q(df[df['estacao'] == est])
            if sum(len(w[0]) for w in ws) < 10 * 48:
                continue
            tr, va, te = F.split(ws)
            # linha de base reajustada no mesmo subconjunto (exige q válido)
            Ttr, Rtr, Qtr, Mtr = packq(tr)
            p = BL.fit(Ttr, Rtr, Mtr)
            base = (np.exp(p[0]), np.exp(p[1]), p[2])
            Tte, Rte, Qte, Mte = packq(te)
            r0, b0 = simulate_corr(base, [], ['T', 'R'], Tte, Rte, Qte, Mte)
            row = dict(sitio=site, estacao=est, dias_ajuste=round(Mtr.sum() / 48, 1),
                       dias_validacao=round(sum(len(w[0]) for w in va) / 48, 1),
                       dias_teste=round(Mte.sum() / 48, 1),
                       a=base[0], tau_h=base[1], Teq_C=base[2] - 273.15,
                       rmse_teste_base_K=r0, vies_teste_base_K=b0)
            Xtr, ytr = samples(tr, base); Xva, yva = samples(va, base); Xte, yte = samples(te, base)
            row['var_alvo_teste'] = np.var(yte)
            for name, cols in VARIANTS.items():
                sel = [{'T': 0, 'R': 1, 'q': 2}[c] for c in cols]
                nets, eps = [], []
                for s in range(N_SEEDS):
                    pr, ep, _ = train(Xtr[:, sel], ytr, Xva[:, sel], yva, seed=s)
                    nets.append(pr); eps.append(ep)
                yhat = np.mean([pr(Xte[:, sel]) for pr in nets], axis=0)
                r, b = simulate_corr(base, nets, cols, Tte, Rte, Qte, Mte)
                row[f'R2_derivada_teste {name}'] = 1 - np.mean((yte - yhat) ** 2) / np.var(yte)
                row[f'rmse_teste_K {name}'] = r
                row[f'vies_teste_K {name}'] = b
                row[f'epocas_medias {name}'] = np.mean(eps)
                print(site, est, name, f'RMSE {r0:.2f} -> {r:.2f} K, viés {b:.2f}', flush=True)
            rows.append(row)
            F.save_merged(rows, os.path.join(DATA, 'correcao_nn_resultados.csv'), '%.5g')


if __name__ == '__main__':
    main()
