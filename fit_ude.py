"""Revisão (parecer dos colegas): correção neural treinada na trajetória, como equação diferencial universal
    dT/dt = a*R - (T - Teq)/tau + f_theta(T, R)
Mesmo subconjunto (janelas com q válido), divisão 70/15/15, linha de base, arquitetura [32, 16] e padronização
das entradas de fit_nn_correction.py. Muda só a perda: em vez do resíduo da derivada, o erro quadrático médio da
trajetória integrada por RK4 (passo de 30 min, R do próprio intervalo) em cada janela de ajuste, com o gradiente
obtido por diferenciação automática através do integrador (Rackauckas et al., 2020).
- camada de saída iniciada em zero: o treino parte da linha de base
- correção limitada a +-2 desvios-padrão do resíduo da derivada (tangente hiperbólica): sem o limite, uma das
  redes do Cuieiras - seca divergiu no teste (RMSE de 16 K), ao atingir combinações de T alta e R baixo que o
  ajuste não visitou
- Adam (lr 3e-3, decaimento de pesos 1e-4); lote = todas as janelas de ajuste
- a cada 5 iterações, erro da trajetória nas janelas de validação; taxa à metade após 10 avaliações sem
  melhora, parada após 20, máximo de MAX_IT iterações; fica o estado de menor erro de validação
- conjunto de N_SEEDS redes; a correção usada é a média
- bootstrap em blocos (B_UDE membros, uma semente cada, base reajustada em cada membro): fração dos membros
  com RMSE de teste menor que o da base do mesmo membro
As redes de um conjunto (sementes ou membros do bootstrap) são treinadas ao mesmo tempo, em lote: cada uma tem a
própria linha de base, escala, janelas, grupo do Adam, redução da taxa e parada; a perda do lote é a soma das
perdas, então o gradiente de cada rede é o da sua própria perda (equivale a treiná-las uma a uma).
Saídas: dados/ude_resultados.csv e dados/ude_modelos/ (pesos e escalas, para as simulações contínuas e a
Figura 3)."""
import os, numpy as np, pandas as pd, torch
from multiprocessing import Pool
import fit_dry_model as F
import fit_baseline as BL
import fit_nn_correction as NN

DATA = F.DATA
MODEL_DIR = os.path.join(DATA, 'ude_modelos')
N_SEEDS = int(os.environ.get('N_SEEDS', 5))
B_UDE = int(os.environ.get('B_UDE', 20))
NPROC = int(os.environ.get('NPROC', 4))
MAX_IT = int(os.environ.get('MAX_IT', 1000))
LR = 3e-3
EVAL_EVERY, PAT_LR, PAT_STOP = 5, 10, 20
LIM = 2.0          # limite da correção, em desvios-padrão do resíduo da derivada do ajuste
ONLY = os.environ.get('ONLY_SETS')   # ex.: "jaru:seca,sinop:seca"
torch.set_default_dtype(torch.float64)


class UDE(torch.nn.Module):
    """Correção f_theta(T, R) em K/h, com entradas padronizadas e saída na escala do resíduo da derivada,
    limitada a +-LIM desvios-padrão por uma tangente hiperbólica: com a correção limitada, a relaxação linear
    domina para desvios grandes e a integração não diverge fora da região visitada no ajuste."""

    def __init__(self, mx, sx, sy):
        super().__init__()
        self.net = NN.MLP(2)
        torch.nn.init.zeros_(self.net.f[-1].weight)
        torch.nn.init.zeros_(self.net.f[-1].bias)
        self.register_buffer('mx', torch.as_tensor(mx, dtype=torch.float64))
        self.register_buffer('sx', torch.as_tensor(sx, dtype=torch.float64))
        self.sy = float(sy)

    def forward(self, y, r):
        x = (torch.stack([y, r], -1) - self.mx) / self.sx
        return LIM * self.sy * torch.tanh(self.net(x)[..., 0] / LIM)


def integrate(base, models, T0, R, clip=False):
    """RK4 de todas as janelas ao mesmo tempo; models = lista de UDE (correção média) ou []."""
    a, tau, Teq = base

    def rhs(y, r):
        f = a * r - (y - Teq) / tau
        if models:
            f = f + torch.stack([m(y, r) for m in models]).mean(0)
        return f
    x, out = T0, [T0]
    for j in range(1, R.shape[1]):
        r = R[:, j]
        k1 = rhs(x, r); k2 = rhs(x + 0.5 * F.DT * k1, r); k3 = rhs(x + 0.5 * F.DT * k2, r)
        k4 = rhs(x + F.DT * k3, r)
        x = x + F.DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if clip:
            x = x.clamp(250.0, 330.0)
        out.append(x)
    return torch.stack(out, 1)


def tensors(ws):
    T, R, Q, M = NN.packq(ws)
    m = M.copy(); m[:, 0] = False
    return (torch.as_tensor(np.where(M, T, 0.0)), torch.as_tensor(R), torch.as_tensor(m))


# ---------- treino em lote (S redes independentes) ----------

def member_params(seed):
    """Pesos iniciais de uma rede, com a mesma inicialização de UDE() após torch.manual_seed(seed)."""
    torch.manual_seed(seed); np.random.seed(seed)
    m = UDE(np.zeros(2), np.ones(2), 1.0)
    return [p.detach().clone().requires_grad_(True) for p in m.net.parameters()]


def corr_batch(P, sc, y, r):
    """Correção das S redes; P = pesos empilhados (W1, b1, W2, b2, W3, b3), sc = (mx [S,2], sx [S,2], sy [S]),
    y e r: [S, n]."""
    W1, b1, W2, b2, W3, b3 = P
    mx, sx, sy = sc
    lk = lambda z: torch.nn.functional.leaky_relu(z, 0.01)
    x = (torch.stack([y, r], -1) - mx[:, None, :]) / sx[:, None, :]
    h = lk(torch.baddbmm(b1[:, None, :], x, W1.transpose(1, 2)))
    h = lk(torch.baddbmm(b2[:, None, :], h, W2.transpose(1, 2)))
    o = torch.baddbmm(b3[:, None, :], h, W3.transpose(1, 2))[..., 0]
    return LIM * sy[:, None] * torch.tanh(o / LIM)


def batch_losses(base, P, sc, T, R, m):
    """Erro quadrático médio da trajetória de cada membro; base = (a, tau, Teq), cada um [S, 1]; T, R, m: [S, n, L]."""
    a, tau, Teq = base
    rhs = lambda y, r: a * r - (y - Teq) / tau + corr_batch(P, sc, y, r)
    x, out = T[:, :, 0], [T[:, :, 0]]
    for j in range(1, R.shape[2]):
        r = R[:, :, j]
        k1 = rhs(x, r); k2 = rhs(x + 0.5 * F.DT * k1, r); k3 = rhs(x + 0.5 * F.DT * k2, r)
        k4 = rhs(x + F.DT * k3, r)
        x = x + F.DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(x)
    S = torch.stack(out, 2)
    return (((S - T) ** 2) * m).sum((1, 2)) / m.sum((1, 2))


def train_batch(bases, scales, data, va, seeds):
    """Treina len(seeds) redes ao mesmo tempo. bases: lista de (a, tau, Teq); scales: lista de (mx, sx, sy);
    data = (T, R, m) com [S, n, L]; va = (T, R, m) de validação, comum a todos. Devolve modelos UDE e iterações."""
    S = len(seeds)
    base = tuple(torch.tensor([[b[i]] for b in bases]) for i in range(3))
    sc = (torch.as_tensor(np.array([s[0] for s in scales])), torch.as_tensor(np.array([s[1] for s in scales])),
          torch.as_tensor(np.array([s[2] for s in scales], float)))
    T, R, m = data
    Tv, Rv, mv = (x.expand(S, *x.shape) for x in va)
    params = [member_params(s) for s in seeds]
    opt = torch.optim.Adam([{'params': p} for p in params], lr=LR, weight_decay=1e-4)
    stack = lambda: [torch.stack([params[s][k] for s in range(S)]) for k in range(6)]
    with torch.no_grad():
        best = batch_losses(base, stack(), sc, Tv, Rv, mv).numpy().copy()
    best_state = [[p.detach().clone() for p in ps] for ps in params]
    wait, sch_bad = np.zeros(S, int), np.zeros(S, int)
    sch_best = np.full(S, np.inf)          # ReduceLROnPlateau(factor=0.5, patience=PAT_LR), limiar relativo 1e-4
    stopped, its = np.zeros(S, bool), np.full(S, MAX_IT)
    for it in range(1, MAX_IT + 1):
        opt.zero_grad()
        L = batch_losses(base, stack(), sc, T, R, m)
        fin = torch.isfinite(L)
        torch.where(fin, L, torch.zeros_like(L)).sum().backward()
        bad = ~fin.numpy()
        for s in np.nonzero(bad | stopped)[0]:
            for p in params[s]:
                p.grad = None
        opt.step()
        for s in np.nonzero(bad & ~stopped)[0]:      # trajetória não finita: volta ao melhor estado, taxa à metade
            with torch.no_grad():
                for p, q in zip(params[s], best_state[s]):
                    p.copy_(q)
            opt.param_groups[s]['lr'] *= 0.5
        if it % EVAL_EVERY == 0:
            with torch.no_grad():
                lv = batch_losses(base, stack(), sc, Tv, Rv, mv).numpy()
            for s in np.nonzero(~stopped)[0]:
                if lv[s] < sch_best[s] * (1 - 1e-4):
                    sch_best[s], sch_bad[s] = lv[s], 0
                else:
                    sch_bad[s] += 1
                    if sch_bad[s] > PAT_LR:
                        opt.param_groups[s]['lr'] *= 0.5; sch_bad[s] = 0
                if np.isfinite(lv[s]) and lv[s] < best[s] - 1e-7:
                    best[s], wait[s] = lv[s], 0
                    best_state[s] = [p.detach().clone() for p in params[s]]
                else:
                    wait[s] += 1
                    if wait[s] >= PAT_STOP:
                        stopped[s], its[s] = True, it
            if stopped.all():
                break
    models = []
    for s in range(S):
        md = UDE(*scales[s])
        with torch.no_grad():
            for p, q in zip(md.net.parameters(), best_state[s]):
                p.copy_(q)
        md.eval(); models.append(md)
    return models, its


def run(job):
    site, est = job
    torch.set_num_threads(1)
    df = pd.read_csv(os.path.join(DATA, f'{site}_30min.csv'), index_col=0, parse_dates=True)
    ws = NN.windows_q(df[df['estacao'] == est])
    tr, va, te = F.split(ws)
    Ttr_np, Rtr_np, _, Mtr_np = NN.packq(tr)
    p = BL.fit(Ttr_np, Rtr_np, Mtr_np)
    base = (float(np.exp(p[0])), float(np.exp(p[1])), float(p[2]))
    Xtr, ytr = NN.samples(tr, base)
    Xte, yte = NN.samples(te, base)
    scale = (Xtr[:, :2].mean(0), Xtr[:, :2].std(0), ytr.std())
    TR, VA, TE = tensors(tr), tensors(va), tensors(te)
    with torch.no_grad():
        S0 = integrate(base, [], TE[0][:, 0], TE[1])
    e0 = (S0 - TE[0])[TE[2]].numpy()
    row = dict(sitio=site, estacao=est, a=base[0], tau_h=base[1], Teq_C=base[2] - 273.15,
               rmse_teste_base_K=np.sqrt(np.mean(e0 ** 2)), vies_teste_base_K=np.mean(e0))
    os.makedirs(MODEL_DIR, exist_ok=True)
    models, its = train_batch([base] * N_SEEDS, [scale] * N_SEEDS,
                              tuple(x.expand(N_SEEDS, *x.shape) for x in TR), VA, list(range(N_SEEDS)))
    rm_seed = []
    for s, md in enumerate(models):
        with torch.no_grad():
            S = integrate(base, [md], TE[0][:, 0], TE[1], clip=True)
        rm_seed.append(float(np.sqrt(np.mean(((S - TE[0])[TE[2]].numpy()) ** 2))))
        torch.save(dict(state=md.state_dict(), scale=scale, base=base, iteracoes=int(its[s])),
                   os.path.join(MODEL_DIR, f'{site}_{est}_semente{s}.pt'))
        print(site, est, 'semente', s, f'{its[s]} iterações, RMSE de teste {rm_seed[-1]:.3f} K', flush=True)
    with torch.no_grad():
        S = integrate(base, models, TE[0][:, 0], TE[1], clip=True)
        e = (S - TE[0])[TE[2]].numpy()
        corr = torch.stack([m(torch.as_tensor(Xte[:, 0]), torch.as_tensor(Xte[:, 1])) for m in models]).mean(0)
    row.update(rmse_teste_ude_K=np.sqrt(np.mean(e ** 2)), vies_teste_ude_K=np.mean(e),
               rmse_sementes_min_K=min(rm_seed), rmse_sementes_max_K=max(rm_seed),
               iteracoes_medias=np.mean(its),
               R2_derivada_teste_ude=1 - np.mean((yte - corr.numpy()) ** 2) / np.var(yte))
    print(site, est, f"conjunto: RMSE de teste {row['rmse_teste_ude_K']:.3f} K "
          f"(base {row['rmse_teste_base_K']:.3f} K)", flush=True)
    # bootstrap em blocos: cada membro com janelas reamostradas, base reajustada e uma semente
    if B_UDE:
        rng = np.random.default_rng(2026)
        ks = [rng.integers(0, len(tr), len(tr)) for _ in range(B_UDE)]
        bb, sb = [], []
        for k in ks:
            trb = [tr[i] for i in k]
            Tb, Rb, _, Mb = NN.packq(trb)
            pb = BL.fit(Tb, Rb, Mb, p0=p)
            bb.append((float(np.exp(pb[0])), float(np.exp(pb[1])), float(pb[2])))
            Xb, yb = NN.samples(trb, bb[-1])
            sb.append((Xb[:, :2].mean(0), Xb[:, :2].std(0), yb.std()))
        data = tuple(torch.stack([x[torch.as_tensor(k)] for k in ks]) for x in TR)
        mbs, _ = train_batch(bb, sb, data, VA, [1000 + b for b in range(B_UDE)])
        better, ru = [], []
        for b, mb in enumerate(mbs):
            with torch.no_grad():
                Sb0 = integrate(bb[b], [], TE[0][:, 0], TE[1])
                Sb = integrate(bb[b], [mb], TE[0][:, 0], TE[1], clip=True)
            r0 = np.sqrt(np.mean(((Sb0 - TE[0])[TE[2]].numpy()) ** 2))
            r1 = np.sqrt(np.mean(((Sb - TE[0])[TE[2]].numpy()) ** 2))
            ru.append(r1); better.append(r1 < r0)
            print(site, est, 'bootstrap', b + 1, f'base {r0:.3f} UDE {r1:.3f}', flush=True)
        row.update(bootstrap_membros=B_UDE, frac_membros_ude_melhor=np.mean(better),
                   rmse_membros_ude_p05_K=np.percentile(ru, 5), rmse_membros_ude_p95_K=np.percentile(ru, 95))
    print(row, flush=True)
    return row


def main():
    par = pd.read_csv(os.path.join(DATA, 'linha_base_parametros.csv'))
    jobs = [tuple(x) for x in par.loc[par['forcante'] == 'Rn', ['sitio', 'estacao']].values]
    if ONLY:
        jobs = [j for j in jobs if f'{j[0]}:{j[1]}' in ONLY.split(',')]
    with Pool(NPROC) as pool:
        rows = pool.map(run, jobs, chunksize=1)
    out = os.path.join(DATA, os.environ.get('UDE_OUT', 'ude_resultados.csv'))
    pd.DataFrame(rows).to_csv(out, index=False, float_format='%.5g')


if __name__ == '__main__':
    main()
