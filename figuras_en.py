"""Figuras do manuscrito do Artigo 2 em inglês (Theoretical and Applied Climatology).
Uso: python3 figuras_en.py [nome ...]
Nomes: mapa, series, estacoes, mensal, transferencia, artigo1, alturas. Sem argumentos, gera todas.
Saída: ../figures/FigN.tif (600 dpi, RGB, LZW), FigN.pdf (vetorial, fontes TrueType
embutidas) e FigN.png (600 dpi, para o Word), com mapa→Fig1, artigo1→Fig2, series→Fig3, estacoes→Fig4,
mensal→Fig5, alturas→Fig6, transferencia→Fig7. A pasta de saída pode ser trocada pela variável FIGURAS_EN.
Mesmos dados, cálculos, leiautes e cores de figuras.py (versão em português); muda só a apresentação: textos em
inglês, ponto decimal, Liberation Sans (métrica do Arial) também no mathtext, largura de 17,4 cm com as alturas na
mesma proporção, letras de 8 a 12 pt (7 pt só nos números das células do mapa de calor e nos rótulos das isolinhas)
e linhas de pelo menos 0,3 pt (as de 0,3 pt passaram a 0,4 pt, por margem).
Ajustes para as letras maiores não se sobreporem: Fig1, escala 0,35° a oeste e fundo branco em 'Manaus'; Fig2,
estimativa inicial em duas linhas e em outra posição, sem o rótulo de 3000 h no eixo de cima; Fig4, painel do Mujuí
com a largura dos demais, limites de x simétricos e legendas no canto superior direito; Fig7, títulos encurtados.
Camadas do mapa (pasta GEO): Natural Earth 1:50m (países, estados) e 1:10m (rios), em GeoJSON do
repositório nvkelso/natural-earth-vector; biomas (IBGE, 2019) e Amazônia Legal pelo pacote geobr (IPEA)."""
import io, os, sys, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from PIL import Image

DATA = os.path.normpath(os.environ.get('DATA_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')))
OUT = os.environ.get('FIGURAS_EN', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures'))
GEO = os.environ.get('GEO', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'geo'))
CM = 1 / 2.54
W = 17.4 * CM          # largura de página inteira da TAAC (174 mm; altura máxima de 234 mm)
ESC = 17.4 / 16        # as alturas da versão em português (W = 16 cm) crescem no mesmo fator
DPI = 600
plt.rcParams.update({
    'font.family': 'Liberation Sans', 'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9,
    'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'lines.linewidth': 1.1,
    'mathtext.fontset': 'custom', 'mathtext.rm': 'Liberation Sans', 'mathtext.it': 'Liberation Sans:italic',
    'mathtext.bf': 'Liberation Sans:bold', 'mathtext.sf': 'Liberation Sans', 'mathtext.cal': 'Liberation Sans:italic',
    'mathtext.tt': 'Liberation Mono', 'mathtext.fallback': 'stixsans', 'savefig.dpi': DPI,
    'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False, 'axes.formatter.use_locale': False})
# Okabe-Ito: uma cor e um marcador por sítio, iguais em todas as figuras
SITE = {'jaru': ('#0072B2', 's', 'Rebio Jaru'), 'sinop': ('#E69F00', '^', 'Sinop'),
        'cuieiras': ('#009E73', 'o', 'Cuieiras (K34)'), 'mujui': ('#CC79A7', 'D', 'Mujuí')}
COORD = {'jaru': (-61.8750, -10.1865), 'sinop': (-55.3250, -11.4125),
         'cuieiras': (-60.2093, -2.6091), 'mujui': (-64.9175, -2.8592)}
EST = {'chuvosa': 'Wet', 'transicao_cs': 'Wet-to-dry', 'seca': 'Dry', 'transicao_sc': 'Dry-to-wet',
       'terrestre': 'Terrestrial', 'cheia': 'Flooded', 'pos_cheia': 'Post-flood', 'inundada': 'Flooded'}
EST_CURTO = {'chuvosa': 'Wet', 'transicao_cs': 'W→D', 'seca': 'Dry', 'transicao_sc': 'D→W',
             'terrestre': 'Terr.', 'cheia': 'Flooded', 'pos_cheia': 'Post-\nflood'}
CONJ = [('jaru', 'chuvosa'), ('jaru', 'seca'), ('cuieiras', 'chuvosa'), ('cuieiras', 'seca'),
        ('sinop', 'seca'), ('mujui', 'terrestre'), ('mujui', 'inundada')]
MESES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def rot(site, est, curto=False):
    n = {'jaru': 'Jaru', 'sinop': 'Sinop', 'cuieiras': 'Cuieiras', 'mujui': 'Mujuí'}[site]
    e = EST[est].replace('\n', ' ').replace('- ', '')
    return f'{n} {e[0]}' if curto else f'{n} – {e.lower()}'


def f(v, nd=2):
    """Número com ponto decimal e sinal de menos tipográfico."""
    return f'{v:.{nd}f}'.replace('-', '−')


def data(t):
    """Data no formato '23 Mar 2008'."""
    return f'{t.day} {MESES[t.month - 1]} {t.year}'


def salvar(fig, nome):
    """FigN.pdf (vetorial), FigN.tif (RGB, LZW) e FigN.png, os dois a 600 dpi e com os mesmos pixels. O TIFF sai
    pelo PIL porque o matplotlib grava TIFF sempre em RGBA; o fundo branco é aplicado antes de tirar o canal alfa."""
    os.makedirs(OUT, exist_ok=True)
    base = os.path.join(OUT, nome)
    fig.savefig(base + '.pdf', dpi=DPI, facecolor='white')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=DPI, facecolor='white')
    plt.close(fig)
    buf.seek(0)
    with Image.open(buf) as im:
        im = im.convert('RGBA')
        rgb = Image.new('RGB', im.size, 'white')
        rgb.paste(im, mask=im.getchannel('A'))
    rgb.save(base + '.tif', compression='tiff_lzw', dpi=(DPI, DPI))
    rgb.save(base + '.png', dpi=(DPI, DPI), optimize=True)
    print('ok', nome, rgb.size, flush=True)


def graus(v, eixo):
    h = ('S' if v < 0 else 'N') if eixo == 'y' else ('W' if v < 0 else 'E')
    return f'{abs(v):.0f}°{h}' if v != 0 else '0°'


# ------------------------------------------------------------------ mapa
def mapa():
    import geopandas as gpd
    from shapely.geometry import box
    paises = gpd.read_file(os.path.join(GEO, 'ne_50m_admin_0_countries.geojson'))
    uf = gpd.read_file(os.path.join(GEO, 'ne_50m_admin_1_states_provinces.geojson'))
    uf = uf[uf['adm0_a3'] == 'BRA']
    rios = gpd.read_file(os.path.join(GEO, 'ne_10m_rivers_lake_centerlines.geojson'))
    bio = gpd.read_file(os.path.join(GEO, 'biomes_2019_simplified.gpkg')).to_crs(4326)
    al = gpd.read_file(os.path.join(GEO, 'amazonia_legal_simplified.gpkg')).to_crs(4326)
    am = paises[paises['CONTINENT'] == 'South America']
    ext = (-68.6, -52.4, -14.4, 0.8)
    fig = plt.figure(figsize=(W, 10.6 * ESC * CM), layout='constrained')
    gs = fig.add_gridspec(1, 2, width_ratios=[0.62, 1.38])
    ax0, ax = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    # (a) América do Sul
    am.plot(ax=ax0, color='#f2f2f2', edgecolor='#9a9a9a', linewidth=0.4)
    am[am['ADM0_A3'] == 'BRA'].plot(ax=ax0, color='#e3e3e3', edgecolor='#555555', linewidth=0.5)
    al.boundary.plot(ax=ax0, color='#2b7a3d', linewidth=0.8)
    ax0.add_patch(Rectangle((ext[0], ext[2]), ext[1] - ext[0], ext[3] - ext[2], fill=False, ec='k', lw=0.9))
    ax0.set_xlim(-82, -34); ax0.set_ylim(-56, 13)
    ax0.set_aspect(1 / np.cos(np.deg2rad(20)))
    ax0.set_xticks([-75, -60, -45]); ax0.set_yticks([-45, -30, -15, 0])
    ax0.set_xticklabels([graus(v, 'x') for v in [-75, -60, -45]])
    ax0.set_yticklabels([graus(v, 'y') for v in [-45, -30, -15, 0]])
    ax0.set_title('(a)', loc='left')
    ax0.set_xlabel(''); ax0.set_ylabel('')
    for s in ['top', 'right']:
        ax0.spines[s].set_visible(True)
    # (b) região dos sítios
    win = box(ext[0] - 1, ext[2] - 1, ext[1] + 1, ext[3] + 1)
    cores = {'Amazônia': '#d9ead3', 'Cerrado': '#f6e7c1'}      # chaves = nomes na camada do IBGE
    bioma_en = {'Amazônia': 'Amazon', 'Cerrado': 'Cerrado'}
    paises.clip(win).plot(ax=ax, color='#f7f7f7', edgecolor='#9a9a9a', linewidth=0.4)
    for nm, c in cores.items():
        b = bio[bio['name_biome'] == nm]
        if len(b):
            b.clip(win).plot(ax=ax, color=c, edgecolor='none')
    uf.clip(win).boundary.plot(ax=ax, color='#7f7f7f', linewidth=0.4)
    r = rios.cx[ext[0]:ext[1], ext[2]:ext[3]]
    r = r[r['scalerank'] <= 6]
    r.plot(ax=ax, color='#4a90c2', linewidth=0.55)
    rotulos_uf = {'AM': (-64.6, -5.6), 'PA': (-54.0, -4.6), 'RO': (-63.0, -11.9), 'MT': (-55.0, -12.7),
                  'AC': (-68.2, -9.4), 'RR': (-61.8, 0.1)}
    for k, (x, y) in rotulos_uf.items():
        ax.text(x, y, k, fontsize=8, color='#555555', ha='center', va='center')
    nomes = {'jaru': ('Rebio Jaru (RO)', (0.35, 0.35), 'left'), 'sinop': ('Sinop (MT)', (0.35, 0.40), 'left'),
             'cuieiras': ('Cuieiras – K34 (AM)', (0.35, 0.40), 'left'),
             'mujui': ('Mujuí – Mamirauá SDR (AM)', (-1.6, -1.15), 'left')}
    for s, (x, y) in COORD.items():
        c, mk, _ = SITE[s]
        ax.plot(x, y, mk, color=c, mec='k', mew=0.6, ms=7, zorder=5)
        t, (dx, dy), ha = nomes[s]
        ax.text(x + dx, y + dy, t, fontsize=8, ha=ha, va='bottom', zorder=6,
                bbox=dict(fc='white', ec='none', alpha=0.75, pad=0.6))
    ax.text(-58.1, -3.55, 'Manaus', fontsize=8, color='#333333', ha='left',
            bbox=dict(fc='white', ec='none', alpha=0.75, pad=0.6))
    ax.plot(-60.02, -3.10, 'o', ms=2.5, color='#333333')
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_aspect(1 / np.cos(np.deg2rad(-6.5)))
    xt, yt = [-66, -62, -58, -54], [-12, -8, -4, 0]
    ax.set_xticks(xt); ax.set_yticks(yt)
    ax.set_xticklabels([graus(v, 'x') for v in xt]); ax.set_yticklabels([graus(v, 'y') for v in yt])
    for s in ['top', 'right']:
        ax.spines[s].set_visible(True)
    # escala de 200 km e norte
    x0, y0 = -68.25, -13.7
    dlon = 200 / (111.32 * np.cos(np.deg2rad(y0)))
    ax.plot([x0, x0 + dlon], [y0, y0], color='k', lw=2.2, solid_capstyle='butt')
    ax.plot([x0, x0 + dlon / 2], [y0, y0], color='white', lw=1.2, solid_capstyle='butt')
    ax.text(x0, y0 + 0.25, '0', fontsize=8, ha='center'); ax.text(x0 + dlon, y0 + 0.25, '200 km', fontsize=8, ha='center')
    ax.annotate('N', xy=(-53.2, -0.2), xytext=(-53.2, -1.6), ha='center', va='center', fontsize=8, fontweight='bold',
                arrowprops=dict(arrowstyle='-|>', color='k', lw=0.9))
    ax.set_title('(b)', loc='left')
    ax.set_xlabel(''); ax.set_ylabel('')
    fig.legend(handles=[Line2D([], [], color='#2b7a3d', lw=0.8, label='Legal Amazon'),
                        Patch(fc='#e3e3e3', ec='#555555', lw=0.5, label='Brazil')]
               + [Patch(fc=c, ec='#999999', lw=0.4, label=f'{bioma_en[nm]} biome (IBGE 2019)') for nm, c in cores.items()]
               + [Line2D([], [], color='#4a90c2', lw=0.8, label='Rivers'),
                  Line2D([], [], color='#7f7f7f', lw=0.5, label='State boundaries')],
               loc='outside lower center', ncol=3, fontsize=8, handlelength=1.6, columnspacing=1.2)
    salvar(fig, 'Fig1')


# ------------------------------------------------------------------ séries de teste
def series():
    """Grade 3 x 3: sete painéis e a legenda nas duas últimas casas. O RMSE vai no título, fora da área dos
    dados, para não cobrir as curvas."""
    s = pd.read_csv(os.path.join(DATA, 'series_teste.csv'), parse_dates=['fim_intervalo'])
    j = pd.read_csv(os.path.join(DATA, 'series_teste_janelas.csv'))
    fig, axs = plt.subplots(3, 3, figsize=(W, 10.8 * ESC * CM), layout='constrained')
    fig.get_layout_engine().set(h_pad=0.02, w_pad=0.02, hspace=0.04, wspace=0.03)
    axs = axs.ravel()
    letras = 'abcdefg'
    for k, (site, est) in enumerate(CONJ):
        ax = axs[k]
        jj = j[(j['sitio'] == site) & (j['estacao'] == est)]
        jj = jj[jj['pontos'] >= min(192, jj['pontos'].max())].sort_values('rmse_T_base')   # janelas de 4 a 5 dias
        jan = jj.iloc[(len(jj) - 1) // 2]['janela']          # janela de RMSE mediano da base
        d = s[(s['sitio'] == site) & (s['estacao'] == est) & (s['janela'] == jan)]
        t = (d['fim_intervalo'] - d['fim_intervalo'].iloc[0]).dt.total_seconds() / 86400
        C = lambda v: v - 273.15
        ax.fill_between(t, C(d['base_p05']), C(d['base_p95']), color='#56B4E9', alpha=0.25, lw=0)
        ax.plot(t, C(d['T_obs']), color='k', lw=1.0)
        ax.plot(t, C(d['T_seco_A1']), color='#7f7f7f', lw=0.9, ls=(0, (1.2, 1.2)))
        ax.plot(t, C(d['T_base']), color='#0072B2', lw=0.9)
        ax.plot(t, C(d['T_base_UDE']), color='#D55E00', lw=0.9, ls=(0, (4, 1.5)))
        r = jj[jj['janela'] == jan].iloc[0]
        ini = d['fim_intervalo'].iloc[0] - pd.Timedelta('30min')
        ax.set_title(f'({letras[k]}) {rot(site, est)}, {data(ini)}\n'
                     f"RMSE (K): {f(r['rmse_T_base'])}; {f(r['rmse_T_base_UDE'])}; {f(r['rmse_T_seco_A1'])}",
                     loc='left', fontsize=8, linespacing=1.2, pad=3)
        ax.set_xlim(0, 5); ax.set_xticks(range(6))
        ax.tick_params(labelsize=8, length=2.5, pad=1.5)
        if k == 6:
            ax.set_xlabel('Days since window start', fontsize=8, labelpad=1)
        if k % 3 == 0:
            ax.set_ylabel('T (°C)', fontsize=8, labelpad=1)
    gs = axs[7].get_subplotspec().get_gridspec()
    axs[7].remove(); axs[8].remove()
    lax = fig.add_subplot(gs[2, 1:]); lax.axis('off')
    lax.legend(handles=[Line2D([], [], color='k', lw=1.0, label='Observed'),
                        Line2D([], [], color='#0072B2', lw=0.9, label='Reference model (Ref.)'),
                        Patch(fc='#56B4E9', alpha=0.25, label='90% prediction interval of Ref.'),
                        Line2D([], [], color='#D55E00', lw=0.9, ls=(0, (4, 1.5)),
                               label='Ref. + trajectory-trained NN'),
                        Line2D([], [], color='#7f7f7f', lw=0.9, ls=(0, (1.2, 1.2)),
                               label=r"Simplified radiative model ($\alpha^{*}R_n - \gamma^{*}T^{4}$), this study's protocol")],
               loc='center', fontsize=8, handlelength=2.6,
               title='RMSE in panel headers: Ref.; Ref. + NN; simplified radiative', title_fontsize=8)
    salvar(fig, 'Fig3')


# ------------------------------------------------------------------ ciclo diurno nas janelas de teste
MODELOS = [('T_seco_A1', 'Simplified radiative model (Eq. 1)', '#7f7f7f', (0, (1.2, 1.2))),
           ('T_base', 'Reference model (Ref., Eq. 4)', '#0072B2', '-'),
           ('T_base_UDE', 'Ref. + trajectory-trained NN (Eq. 6)', '#D55E00', (0, (4, 1.5)))]


def metricas(obs, sim):
    """RMSE, MAE e R2 (1 - SQE/SQT) nos pontos de teste."""
    e = sim - obs
    return np.sqrt(np.mean(e ** 2)), np.mean(np.abs(e)), 1 - np.sum(e ** 2) / np.sum((obs - obs.mean()) ** 2)


def ciclo():
    """Ciclo diurno médio nas janelas de teste: observado (IC de 95% da média entre dias) e os três modelos, com o
    resíduo médio (simulado - observado) embaixo e RMSE, MAE e R2 de todos os pontos de teste no quadro. O
    primeiro ponto de cada janela, igual à observação por construção, fica de fora, como no protocolo."""
    from scipy.stats import t as student
    s = pd.read_csv(os.path.join(DATA, 'series_teste.csv'), parse_dates=['fim_intervalo'])
    s = s[s.groupby(['sitio', 'estacao', 'janela']).cumcount() > 0].copy()
    assert s[['T_obs'] + [m for m, *_ in MODELOS]].gt(250).all().all()       # K, sem valores fisicamente absurdos
    meio = s['fim_intervalo'] - pd.Timedelta('15min')                          # rótulo no fim do intervalo
    s['hora'] = meio.dt.hour + meio.dt.minute / 60
    s['dia'] = meio.dt.normalize()
    fig = plt.figure(figsize=(W, 22.0 * CM), layout='constrained')
    fig.get_layout_engine().set(h_pad=0.03, w_pad=0.02, hspace=0.05, wspace=0.04)
    gs = fig.add_gridspec(4, 2)
    linhas = []
    for k, (site, est) in enumerate(CONJ):
        sub = gs[k // 2, k % 2].subgridspec(2, 1, height_ratios=[2.6, 1], hspace=0.06)
        ax, axr = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
        d = s[(s['sitio'] == site) & (s['estacao'] == est)]
        g = d.groupby('hora')
        n = g['dia'].nunique()
        mo, sd = g['T_obs'].mean() - 273.15, g['T_obs'].std()
        ic = student.ppf(0.975, n - 1) * sd / np.sqrt(n)
        h = mo.index.values
        ax.fill_between(h, mo - ic, mo + ic, color='0.75', lw=0)
        ax.plot(h, mo, color='k', lw=1.1)
        txt = []
        for col, _, cor, ls in MODELOS:
            ax.plot(h, g[col].mean() - 273.15, color=cor, lw=1.0, ls=ls)
            axr.plot(h, (g[col].mean() - g['T_obs'].mean()).values, color=cor, lw=1.0, ls=ls)
            rmse, mae, r2 = metricas(d['T_obs'].values, d[col].values)
            txt.append(f'{f(rmse)}  {f(mae)}  {f(r2)}')
            linhas.append(dict(sitio=site, estacao=est, modelo=col, pontos=len(d), dias=d['dia'].nunique(),
                               rmse_K=rmse, mae_K=mae, R2=r2))
        axr.axhline(0, color='k', lw=0.5)
        ax.text(0.02, 0.97, 'RMSE   MAE    R²\n' + '\n'.join(txt), transform=ax.transAxes, va='top', ha='left',
                fontsize=6.3, family='Liberation Mono', linespacing=1.15,
                bbox=dict(fc='white', ec='0.7', lw=0.4, pad=1.5, alpha=0.9))
        ax.set_title(f"({'abcdefg'[k]}) {rot(site, est)} ({d['dia'].nunique()} test days)", loc='left',
                     fontsize=8, pad=2)
        for a in (ax, axr):
            a.set_xlim(0, 24); a.set_xticks(range(0, 25, 6))
            a.tick_params(labelsize=7, length=2.5, pad=1.5)
        ax.tick_params(labelbottom=False)
        if k % 2 == 0:
            ax.set_ylabel('T (°C)', fontsize=8, labelpad=1)
            axr.set_ylabel('Sim. − obs.\n(K)', fontsize=7, labelpad=1)
        if k >= 5:
            axr.set_xlabel('Local time (h)', fontsize=8, labelpad=1)
    lax = fig.add_subplot(gs[3, 1]); lax.axis('off')
    lax.legend(handles=[Line2D([], [], color='k', lw=1.1, label='Observed, test-window mean'),
                        Patch(fc='0.75', label='95% confidence interval of the observed mean')] +
               [Line2D([], [], color=c, lw=1.0, ls=ls, label=lab) for _, lab, c, ls in MODELOS],
               loc='center', fontsize=7.5, handlelength=2.6,
               title='Box: RMSE (K), MAE (K) and R² of all test points,\none line per model in the order below',
               title_fontsize=7.5)
    pd.DataFrame(linhas).to_csv(os.path.join(DATA, 'ciclo_diurno_teste_metricas.csv'), index=False,
                                float_format='%.6g')
    salvar(fig, 'Fig_ciclo')


# ------------------------------------------------------------------ fluxos no Cuieiras
def fluxos():
    """tau de cada janela de 5 dias no Cuieiras contra a umidade relativa, a fração evaporativa e a temperatura
    médias da janela, com o rho de Spearman das janelas válidas (o mesmo da Tabela 6)."""
    from scipy.stats import spearmanr
    d = pd.read_csv(os.path.join(DATA, 'fluxos_cuieiras_janelas.csv'))
    d = d[d['valida']]
    assert d['UR_media'].between(0, 100).all() and d['tau_h'].gt(0).all()
    cores = {'chuvosa': ('#0072B2', 'o', 'Wet season'), 'transicao_cs': ('#56B4E9', 'v', 'Wet-to-dry'),
             'seca': ('#D55E00', 's', 'Dry season'), 'transicao_sc': ('#E69F00', '^', 'Dry-to-wet')}
    fig, axs = plt.subplots(1, 3, figsize=(W, 6.2 * CM), layout='constrained', sharey=True)
    for ax, (col, rot_x), letra in zip(axs, [('UR_media', 'Relative humidity (%)'),
                                              ('EF', 'Evaporative fraction, ΣLE/Σ(H + LE)'),
                                              ('T_media_C', 'Air temperature (°C)')], 'abc'):
        for est, (c, mk, lab) in cores.items():
            e = d[d['estacao'] == est]
            ax.scatter(e[col], e['tau_h'], s=14, color=c, marker=mk, lw=0, label=lab)
        rho, p = spearmanr(d[col], d['tau_h'])
        ax.set_title(f'({letra}) Spearman ρ = {f(rho)}, p = {p:.3f}'.replace('-', '−'), loc='left', fontsize=8)
        ax.set_xlabel(rot_x, fontsize=8)
        ax.tick_params(labelsize=7, length=2.5)
    axs[0].set_ylabel('τ (h)', fontsize=8)
    axs[0].set_yscale('log'); axs[0].set_yticks([1, 2, 5, 10, 20]); axs[0].set_yticklabels(['1', '2', '5', '10', '20'])
    axs[2].legend(loc='upper left', fontsize=7, handletextpad=0.2, borderaxespad=0.2)
    salvar(fig, 'Fig_fluxos')


# ------------------------------------------------------------------ tau de dia e de noite
def dia_noite():
    """(a) Jaru por ano e altura: tau diurno e noturno na chuvosa e na seca; (b) e (c) perfis do Cuieiras e de
    Sinop. Barras: IC de 90% (bootstrap em blocos). Ajustes em que tau_n não converge (sem decaimento noturno
    identificável, tau_n > 100 h) ficam fora e são citados na legenda."""
    d = pd.read_csv(os.path.join(DATA, 'tau_dia_noite.csv'))
    ok = d['tau_noite_h'] < 100
    cor = {'chuvosa': '#0072B2', 'seca': '#D55E00'}
    fig = plt.figure(figsize=(W, 13.0 * CM), layout='constrained')
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05])
    ax = fig.add_subplot(gs[0, :])
    j = d[(d['grupo'] == 'jaru_ano') & ok].copy()
    j['ano'] = j['ano'].astype(int)
    ordem = [('62m', 2004), ('62m', 2008), ('62m', 2010), ('62m', 2009), ('22m', 2005), ('22m', 2006),
             ('22m', 2008)]
    rot_x = []
    for i, (alt, ano) in enumerate(ordem):
        g = j[(j['altura'] == alt) & (j['ano'] == ano)]
        rot_x.append(f"{alt.replace('m', ' m')}\n{ano}" + ('*' if (g['obs'] == 'sensor').any() else ''))
        for est, dx in [('chuvosa', -0.12), ('seca', 0.12)]:
            r = g[g['estacao'] == est]
            if r.empty:
                continue
            r = r.iloc[0]
            for per, mfc, ddx in [('dia', 'white', -0.05), ('noite', cor[est], 0.05)]:
                v, lo, hi = r[f'tau_{per}_h'], r[f'tau_{per}_p05'], r[f'tau_{per}_p95']
                ax.errorbar(i + dx + ddx, v, yerr=[[v - lo], [hi - v]], fmt='o', ms=4.5, mfc=mfc, mec=cor[est],
                            ecolor=cor[est], elinewidth=0.8, capsize=1.5, lw=0)
    ax.axvline(3.5, color='0.6', lw=0.6, ls=':')
    ax.set_xticks(range(len(ordem))); ax.set_xticklabels(rot_x, fontsize=7.5)
    ax.set_ylabel('τ (h)', fontsize=8); ax.set_ylim(0, None)
    ax.set_title('(a) Jaru, by height and year (*: series with sensor problems)', loc='left', fontsize=8)
    ax.tick_params(labelsize=7.5, length=2.5)
    ax.legend(handles=[Line2D([], [], marker='o', ls='', mfc='white', mec='k', ms=4.5, label='Daytime τ (τ$_d$)'),
                       Line2D([], [], marker='o', ls='', mfc='k', mec='k', ms=4.5, label='Nighttime τ (τ$_n$)'),
                       Patch(fc=cor['chuvosa'], label='Wet season'), Patch(fc=cor['seca'], label='Dry season')],
              loc='upper left', fontsize=7, ncol=2, handletextpad=0.3, columnspacing=0.8)
    for k, (site, letra, nome) in enumerate([('cuieiras', 'b', 'Cuieiras'), ('sinop', 'c', 'Sinop')]):
        ax = fig.add_subplot(gs[1, k])
        g = d[(d['grupo'] == 'perfil') & (d['sitio'] == site) & ok]
        for est, ls in [('seca', '-'), ('chuvosa', '--')]:
            e = g[g['estacao'] == est].sort_values('z_sobre_h')
            if e.empty:
                continue
            for per, mfc in [('dia', 'white'), ('noite', cor[est])]:
                ax.errorbar(e[f'tau_{per}_h'], e['z_sobre_h'],
                            xerr=[e[f'tau_{per}_h'] - e[f'tau_{per}_p05'], e[f'tau_{per}_p95'] - e[f'tau_{per}_h']],
                            fmt='o', ms=4, mfc=mfc, mec=cor[est], ecolor=cor[est], elinewidth=0.7, capsize=1.5,
                            ls=ls, color=cor[est], lw=0.8)
        ax.axhspan(0, 1, color='#009E73', alpha=0.08, lw=0)
        ax.set_xscale('log'); ax.set_xticks([1, 2, 5, 10, 20]); ax.set_xticklabels(['1', '2', '5', '10', '20'])
        ax.set_xlim(1, 20); ax.set_ylim(0, 1.6)
        ax.set_xlabel('τ (h)', fontsize=8); ax.set_title(f'({letra}) {nome} profile', loc='left', fontsize=8)
        ax.tick_params(labelsize=7.5, length=2.5)
        if k == 0:
            ax.set_ylabel('z/h', fontsize=8)
            ax.legend(handles=[Line2D([], [], color='k', ls='-', label='Dry season'),
                               Line2D([], [], color='k', ls='--', label='Wet season')],
                      loc='lower right', fontsize=7)
    salvar(fig, 'Fig_dianoite')


# ------------------------------------------------------------------ estações x anos
def estacoes():
    e = pd.read_csv(os.path.join(DATA, 'estacoes_anos_parametros.csv'), dtype={'ano': str})
    e = e[e['tau_h'].notna()]
    # séries com defeito de sensor (T a 62 m em 2009 e a 22 m em 2008) ficam fora da análise
    e = e[~((e['sitio'] == 'jaru') & (((e['altura'] == '62m') & (e['ano'] == '2009')) |
                                      ((e['altura'] == '22m') & (e['ano'] == '2008'))))]
    paineis = [('Jaru, 62 m', (e['sitio'] == 'jaru') & (e['altura'] == '62m'), ['chuvosa', 'transicao_cs', 'seca', 'transicao_sc']),
               ('Jaru, 22 m', (e['sitio'] == 'jaru') & (e['altura'] == '22m'), ['chuvosa', 'transicao_cs', 'seca', 'transicao_sc']),
               ('Sinop, 40 m', e['sitio'] == 'sinop', ['chuvosa', 'transicao_cs', 'seca', 'transicao_sc']),
               ('Cuieiras, 51 m', e['sitio'] == 'cuieiras', ['chuvosa', 'transicao_cs', 'seca', 'transicao_sc']),
               ('Mujuí, 12 m', e['sitio'] == 'mujui', ['terrestre', 'cheia', 'pos_cheia'])]
    suspeitos = {('jaru', '62m', '2009'), ('jaru', '22m', '2008')}
    estilos = ['-', '--', '-.', ':']
    fig, axs = plt.subplots(2, 5, figsize=(W, 11 * ESC * CM), sharey='row', layout='constrained',
                            gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 1.0]))
    for c, (titulo, msk, cats) in enumerate(paineis):
        sub = e[msk]
        anos = sorted(sub['ano'].unique())
        for i, ano in enumerate(anos):
            g = sub[sub['ano'] == ano].set_index('estacao').reindex(cats)
            susp = (g['sitio'].dropna().iloc[0], g['altura'].dropna().iloc[0], ano) in suspeitos
            site = g['sitio'].dropna().iloc[0]
            col, mk, _ = SITE[site]
            col = '#9a9a9a' if susp else col
            n = len(anos)
            x = np.arange(len(cats)) + (i - (n - 1) / 2) * 0.12
            for r, (var, lo, hi) in enumerate([('tau_h', 'tau_h_p05', 'tau_h_p95'), ('Teq_C', 'Teq_C_p05', 'Teq_C_p95')]):
                ax = axs[r, c]
                y = g[var].values
                ok = np.isfinite(y)
                ax.errorbar(x[ok], y[ok], yerr=[y[ok] - g[lo].values[ok], g[hi].values[ok] - y[ok]], fmt=mk,
                            ms=4.2, mfc=col if not susp else 'white', mec=col, color=col, elinewidth=0.7, capsize=1.5,
                            ls='none', zorder=3)
                ax.plot(x, y, color=col, lw=0.8, ls=estilos[i % 4], zorder=2)
                if r == 0:
                    lab = f'{ano}' + (' *' if susp else '')
                    ax.plot([], [], color=col, ls=estilos[i % 4], marker=mk, ms=3.5, mfc=col if not susp else 'white',
                            lw=0.8, label=lab)
        for r in range(2):
            ax = axs[r, c]
            ax.set_xticks(range(len(cats)))
            ax.set_xticklabels([EST_CURTO[k] for k in cats] if r == 1 else [], fontsize=8)
            ax.set_xlim(-0.5, len(cats) - 0.5)        # simétrico: a legenda passou para a direita
            ax.grid(axis='y', color='#e5e5e5', lw=0.5)
        axs[0, c].set_title(titulo, fontsize=8.5)
        axs[0, c].legend(loc='upper right', fontsize=8, handlelength=2.2, borderaxespad=0.2)
    axs[0, 0].set_ylabel('τ (h)')
    axs[1, 0].set_ylabel('$T_{eq}$ (°C)')
    axs[0, 0].set_ylim(0, 19); axs[0, 0].set_yticks([0, 3, 6, 9, 12, 15, 18])
    salvar(fig, 'Fig4')


# ------------------------------------------------------------------ ciclo mensal e umidade
def mensal():
    from scipy.stats import spearmanr
    m = pd.read_csv(os.path.join(DATA, 'meses_parametros.csv'), dtype={'ano_estacao': str})
    m['mes_num'] = pd.PeriodIndex(m['mes'], freq='M').month
    m['ano'] = pd.PeriodIndex(m['mes'], freq='M').year
    susp = ((m['sitio'] == 'jaru') & (((m['altura'] == '62m') & (m['ano'] == 2009)) |
                                      ((m['altura'] == '22m') & (m['ano'] == 2008))))
    m = m[~susp]          # séries com defeito de sensor ficam fora da análise
    susp = susp[~susp]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 8.6 * ESC * CM), layout='constrained',
                                 gridspec_kw=dict(width_ratios=[1.35, 1]))
    est = ['-', '--', '-.', ':', (0, (3, 1, 1, 1, 1, 1))]
    jl = m[(m['sitio'] == 'jaru') & ~susp]
    for i, ((alt, ano), g) in enumerate(sorted(jl.groupby(['altura', 'ano']), key=lambda kv: (kv[0][0] != '62m', kv[0][1]))):
        g = g.set_index('mes_num').reindex(range(1, 13))
        lw = 1.2 if alt == '62m' else 0.8
        a1.plot(g.index, g['tau_h'], ls=est[i % 5], marker='s', ms=3, lw=lw, color='#0072B2' if alt == '62m' else '#56B4E9',
                label=f'Jaru {ano} ({alt.replace("m", " m")})')
    for site in ['sinop', 'cuieiras', 'mujui']:
        g = m[m['sitio'] == site]
        c, mk, nm = SITE[site]
        a1.plot(g['mes_num'], g['tau_h'], mk, color=c, ms=4, mec='k', mew=0.4, ls='none', label=nm)
    a1.axvspan(6.5, 9.5, color='#f3e6cc', alpha=0.6, lw=0, zorder=0)
    a1.text(8, 15.2, 'Jaru\ndry season', ha='center', va='top', fontsize=8, color='#7a5a1a')
    a1.set_xticks(range(1, 13)); a1.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    a1.set_xlim(0.5, 12.5); a1.set_ylim(0, 16)
    a1.set_xlabel('Month'); a1.set_ylabel('τ (h)'); a1.set_title('(a)', loc='left')
    ok = m[~susp]
    rho = spearmanr(ok['UR_media'], ok['tau_h'])[0]
    for site in ['jaru', 'sinop', 'cuieiras', 'mujui']:
        c, mk, nm = SITE[site]
        g = ok[ok['sitio'] == site]
        a2.plot(g['UR_media'], g['tau_h'], mk, color=c, ms=4.2, mec='k', mew=0.4, ls='none', label=nm)
        g = m[susp & (m['sitio'] == site)]
        if len(g):
            a2.plot(g['UR_media'], g['tau_h'], mk, mfc='white', mec='#7f7f7f', ms=4.2, mew=0.7, ls='none',
                    label='Jaru, suspect series*')
    a2.text(0.42, 0.97, f'Spearman ρ = {f(rho)}\n(n = {len(ok)} months)', transform=a2.transAxes,
            ha='center', va='top', fontsize=8)
    a2.set_xlabel('Monthly mean relative humidity (%)'); a2.set_ylabel('τ (h)'); a2.set_title('(b)', loc='left')
    a2.set_ylim(0, 16)
    h1, l1 = a1.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    extra = [(h, l) for h, l in zip(h2, l2) if 'suspect' in l][:1]
    fig.legend(h1 + [h for h, _ in extra], l1 + [l for _, l in extra], loc='outside lower center', ncol=5,
               fontsize=8, handlelength=2.4, columnspacing=1.0)
    salvar(fig, 'Fig5')
    print('rho', rho, 'n', len(ok))


# ------------------------------------------------------------------ transferência
def transferencia():
    t = pd.read_csv(os.path.join(DATA, 'transferencia_base.csv'))
    ordem = [f'{s} {e}' for s, e in CONJ]
    rot_c = {f'{s} {e}': rot(s, e, curto=True) for s, e in CONJ}
    variantes = [('rmse_completa_K', '(a) Source α, τ and $T_{eq}$'),
                 ('rmse_dinamica_K', '(b) Source α and τ; local $T_{eq}$'),
                 ('rmse_so_tau_K', '(c) Source τ; local α and $T_{eq}$')]
    variantes = [v for v in variantes if v[0] in t.columns]
    loc = t[t['fonte'] == t['alvo']].set_index('alvo')['rmse_local_K']
    fig, axs = plt.subplots(1, len(variantes), figsize=(W, 7.2 * ESC * CM), layout='constrained', sharey=True)
    norm = mcolors.TwoSlopeNorm(vmin=0.8, vcenter=1.0, vmax=2.5)
    cmap = plt.get_cmap('RdBu_r')
    for ax, (col, tit) in zip(np.atleast_1d(axs), variantes):
        P = t.pivot(index='fonte', columns='alvo', values=col).reindex(index=ordem, columns=ordem)
        Rz = P / loc.reindex(ordem).values[None, :]
        im = ax.imshow(Rz.values, cmap=cmap, norm=norm, aspect='equal')
        for i in range(len(ordem)):
            for k in range(len(ordem)):
                v = P.values[i, k]
                ax.text(k, i, f(v, 1), ha='center', va='center', fontsize=7,
                        color='white' if Rz.values[i, k] > 1.9 else 'k', fontweight='bold' if i == k else None)
        ax.set_xticks(range(len(ordem))); ax.set_xticklabels([rot_c[o] for o in ordem], rotation=60, ha='right')
        ax.set_yticks(range(len(ordem))); ax.set_yticklabels([rot_c[o] for o in ordem])
        ax.set_xlabel('Target (test)'); ax.set_title(tit, fontsize=8.5, loc='left')
        for s in ['top', 'right']:
            ax.spines[s].set_visible(True)
    np.atleast_1d(axs)[0].set_ylabel('Parameter source')
    cb = fig.colorbar(im, ax=axs, shrink=0.8, pad=0.01, extend='max', ticks=[0.8, 0.9, 1.0, 1.5, 2.0, 2.5])
    from matplotlib.ticker import FuncFormatter
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p_: f(v, 1)))
    cb.set_label('RMSE / local RMSE')
    salvar(fig, 'Fig7')


# ------------------------------------------------------------------ Artigo 1: superfície do ajuste
def artigo1():
    """RMSE da integração única da estação (como no Artigo 1) no plano (α*, r), r = γ*<T^4>/(α* R̄).
    r = 1 é a curva do balanço médio (α* R̄ = γ* <T^4>); ao longo dela, α* fixa τ."""
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    U = 1e6 / 3600.0
    dt = 0.5 / U
    d = pd.read_csv(os.path.join(DATA, 'jaru_30min.csv'), index_col=0, parse_dates=True)
    rep = pd.read_csv(os.path.join(DATA, 'replica_artigo1.csv'))
    fig, axs = plt.subplots(1, 2, figsize=(W, 8.4 * ESC * CM), layout='constrained')
    cfg = {'seca': ((1.2, 1e-8), 40.0, (0.25, 4.0), (0.95, 1.05), [3.15, 3.3, 3.5, 3.75, 4, 4.5, 5, 6, 8]),
           'chuvosa': ((0.2, 1e-8), 4.0, (0.03, 1.2), (0.95, 1.05), [2.2, 2.3, 2.4, 2.5, 2.6, 2.8, 3, 3.5, 4, 5])}
    for ax, (est, (g0, k, (amin, amax), (rmin, rmax), lev)) in zip(axs, cfg.items()):
        s_ = d[d['estacao'] == est].sort_index()
        T = s_['T_K'].interpolate(limit_direction='both').values
        R = s_['Rn'].interpolate(limit_direction='both').values
        Rm, T4, Tm = R.mean(), (T ** 4).mean(), T.mean()
        c = T4 / Rm
        A, Rr = np.meshgrid(np.geomspace(amin, amax, 160), np.linspace(rmin, rmax, 121))
        a = A.ravel(); g = Rr.ravel() * a / c
        x = np.full(a.shape, T[0]); se = np.zeros(a.shape)
        with np.errstate(all='ignore'):
            for i in range(1, len(T)):
                ar = a * R[i - 1]
                k1 = dt * (ar - g * x ** 4); y = x + 0.5 * k1
                k2 = dt * (ar - g * y ** 4); y = x + 0.5 * k2
                k3 = dt * (ar - g * y ** 4); y = x + k3
                k4 = dt * (ar - g * y ** 4)
                x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
                x = np.where(np.isfinite(x), np.clip(x, 150, 450), 450)
                se += (T[i] - x) ** 2
        rm = np.sqrt(se / len(T)).reshape(A.shape)
        lev = [np.floor(rm.min() * 50) / 50] + [v for v in lev if v > rm.min() + 0.02]
        bounds = lev + [max(lev[-1] * 2, 60)]
        norm = BoundaryNorm(bounds, 256)
        cf = ax.contourf(A, Rr, rm, levels=bounds, cmap='cividis_r', norm=norm, extend='neither')
        cs = ax.contour(A, Rr, rm, levels=lev, colors='k', linewidths=0.4)
        ax.clabel(cs, fmt=lambda v: f(v, 2 if v < 4 else 1), fontsize=7, inline_spacing=1)
        # região fora dos limites do ajuste do Artigo 1 (hachurada)
        a_lo, a_hi, g_lo, g_hi = g0[0] / k, g0[0] * k, g0[1] / k, g0[1] * k
        fora = (A < a_lo) | (A > a_hi) | (Rr * A / c < g_lo) | (Rr * A / c > g_hi)
        if fora.any():
            with plt.rc_context({'hatch.color': '#a0522d', 'hatch.linewidth': 0.5}):
                ax.contourf(A, Rr, fora.astype(float), levels=[0.5, 1.5], colors='none', hatches=['//'])
            ax.contour(A, Rr, fora.astype(float), levels=[0.5], colors='#D55E00', linewidths=0.9, linestyles=':')
        ax.axhline(1.0, color='white', lw=0.8, ls='--')
        r = rep[rep['estacao'] == est]
        pts = [('publicado', 's', '#E69F00', 6.5), ('artigo1', 'X', 'k', 6.5), ('otimo', '*', '#CC79A7', 11)]
        for chave, mk, col, ms in pts:
            q = r[r['procedimento'].str.startswith(chave)].iloc[0]
            ax.plot(q['a_A1'], q['g_A1'] * c / q['a_A1'], mk, color=col, mec='k', mew=0.5, ms=ms, zorder=6)
        r0 = g0[1] * c / g0[0]
        ax.annotate(f'initial estimate:\nα* = {f(g0[0], 1)}; r = {f(r0, 2)}', xy=(g0[0], rmax if r0 > rmax else rmin),
                    xytext=(0.80, 0.87) if r0 > rmax else (0.40, 0.15), textcoords='axes fraction', ha='center',
                    va='center', fontsize=8,
                    arrowprops=dict(arrowstyle='->', lw=0.7, color='k'),
                    bbox=dict(fc='white', ec='none', alpha=0.85, pad=0.6))
        ax.set_xscale('log'); ax.set_xlim(amin, amax); ax.set_ylim(rmin, rmax)
        ticks = [v for v in [0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3, 4] if amin <= v <= amax]
        ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p_: '%g' % v))
        ax.set_xlabel(r'$\alpha^*$ (K per $10^6$ s per W m$^{-2}$)')
        ax.set_ylabel(r'$r = \gamma^* \langle T^4 \rangle / (\alpha^* R_m)$')
        ax.set_title(f"({'a' if est == 'seca' else 'b'}) Jaru 2008, {EST[est].lower()} season", loc='left')
        tau = lambda al: U * T4 / (4 * np.asarray(al, float) * Rm * Tm ** 3)
        sec = ax.secondary_xaxis('top', functions=(tau, tau))
        sec.set_xlabel(r'$\tau_1$ along the curve $r$ = 1 (h)', fontsize=8)
        tt = [v for v in [50, 100, 150, 200, 300, 500, 1000, 2000] if tau(amax) <= v <= tau(amin)]
        sec.xaxis.set_major_locator(FixedLocator(tt)); sec.xaxis.set_minor_formatter(NullFormatter())
        sec.xaxis.set_major_formatter(FuncFormatter(lambda v, p_: '%g' % v))
        sec.tick_params(labelsize=8)
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(True)
        cb = fig.colorbar(cf, ax=ax, shrink=0.9, pad=0.01, ticks=lev[::2])
        cb.set_label('RMSE (K)', fontsize=8)
        cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p_: f(v, 2 if v < 4 else 1)))
        cb.ax.tick_params(labelsize=8)
        print(est, 'min grid', rm.min().round(3), 'std T', T.std().round(3), flush=True)
    fig.legend(handles=[Line2D([], [], color='w', marker='s', mfc='#E69F00', mec='k', ms=6, label='published (da Silva et al. 2026)'),
                        Line2D([], [], color='w', marker='X', mfc='k', mec='k', ms=6, label='replication (L-BFGS-B, as in the original)'),
                        Line2D([], [], color='w', marker='*', mfc='#CC79A7', mec='k', ms=10, label='minimum (Nelder–Mead, log scale)'),
                        Line2D([], [], color='#777777', ls='--', lw=0.8, label='mean-balance curve (r = 1)'),
                        Patch(fc='white', ec='#a0522d', hatch='//', ls=':', label='outside the original search bounds')],
               loc='outside lower center', ncol=3, fontsize=8, columnspacing=1.2)
    salvar(fig, 'Fig2')


# ------------------------------------------------------------------ alturas de medida
def alturas():
    """Perfis de τ e da amplitude do ciclo diurno médio em função da altura relativa ao dossel (z/h), com o mesmo
    Rn e só os instantes em que todas as alturas do sítio têm T válida (alturas.py). Jaru, sem perfil, entra com
    os conjuntos principais a 62 m (dossel de cerca de 33 m), como no texto."""
    a = pd.read_csv(os.path.join(DATA, 'alturas_parametros.csv'))
    b = pd.read_csv(os.path.join(DATA, 'linha_base_parametros.csv'))
    b = b[b['forcante'] == 'Rn']
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 7.4 * ESC * CM), layout='constrained', sharey=True,
                                 gridspec_kw=dict(width_ratios=[1.25, 1]))
    estilo = {'seca': '-', 'chuvosa': '--', 'terrestre': '-', 'inundada': '--'}
    for (site, est), g in a.groupby(['sitio', 'estacao'], sort=False):
        g = g.sort_values('z_sobre_h')
        c, mk, nm = SITE[site]
        mfc = c if estilo[est] == '-' else 'white'
        lab = f"{nm.split(' (')[0]} – {EST[est].lower()}"
        a1.errorbar(g['tau_h'], g['z_sobre_h'], xerr=[g['tau_h'] - g['tau_h_p05'], g['tau_h_p95'] - g['tau_h']],
                    fmt=mk, ms=4, mfc=mfc, mec=c, color=c, elinewidth=0.6, capsize=1.5, ls='none', zorder=3)
        a1.plot(g['tau_h'], g['z_sobre_h'], color=c, ls=estilo[est], lw=0.9, zorder=2)
        a2.plot(g['amplitude_ciclo_medio_K'], g['z_sobre_h'], color=c, ls=estilo[est], lw=0.9, marker=mk, ms=4,
                mfc=mfc, mec=c, zorder=3, label=lab)
    c, mk, _ = SITE['jaru']
    zj = 62 / 33
    for est in ['chuvosa', 'seca']:
        r = b[(b['sitio'] == 'jaru') & (b['estacao'] == est)].iloc[0]
        mfc = c if est == 'seca' else 'white'
        a1.errorbar([r['tau_h']], [zj], xerr=[[r['tau_h'] - r['tau_h_p05']], [r['tau_h_p95'] - r['tau_h']]],
                    fmt=mk, ms=4.5, mfc=mfc, mec=c, color=c, elinewidth=0.6, capsize=1.5, zorder=3)
        a2.plot([], [], ls='none', marker=mk, ms=4.5, mfc=mfc, mec=c, label=f'Jaru – {EST[est].lower()} (62 m)')
    for ax in (a1, a2):
        ax.axhspan(0, 1, color='#e3f0e3', lw=0, zorder=0)
        ax.axhline(1, color='#5a8a5a', lw=0.6, ls=':', zorder=1)
        ax.grid(axis='x', color='#e5e5e5', lw=0.5)
    a1.text(0.98, 0.5, 'canopy', transform=a1.get_yaxis_transform(), ha='right', va='center', fontsize=8,
            color='#3f6f3f')
    a1.set_ylim(0, 2.0); a1.set_xlim(1.5, 7.5)
    a1.set_xlabel('τ (h)'); a1.set_ylabel('Relative height ($z/h$)')
    a2.set_xlabel('Amplitude of the mean diurnal cycle (K)')
    a1.set_title('(a)', loc='left'); a2.set_title('(b)', loc='left')
    h, l = a2.get_legend_handles_labels()
    fig.legend(h, l, loc='outside right center', fontsize=8, handlelength=2.4)
    salvar(fig, 'Fig6')


if __name__ == '__main__':
    nomes = sys.argv[1:] or ['mapa', 'series', 'ciclo', 'estacoes', 'mensal', 'fluxos', 'dia_noite', 'transferencia', 'artigo1',
                             'alturas']
    for n in nomes:
        globals()[n]()
