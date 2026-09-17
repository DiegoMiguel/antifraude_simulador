"""
Simulador de Ponto de Corte — Antifraude Financiamento Veicular
Projeto: Pós em Análise de Dados e IA (CESAR School / Neurotech)

Este app NÃO executa o modelo XGBoost. Ele lê uma tabela cumulativa
já extraída do TESTE_OOT (lookup_table.csv) e faz a matemática das
3 zonas de decisão (aprovar / mesa de análise / bloquear) em R$.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Simulador Antifraude — Ponto de Corte",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# TEMA DARK CYBER-ANALYTICS (CSS customizado)
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --bg-primary: #0B0F19;
    --bg-secondary: #131B2E;
    --bg-card: #182238;
    --accent-cyan: #00E5FF;
    --accent-violet: #A855F7;
    --accent-green: #00FF87;
    --accent-red: #FF3B30;
    --text-primary: #FFFFFF;
    --text-secondary: #8E9BAE;
    --border-subtle: rgba(255, 255, 255, 0.08);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

[data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background-color: var(--bg-primary);
}

[data-testid="stSidebar"] {
    background-color: var(--bg-secondary);
    border-right: 1px solid var(--border-subtle);
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif;
    color: var(--text-primary);
    letter-spacing: -0.02em;
}

.subtitle {
    color: var(--text-secondary);
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
    margin-bottom: 4px;
}

.metric-card {
    background: linear-gradient(145deg, var(--bg-card), var(--bg-secondary));
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 20px 22px;
    box-shadow: 0 0 24px rgba(0, 229, 255, 0.04);
    height: 100%;
}

.metric-label {
    color: var(--text-secondary);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600;
    margin-bottom: 6px;
}

.metric-value {
    font-size: 30px;
    font-weight: 800;
    color: var(--text-primary);
    font-family: 'Space Grotesk', sans-serif;
    line-height: 1.1;
}

.metric-business {
    color: var(--text-secondary);
    font-size: 12.5px;
    margin-top: 8px;
    line-height: 1.4;
}

.value-cyan { color: var(--accent-cyan); }
.value-violet { color: var(--accent-violet); }
.value-green { color: var(--accent-green); }
.value-red { color: var(--accent-red); }

.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.badge-green { background: rgba(0, 255, 135, 0.15); color: var(--accent-green); }
.badge-red { background: rgba(255, 59, 48, 0.15); color: var(--accent-red); }

hr { border-color: var(--border-subtle); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# CONSTANTES DE NEGÓCIO (validadas no SSOT / EDA do projeto)
# ---------------------------------------------------------------------------
VAL_FINANCIADO_FRAUDE = 72_372.43   # valor_financiado médio | fraude_confirmada = 1
VAL_FINANCIADO_BOM = 68_538.69      # valor_financiado médio | fraude_confirmada = 0
FATOR_MARGEM = 0.57                 # fator médio de custo do financiamento (juros) - 1, validado na EDA

TOTAL_FRAUDE = 1432
TOTAL_BOM = 15533


# ---------------------------------------------------------------------------
# CARREGAMENTO DA LOOKUP TABLE (gerada no Colab a partir do TESTE_OOT)
# ---------------------------------------------------------------------------
@st.cache_data
def carregar_lookup():
    df = pd.read_csv("lookup_table.csv")
    df["threshold"] = df["threshold"].round(2)
    return df


df_lookup = carregar_lookup()
THRESHOLDS = df_lookup["threshold"].tolist()


def contagem_em(t: float):
    """Retorna (fraude_score>=t, bom_score>=t) buscando na lookup table."""
    linha = df_lookup.loc[df_lookup["threshold"] == round(t, 2)]
    return int(linha["fraude_score_maior_igual"].values[0]), int(linha["bom_score_maior_igual"].values[0])


def calcular_zonas(li: float, ls: float, custo_mesa: float):
    f_li, b_li = contagem_em(li)
    f_ls, b_ls = contagem_em(ls)

    vp_bloq = f_ls
    fp_bloq = b_ls
    fraude_mesa = f_li - f_ls
    bom_mesa = b_li - b_ls
    fn_aprova = TOTAL_FRAUDE - f_li
    bom_aprova = TOTAL_BOM - b_li

    custo_evitado = vp_bloq * VAL_FINANCIADO_FRAUDE
    risco_assumido = fn_aprova * VAL_FINANCIADO_FRAUDE
    margem_perdida = fp_bloq * FATOR_MARGEM * VAL_FINANCIADO_BOM
    casos_mesa = fraude_mesa + bom_mesa
    custo_operacional = casos_mesa * custo_mesa

    resultado_liquido = custo_evitado - risco_assumido - margem_perdida - custo_operacional

    return dict(
        vp_bloq=vp_bloq, fp_bloq=fp_bloq, fraude_mesa=fraude_mesa, bom_mesa=bom_mesa,
        fn_aprova=fn_aprova, bom_aprova=bom_aprova, casos_mesa=casos_mesa,
        custo_evitado=custo_evitado, risco_assumido=risco_assumido,
        margem_perdida=margem_perdida, custo_operacional=custo_operacional,
        resultado_liquido=resultado_liquido,
    )


def fmt_reais(v: float) -> str:
    s = f"{v:,.2f}"
    s = s.replace(",", "§").replace(".", ",").replace("§", ".")
    sinal = "-" if v < 0 else ""
    return f"{sinal}R$ {s.lstrip('-')}"


# ---------------------------------------------------------------------------
# SIDEBAR — PARÂMETROS
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<p class="subtitle">Parametrização do Modelo</p>', unsafe_allow_html=True)
    st.title("🛡️ Painel de Controle")
    st.markdown("Ajuste os pontos de corte e o custo operacional para simular o impacto financeiro em tempo real.")
    st.markdown("---")

    limite_inferior = st.select_slider(
        "Limite inferior — abaixo disso, aprova automático",
        options=THRESHOLDS,
        value=0.20,
    )

    ls_opcoes = [t for t in THRESHOLDS if t > limite_inferior]
    valor_default_ls = 0.50 if 0.50 in ls_opcoes else ls_opcoes[len(ls_opcoes) // 2]
    limite_superior = st.select_slider(
        "Limite superior — acima disso, bloqueia automático",
        options=ls_opcoes,
        value=valor_default_ls,
    )

    st.markdown("---")
    st.markdown('<p class="subtitle">Custo Operacional</p>', unsafe_allow_html=True)
    custo_mesa = st.number_input(
        "Custo por caso enviado à mesa de análise (R$)",
        min_value=0.0,
        value=0.0,
        step=5.0,
        help="Parâmetro livre — ainda não há benchmark de custo de analista validado pelo time. Ajuste livremente.",
    )
    st.caption("⚠️ Sem valor de referência fixado — ajuste livremente e observe o impacto no resultado líquido.")

# ---------------------------------------------------------------------------
# CÁLCULO
# ---------------------------------------------------------------------------
r = calcular_zonas(limite_inferior, limite_superior, custo_mesa)

# ---------------------------------------------------------------------------
# CABEÇALHO
# ---------------------------------------------------------------------------
st.markdown('<p class="subtitle">Frente C · Estratégia de Decisão</p>', unsafe_allow_html=True)
st.title("Simulador de Ponto de Corte — Impacto em R$")
st.markdown(
    f'Zona atual: <span class="badge badge-green">Aprovar &lt; {limite_inferior:.2f}</span> '
    f'&nbsp; <span class="badge" style="background:rgba(168,85,247,0.15);color:#A855F7;">Mesa {limite_inferior:.2f}–{limite_superior:.2f}</span> '
    f'&nbsp; <span class="badge badge-red">Bloquear ≥ {limite_superior:.2f}</span>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ---------------------------------------------------------------------------
# CARDS DE MÉTRICAS
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Custo Evitado</div>
        <div class="metric-value value-cyan">{fmt_reais(r['custo_evitado'])}</div>
        <div class="metric-business">{r['vp_bloq']} fraudes capturadas no bloqueio automático</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Risco Assumido</div>
        <div class="metric-value value-red">{fmt_reais(r['risco_assumido'])}</div>
        <div class="metric-business">{r['fn_aprova']} fraudes que ainda passam direto na aprovação</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Margem Perdida</div>
        <div class="metric-value value-violet">{fmt_reais(r['margem_perdida'])}</div>
        <div class="metric-business">{r['fp_bloq']} bons clientes bloqueados por engano</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Custo Operacional</div>
        <div class="metric-value" style="color:#8E9BAE;">{fmt_reais(r['custo_operacional'])}</div>
        <div class="metric-business">{r['casos_mesa']} casos enviados à mesa de análise</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

cor_resultado = "value-green" if r["resultado_liquido"] >= 0 else "value-red"
badge_resultado = "badge-green" if r["resultado_liquido"] >= 0 else "badge-red"
st.markdown(f"""
<div class="metric-card" style="border: 1px solid rgba(0,229,255,0.25);">
    <div class="metric-label">Resultado Líquido Simulado</div>
    <div class="metric-value {cor_resultado}" style="font-size:42px;">{fmt_reais(r['resultado_liquido'])}</div>
    <div class="metric-business">Custo Evitado − Risco Assumido − Margem Perdida − Custo Operacional</div>
    <span class="badge {badge_resultado}">{"Ponto de corte financeiramente favorável" if r["resultado_liquido"] >= 0 else "Atenção: resultado negativo neste ponto de corte"}</span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# GRÁFICO — RESULTADO LÍQUIDO EM FUNÇÃO DO LIMITE SUPERIOR
# ---------------------------------------------------------------------------
st.subheader("Resultado Líquido vs. Limite Superior")
st.caption(f"Limite inferior fixo em {limite_inferior:.2f} — variando o limite superior de bloqueio.")

thresholds_validos = [t for t in THRESHOLDS if t > limite_inferior]
resultados_curva = [calcular_zonas(limite_inferior, t, custo_mesa)["resultado_liquido"] for t in thresholds_validos]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=thresholds_validos, y=resultados_curva,
    mode="lines",
    line=dict(color="#00E5FF", width=3),
    fill="tozeroy",
    fillcolor="rgba(0, 229, 255, 0.12)",
    name="Resultado Líquido",
))
fig.add_trace(go.Scatter(
    x=[limite_superior], y=[r["resultado_liquido"]],
    mode="markers",
    marker=dict(color="#A855F7", size=14, line=dict(color="#FFFFFF", width=2)),
    name="Ponto atual",
))
fig.update_layout(
    plot_bgcolor="#131B2E",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#8E9BAE", family="Inter"),
    xaxis=dict(title="Limite superior (threshold de bloqueio)", gridcolor="rgba(255,255,255,0.06)"),
    yaxis=dict(title="Resultado líquido (R$)", gridcolor="rgba(255,255,255,0.06)"),
    margin=dict(l=10, r=10, t=10, b=10),
    height=380,
    showlegend=False,
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Fonte: contagens cumulativas extraídas do conjunto TESTE_OOT (modelo XGBoost). "
    "Nenhuma inferência do modelo é executada neste app — apenas leitura da tabela pré-computada."
)
