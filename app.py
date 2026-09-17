import csv
import re
import streamlit as st
from src.agente import responder, buscar_propostas_candidato, sintetizar_resposta

st.set_page_config(page_title="Eleições 2026 - Propostas", layout="wide")

with open("data/candidatos/candidatos.csv", encoding="utf-8-sig") as f:
    candidatos = list(csv.DictReader(f))

nomes = [c["nome_urna"] for c in candidatos]

# Sinalizador único: True sempre que alguma chamada ao agente está em andamento,
# em QUALQUER modo. Usado pra desabilitar todos os widgets (inclusive a sidebar)
# e evitar que uma troca de widget interrompa uma consulta em andamento.
ocupado = st.session_state.get("ocupado", False)

with st.sidebar:
    st.markdown("### Modo")
    modo = st.radio("modo", ["Chat", "Comparar temas"], label_visibility="collapsed", disabled=ocupado)

    with st.expander("Ver todos os candidatos (13)"):
        for c in candidatos:
            st.write(f"**{c['nome_urna']}** — {c['partido']}, nº {c['numero']}")

st.title("Eleições 2026 - Propostas de governo")

if modo == "Chat":
    if "mensagens" not in st.session_state:
        st.session_state.mensagens = []

    st.write("Experimente:")
    col1, col2, col3 = st.columns(3)
    sugestoes = [
        "O que Lula propõe para educação?",
        "O que Renan Santos propõe para segurança pública?",
        "O que Flávio Bolsonaro propõe para economia?",
    ]
    pergunta_via_botao = None
    for col, texto in zip([col1, col2, col3], sugestoes):
        if col.button(texto, disabled=ocupado, use_container_width=True):
            pergunta_via_botao = texto

    for msg in st.session_state.mensagens:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    pergunta_digitada = st.chat_input(
        "Pergunte sobre um ou mais candidatos...",
        disabled=ocupado,
    )
    pergunta = pergunta_digitada or pergunta_via_botao

    # Fase 1: pergunta nova chegou -> guarda, trava a tela (ocupado=True) e força
    # redesenho ANTES de gastar tempo chamando a API.
    if pergunta and not ocupado:
        st.session_state.mensagens.append({"role": "user", "content": pergunta})
        st.session_state.ocupado = True
        st.rerun()

    # Fase 2: já está travado -> agora sim chama o agente.
    if ocupado:
        if st.button("Cancelar consulta"):
            st.session_state.ocupado = False
            st.rerun()

        with st.chat_message("assistant"):
            with st.spinner("Consultando..."):
                try:
                    resposta = responder(st.session_state.mensagens[-1]["content"])
                except Exception:
                    resposta = "Não consegui consultar agora — tente novamente em instantes."
            st.write(resposta)
        st.session_state.mensagens.append({"role": "assistant", "content": resposta})
        st.session_state.ocupado = False
        st.rerun()

elif modo == "Comparar temas":
    with open("data/temas.csv", encoding="utf-8-sig") as f:
        temas = list(csv.DictReader(f))

    nomes_temas = [t["nome_tema"] for t in temas]
    tema_selecionado = st.selectbox("Escolha um tema", nomes_temas, disabled=ocupado)
    tema_info = next(t for t in temas if t["nome_tema"] == tema_selecionado)

    col_sel1, col_sel2 = st.columns(2)
    candidato_1 = col_sel1.selectbox("Candidato 1", nomes, index=0, disabled=ocupado)
    candidato_2 = col_sel2.selectbox("Candidato 2", nomes, index=1, disabled=ocupado)

    if candidato_1 == candidato_2:
        st.warning("Escolha dois candidatos diferentes para comparar.")

    comparar = st.button("Comparar", disabled=(ocupado or candidato_1 == candidato_2))

    # Fase 1: clicou em Comparar -> guarda o pedido, trava a tela, força redesenho.
    if comparar and not ocupado:
        st.session_state.pedido_comparacao = {
            "tema": tema_selecionado,
            "c1": candidato_1,
            "c2": candidato_2,
        }
        st.session_state.ocupado = True
        st.rerun()

    # Fase 2: já está travado e existe um pedido pendente -> executa as buscas.
    if ocupado and "pedido_comparacao" in st.session_state:
        pedido = st.session_state.pedido_comparacao
        resultado = {"tema": pedido["tema"], "c1": pedido["c1"], "c2": pedido["c2"]}

        for chave, candidato in [("c1", pedido["c1"]), ("c2", pedido["c2"])]:
            with st.spinner(f"Buscando propostas de {candidato}..."):
                try:
                    trechos = buscar_propostas_candidato.invoke({
                        "candidato": candidato,
                        "pergunta": tema_info["pergunta_padrao"],
                    })
                    resumo = sintetizar_resposta(candidato, pedido["tema"], trechos)
                except Exception:
                    resumo, trechos = "Não foi possível buscar agora. Tente novamente.", ""
            resultado[chave] = {"nome": candidato, "resumo": resumo, "trechos": trechos}

        st.session_state.resultado_comparacao = resultado
        del st.session_state.pedido_comparacao
        st.session_state.ocupado = False
        st.rerun()

    resultado = st.session_state.get("resultado_comparacao")
    mesma_selecao = (
        resultado
        and resultado["tema"] == tema_selecionado
        and resultado["c1"]["nome"] == candidato_1
        and resultado["c2"]["nome"] == candidato_2
    )

    if mesma_selecao:
        col1, col2 = st.columns(2)
        for col, chave in zip([col1, col2], ["c1", "c2"]):
            dados = resultado[chave]
            info = next(c for c in candidatos if c["nome_urna"] == dados["nome"])
            with col:
                if info["tem_foto"] == "sim":
                    st.image(f"data/candidatos/fotos/{info['id_candidato']}.jpg", width=120)
                st.subheader(dados["nome"])
                st.caption(f"{info['partido']} · Número {info['numero']}")
                st.write(dados["resumo"])

                paginas = sorted(set(int(p) for p in re.findall(r"Página (\d+)", dados["trechos"])))
                if paginas:
                    st.caption(f"Fonte: páginas {', '.join(map(str, paginas))} do documento de propostas")

                if info["tem_proposta"] == "sim":
                    caminho_pdf = f"data/candidatos/propostas/{info['id_candidato']}.pdf"
                    with open(caminho_pdf, "rb") as f:
                        st.download_button(
                            "Baixar proposta completa (PDF)",
                            data=f,
                            file_name=f"proposta_{info['id_candidato']}.pdf",
                            mime="application/pdf",
                        )
    elif not ocupado:
        st.info("Selecione os candidatos e clique em Comparar.")