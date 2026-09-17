import csv

from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage


load_dotenv()


# ============================================================
# CANDIDATOS
# ============================================================

with open("data/candidatos/candidatos.csv", encoding="utf-8-sig") as f:
    candidatos = list(csv.DictReader(f))

nome_para_id = {
    c["nome_urna"]: c["id_candidato"]
    for c in candidatos
}


# ============================================================
# BANCO VETORIAL
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

vectorstore = Chroma(
    persist_directory="data/vectorstores/chroma_candidatos",
    embedding_function=embeddings,
)


# ============================================================
# BUSCA WEB
# ============================================================

busca_duckduckgo = DuckDuckGoSearchRun()


# ============================================================
# TOOLS
# ============================================================

@tool
def buscar_propostas_candidato(candidato: str, pergunta: str) -> str:
    """
    Busca trechos das propostas de governo de um candidato
    à presidência do Brasil em 2026.

    Use esta ferramenta para perguntas sobre propostas,
    planos de governo ou posições de um candidato em
    algum tema de política pública.
    """

    id_candidato = nome_para_id.get(candidato)

    if id_candidato is None:
        candidatos_disponiveis = ", ".join(nome_para_id.keys())

        return (
            f"Candidato '{candidato}' não encontrado. "
            f"Candidatos disponíveis: {candidatos_disponiveis}"
        )

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 4,
            "filter": {"candidato": id_candidato}
        }
    )

    resultados = retriever.invoke(pergunta)

    if not resultados:
        return (
            f"Nenhum trecho encontrado sobre '{pergunta}' "
            f"nas propostas de {candidato}."
        )

    trechos_formatados = []

    for r in resultados:
        pagina = r.metadata["page"] + 1
        nome = r.metadata["nome_urna"]

        trechos_formatados.append(
            f"[{nome} - Página {pagina}] {r.page_content}"
        )

    return "\n\n".join(trechos_formatados)


@tool
def buscar_informacoes_web(candidato: str, tema: str) -> str:
    """
    Busca na internet informações atuais sobre um candidato
    à presidência do Brasil em 2026.

    Use para notícias, biografia, trajetória política ou
    informações que não estejam nas propostas de governo.
    """

    if candidato not in nome_para_id:
        candidatos_disponiveis = ", ".join(nome_para_id.keys())

        return (
            f"'{candidato}' não é um dos candidatos à "
            f"presidência cobertos por este projeto. "
            f"Candidatos disponíveis: {candidatos_disponiveis}"
        )

    query = (
        f"{candidato} candidato presidente Brasil "
        f"eleições 2026 {tema}"
    )

    return busca_duckduckgo.invoke(query)


ferramentas = [
    buscar_propostas_candidato,
    buscar_informacoes_web
]


# ============================================================
# MODELOS
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)


# ============================================================
# GUARDRAIL DE ESCOPO
# ============================================================

PROMPT_CLASSIFICADOR = """
Você é um classificador de perguntas.

O sistema para o qual você trabalha é especializado EXCLUSIVAMENTE
nas Eleições Presidenciais do Brasil de 2026 e nos candidatos
à Presidência.

Considere como DENTRO DO ESCOPO perguntas sobre:

- eleições presidenciais de 2026;
- candidatos à Presidência;
- propostas de governo;
- planos de governo;
- posições políticas dos candidatos;
- trajetória política pública dos candidatos;
- biografia pública dos candidatos;
- temas de políticas públicas relacionados à eleição;
- comparação factual entre propostas ou posições dos candidatos;
- informações eleitorais relacionadas à eleição de 2026.

Considere como FORA DO ESCOPO perguntas como:

- receitas de bolo;
- programação;
- matemática;
- filmes;
- séries;
- viagens;
- esportes;
- saúde pessoal;
- culinária;
- recomendações de produtos;
- assuntos pessoais;
- qualquer outro assunto que não esteja relacionado
  às Eleições Presidenciais de 2026.

IMPORTANTE:

Responda SOMENTE com uma destas duas palavras:

DENTRO

ou

FORA

Não explique sua decisão.
Não responda à pergunta.
"""


def pergunta_dentro_do_escopo(pergunta: str) -> bool:
    """
    Verifica se a pergunta pertence ao domínio do projeto.

    Retorna:
        True  -> pergunta relacionada às Eleições 2026
        False -> pergunta fora do escopo
    """

    resposta = llm.invoke(
        [
            SystemMessage(content=PROMPT_CLASSIFICADOR),
            ("user", pergunta)
        ]
    )

    resultado = _extrair_texto(resposta.content)

    resultado = resultado.strip().upper()

    return resultado == "DENTRO"


# ============================================================
# PROMPT PRINCIPAL DO AGENTE
# ============================================================

SYSTEM_PROMPT = """
Você é um assistente especializado nas propostas de governo
dos candidatos à Presidência do Brasil na eleição de 2026,
com base em documentos oficiais do TSE.

Responda somente perguntas relacionadas a:

- propostas de governo;
- planos de governo;
- biografia pública;
- trajetória política pública dos candidatos;
- posições dos candidatos;
- temas de políticas públicas ligados à eleição de 2026.

Quando a pergunta for sobre propostas de um candidato,
use a ferramenta buscar_propostas_candidato.

Quando precisar de informações atuais sobre um candidato,
como notícias, biografia ou trajetória política,
use buscar_informacoes_web.

Não invente informações.

Quando não houver informação suficiente nas fontes disponíveis,
deixe isso claro para o usuário.

Se a pergunta não estiver relacionada às Eleições Presidenciais
de 2026 ou aos candidatos, não tente responder ao assunto.
"""


# ============================================================
# LLM COM FERRAMENTAS
# ============================================================

llm_com_ferramentas = llm.bind_tools(ferramentas)


# ============================================================
# GRAFO DO AGENTE
# ============================================================

def no_llm(state: MessagesState):

    resposta = llm_com_ferramentas.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            *state["messages"]
        ]
    )

    return {
        "messages": [resposta]
    }


def deve_continuar(state: MessagesState):

    ultima_mensagem = state["messages"][-1]

    if ultima_mensagem.tool_calls:
        return "tools"

    return END


workflow = StateGraph(MessagesState)

workflow.add_node("agente", no_llm)

workflow.add_node(
    "tools",
    ToolNode(ferramentas)
)

workflow.set_entry_point("agente")

workflow.add_conditional_edges(
    "agente",
    deve_continuar,
    {
        "tools": "tools",
        END: END
    }
)

workflow.add_edge(
    "tools",
    "agente"
)

grafo = workflow.compile()


# ============================================================
# RESPOSTA FORA DO ESCOPO
# ============================================================

RESPOSTA_FORA_DO_ESCOPO = (
    "Desculpe, mas eu sou um assistente especializado nas "
    "Eleições Presidenciais de 2026 e nos candidatos à Presidência. "
    "Posso ajudar com propostas de governo, trajetória pública "
    "dos candidatos e temas relacionados à eleição."
)


# ============================================================
# FUNÇÃO PRINCIPAL USADA PELO STREAMLIT
# ============================================================

def responder(pergunta: str) -> str:
    """
    Recebe uma pergunta do usuário e retorna a resposta final.

    Antes de executar o agente, verifica se a pergunta está
    dentro do escopo do projeto.
    """

    # --------------------------------------------------------
    # 1. GUARDRAIL
    # --------------------------------------------------------

    if not pergunta_dentro_do_escopo(pergunta):
        return RESPOSTA_FORA_DO_ESCOPO

    # --------------------------------------------------------
    # 2. AGENTE
    # --------------------------------------------------------

    resultado = grafo.invoke(
        {
            "messages": [
                ("user", pergunta)
            ]
        }
    )

    conteudo = resultado["messages"][-1].content

    return _extrair_texto(conteudo)


# ============================================================
# EXTRAÇÃO DO TEXTO
# ============================================================

def _extrair_texto(conteudo) -> str:

    if isinstance(conteudo, str):
        return conteudo

    return "\n".join(
        bloco["text"]
        for bloco in conteudo
        if isinstance(bloco, dict)
        and bloco.get("type") == "text"
    )


# ============================================================
# SÍNTESE DE PROPOSTAS
# ============================================================

def sintetizar_resposta(
    candidato: str,
    tema: str,
    trechos: str
) -> str:

    """
    Resume trechos brutos de propostas em tópicos curtos
    e legíveis.
    """

    prompt = f"""
Com base nos trechos abaixo, extraídos do documento de
propostas de governo de {candidato},

resuma em até 4 tópicos curtos as propostas para o tema
"{tema}".

Use apenas as informações dos trechos,
sem adicionar nada de fora deles.

Trechos:

{trechos}
"""

    resposta = llm.invoke(prompt)

    return _extrair_texto(resposta.content)