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

# --- Candidatos e dicionário de tradução (nome comum -> id_candidato) ---

with open("data/candidatos/candidatos.csv", encoding="utf-8-sig") as f:
    candidatos = list(csv.DictReader(f))

nome_para_id = {c["nome_urna"]: c["id_candidato"] for c in candidatos}

# --- Banco vetorial ---

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

vectorstore = Chroma(
    persist_directory="data/vectorstores/chroma_candidatos",
    embedding_function=embeddings,
)

busca_duckduckgo = DuckDuckGoSearchRun()


# --- Tools ---

@tool
def buscar_propostas_candidato(candidato: str, pergunta: str) -> str:
    """Busca trechos das propostas de governo de um candidato à presidência do Brasil em 2026.

    Use esta ferramenta sempre que o usuário perguntar sobre propostas, planos de governo
    ou posições de um candidato específico em algum tema (educação, saúde, economia, etc).

    Args:
        candidato: o nome comum do candidato, exatamente como é conhecido (ex: "Lula", "Renan Santos").
        pergunta: a pergunta ou tema a buscar dentro do documento de propostas desse candidato.
    """
    id_candidato = nome_para_id.get(candidato)

    if id_candidato is None:
        candidatos_disponiveis = ", ".join(nome_para_id.keys())
        return f"Candidato '{candidato}' não encontrado. Candidatos disponíveis: {candidatos_disponiveis}"

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4, "filter": {"candidato": id_candidato}}
    )
    resultados = retriever.invoke(pergunta)

    if not resultados:
        return f"Nenhum trecho encontrado sobre '{pergunta}' nas propostas de {candidato}."

    trechos_formatados = []
    for r in resultados:
        pagina = r.metadata["page"] + 1
        nome = r.metadata["nome_urna"]
        trechos_formatados.append(f"[{nome} - Página {pagina}] {r.page_content}")

    return "\n\n".join(trechos_formatados)


@tool
def buscar_informacoes_web(candidato: str, tema: str) -> str:
    """Busca na internet informações atuais sobre um candidato à presidência do Brasil em 2026.

    Use esta ferramenta apenas para perguntas que NÃO sejam sobre propostas de governo
    (isso já é coberto pela ferramenta buscar_propostas_candidato). Prefira esta ferramenta
    para: notícias recentes, biografia, trajetória política, ou quando a busca nas propostas
    não encontrou nada sobre o tema perguntado.

    Args:
        candidato: nome comum do candidato, exatamente como conhecido (ex: "Lula", "Renan Santos").
        tema: o que buscar sobre esse candidato (ex: "biografia", "últimas notícias", "trajetória política").
    """
    if candidato not in nome_para_id:
        candidatos_disponiveis = ", ".join(nome_para_id.keys())
        return f"'{candidato}' não é um dos candidatos à presidência cobertos por este projeto. Candidatos disponíveis: {candidatos_disponiveis}"

    query = f"{candidato} candidato presidente Brasil eleições 2026 {tema}"
    return busca_duckduckgo.invoke(query)


ferramentas = [buscar_propostas_candidato, buscar_informacoes_web]

SYSTEM_PROMPT = """Você é um assistente especializado nas propostas de governo dos candidatos \
à Presidência do Brasil na eleição de 2026, com base em documentos oficiais do TSE.

Responda apenas perguntas relacionadas a este tema: propostas de governo, biografia e \
trajetória pública dos candidatos, e temas de políticas públicas ligados à eleição.

Se a pergunta não tiver relação com este tema, explique educadamente que este assistente \
é focado apenas nas propostas dos candidatos à Presidência 2026, sem tentar responder \
sobre outros assuntos."""

# --- LLM com tools vinculadas ---

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
llm_com_ferramentas = llm.bind_tools(ferramentas)


# --- Grafo do agente ---

def no_llm(state: MessagesState):
    resposta = llm_com_ferramentas.invoke(state["messages"])
    return {"messages": [resposta]}


def deve_continuar(state: MessagesState):
    ultima_mensagem = state["messages"][-1]
    if ultima_mensagem.tool_calls:
        return "tools"
    return END


workflow = StateGraph(MessagesState)
workflow.add_node("agente", no_llm)
workflow.add_node("tools", ToolNode(ferramentas))

workflow.set_entry_point("agente")
workflow.add_conditional_edges("agente", deve_continuar, {"tools": "tools", END: END})
workflow.add_edge("tools", "agente")

grafo = workflow.compile()


# --- Função de conveniência para o Streamlit ---

def responder(pergunta: str) -> str:
    """Envia uma pergunta ao agente e devolve só o texto da resposta final."""
    resultado = grafo.invoke({
        "messages": [SystemMessage(content=SYSTEM_PROMPT), ("user", pergunta)]
    })
    conteudo = resultado["messages"][-1].content
    return _extrair_texto(conteudo)

def _extrair_texto(conteudo) -> str:
    if isinstance(conteudo, str):
        return conteudo
    return "\n".join(
        bloco["text"] for bloco in conteudo
        if isinstance(bloco, dict) and bloco.get("type") == "text"
    )


def sintetizar_resposta(candidato: str, tema: str, trechos: str) -> str:
    """Resume trechos brutos de propostas em tópicos curtos e legíveis."""
    prompt = f"""Com base nos trechos abaixo, extraídos do documento de propostas de governo de {candidato},
resuma em até 4 tópicos curtos as propostas para o tema "{tema}".
Use apenas as informações dos trechos, sem adicionar nada de fora deles.

Trechos:
{trechos}"""
    resposta = llm.invoke(prompt)
    return _extrair_texto(resposta.content)
