# Eleições 2026 — Propostas de Governo (Presidência)

Aplicação de RAG (Retrieval-Augmented Generation) que permite consultar e comparar as propostas de governo dos candidatos à Presidência do Brasil em 2026, com base nos documentos oficiais registrados no TSE.

Projeto de estudo/portfólio em Engenharia de IA, construído com LangChain, LangGraph, ChromaDB e Streamlit.

> **Aviso:** este projeto é uma ferramenta de consulta a documentos públicos oficiais do TSE. As respostas são geradas a partir dos PDFs de propostas de governo de cada candidato, sem interpretação, avaliação ou opinião do desenvolvedor sobre qualquer candidato. Em caso de dúvida, consulte sempre a fonte original no [Portal de Dados Abertos do TSE](https://dadosabertos.tse.jus.br).

## O que o app faz

- **Chat**: pergunte livremente sobre um ou mais candidatos ("O que Lula propõe para educação?", "Compare Zema e Renan Santos em economia") e um agente decide sozinho quais fontes consultar.
- **Comparar temas**: escolha um tema (Educação, Saúde, Segurança, Economia, Meio Ambiente) e dois candidatos, e veja as propostas lado a lado, com foto, partido, número, páginas-fonte e o PDF original disponível para download.

## Escopo

Cobre os **13 candidatos à Presidência da República em 2026** registrados no TSE. Não inclui candidatos a outros cargos (governador, senador, etc.) nem pesquisas eleitorais — decisões de escopo explicadas abaixo.

## Fonte dos dados

Todos os dados (ficha de candidatura, foto oficial e proposta de governo em PDF) vêm do [Portal de Dados Abertos do TSE](https://dadosabertos.tse.jus.br), dataset "Candidatos 2026", filtrado para o cargo de Presidente. São dados públicos oficiais — este projeto não interpreta, resume com opinião própria, nem avalia as propostas; apenas recupera e organiza o que está nos documentos.

**Nota:** situações de registro de candidatura (elegibilidade, impugnações etc.) podem mudar até a eleição. Os dados aqui refletem o momento em que foram baixados do portal, não necessariamente o status mais atual.

## Arquitetura

```
PDFs de propostas (TSE)
        │
        ▼
Ingestão: PyPDFLoader + chunking (LangChain)
        │
        ▼
Embeddings locais (sentence-transformers, multilíngue)
        │
        ▼
ChromaDB (com metadado: candidato, página, partido)
        │
        ▼
Agente LangGraph (LLM + tools: RAG filtrado + busca web)
        │
        ▼
Interface Streamlit (Chat / Comparar temas)
```

## Como rodar localmente

1. Clone o repositório e crie um ambiente virtual:
   ```
   python -m venv venv
   venv\Scripts\Activate.ps1   # Windows
   source venv/bin/activate     # Mac/Linux
   pip install -r requirements.txt
   ```

2. Crie um arquivo `.env` na raiz com sua chave do Gemini:
   ```
   GOOGLE_API_KEY=sua_chave_aqui
   ```
   (gere uma gratuitamente em [aistudio.google.com](https://aistudio.google.com))

3. **Gere o banco vetorial** (não incluído no repositório — veja o porquê abaixo) rodando, em ordem, os notebooks em `notebooks/`:
   - `01_ingestao_chroma.ipynb`

4. Rode o app:
   ```
   streamlit run app.py
   ```

## Decisões técnicas e trade-offs

Esta seção documenta as decisões de engenharia tomadas durante o desenvolvimento — incluindo mudanças de rumo, não só o resultado final.

- **Chroma em vez de FAISS**: o projeto compara 13 candidatos diferentes, e as respostas não podem se misturar entre eles. O Chroma permite filtrar a busca por metadado (`filter={"candidato": id}`) nativamente; com FAISS, seria necessário buscar um volume maior de resultados e filtrar manualmente depois, sem garantia de cobertura.

- **Embeddings locais (sentence-transformers) em vez da API do Gemini**: a ingestão inicial usou embeddings do Gemini, mas a cota gratuita diária (1.000 requisições/dia) não comportou o volume de ~2.450 fatias geradas pelos 13 PDFs. A troca para `paraphrase-multilingual-mpnet-base-v2` (rodando localmente, sem chamada de API) eliminou essa dependência inteiramente — como efeito colateral positivo, o app publicado também não depende de cota externa para essa etapa.

- **DuckDuckGo em vez de Tavily para busca web**: Tavily é mais estruturado e é hoje um padrão de mercado para agentes LLM, mas exige cadastro e chave de API própria. Como a busca web só cobre perguntas fora do escopo dos PDFs (biografia, notícias), o retorno mais simples do DuckDuckGo é suficiente, sem fricção de configuração adicional.

- **Modelo de chat: Gemini 3.5 Flash Lite**: passou por três modelos diferentes ao longo do projeto. `gemini-3-flash-preview` (usado inicialmente) tem cota gratuita diária muito baixa (20 requisições/dia) e maior instabilidade de servidor por ser modelo preview. `gemini-2.5-flash` tinha a mesma cota de 20/dia na conta usada. `gemini-3.5-flash-lite`, por ser um modelo "Lite" de geração mais recente, oferece 500 requisições/dia — 25x mais folga — com qualidade suficiente para as tarefas do agente (decidir qual tool usar, resumir trechos já recuperados), que não exigem raciocínio complexo.

- **Modo "Comparar temas" não passa pelo agente completo**: em vez de deixar o LLM decidir quais tools chamar (como no Chat), esse modo chama a tool de busca diretamente, já sabendo candidato e tema. Isso evita múltiplas chamadas sequenciais ao LLM (uma por candidato, mais uma para decidir), tornando a comparação sensivelmente mais rápida.

- **Sem seleção de candidato "de contexto" na sidebar**: uma versão inicial injetava automaticamente o candidato selecionado na sidebar em toda pergunta do chat. Foi removida por ser um vínculo invisível — o usuário não tinha como saber, sem ler documentação, que a resposta dependia de uma seleção feita em outro canto da tela. O agente já é capaz de identificar candidatos citados diretamente na pergunta (inclusive múltiplos, para comparações), o que resolveu a necessidade sem essa camada extra.

- **Trava de interface durante o processamento**: o Streamlit reexecuta o script inteiro a cada interação, o que inicialmente permitia que uma troca de widget no meio de uma consulta interrompesse a chamada em andamento (e disparasse uma nova, desperdiçando cota de API). A solução foi um sinalizador de estado (`ocupado`) que desabilita todos os widgets da tela enquanto uma resposta está sendo processada.

## Limitações conhecidas / próximos passos

- O banco vetorial (ChromaDB) não é versionado no Git — é recriado localmente a partir dos PDFs. Isso significa que rodar o projeto pela primeira vez exige rodar a ingestão antes do app funcionar.
- Cobre apenas candidatos à Presidência. Governadores ficam como extensão natural (a estrutura de dados já usa `cargo` como campo, facilitando a expansão).
- Pesquisas eleitorais não estão incluídas — não há fonte gratuita e confiável via API para isso; ficaria como um CSV curado manualmente, atualizado periodicamente.
- O download do PDF de propostas sempre abre o documento do início, não na página exata citada — melhorar isso exigiria um leitor de PDF navegável embutido na interface.
