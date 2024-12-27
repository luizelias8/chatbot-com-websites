import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Carrega as variáveis de ambiente
load_dotenv()

def inicializar_modelo_llm():
    # Inicializa o modelo de linguagem
    return ChatGroq(
        model='llama-3.3-70b-versatile', # Modelo LLM a ser usado
        temperature=0.2, # Baixa temperatura para respostas mais precisas
        max_tokens=500 # Limite de tokens na resposta
    )

def obter_base_vetores_da_url(url):
    """Carrega o conteúdo de um website, divide o texto em pedaços e cria uma base vetorial a partir do conteúdo."""
    # Carrega o conteúdo do website
    carregador = WebBaseLoader(url)
    documento = carregador.load()

    # Divide o texto em pedaços menores para análise
    divisor_texto = RecursiveCharacterTextSplitter()
    pedacos_documento = divisor_texto.split_documents(documento)

    # Configura o modelo de embeddings
    modelo_embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2', # Modelo a ser usado
        model_kwargs={'device': 'cpu'}, # Usa CPU para processamento
        encode_kwargs={'normalize_embeddings': False} # Não normaliza os embeddings
    )

    # Cria a base vetorial para armazenar embeddings
    base_vetores = FAISS.from_documents(pedacos_documento, modelo_embeddings)
    return base_vetores

def obter_cadeia_recuperador_contexto(base_vetores):
    """Configura uma cadeia de recuperação de contexto baseada no histórico de conversação."""
    # Inicializa o modelo de linguagem
    modelo = inicializar_modelo_llm()

    # Define o recuperador de documentos baseado na base vetorial
    recuperador = base_vetores.as_retriever()

    # Configura o prompt para geração de consultas de busca
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name='chat_history'),
        ('user', '{input}'),
        ('user', 'Com base na conversa acima, gere uma consulta de busca para encontrar informações relevantes')
    ])

    # Cria a cadeia de recuperação sensível ao histórico de chat
    cadeia_recuperador = create_history_aware_retriever(modelo, recuperador, prompt)
    return cadeia_recuperador

def obter_cadeia_rag_conversacional(cadeia_recuperador):
    """Configura a cadeia de RAG conversacional para gerar respostas baseadas no contexto recuperado."""
    # Inicializa o modelo de linguagem
    modelo = inicializar_modelo_llm()

    # Configura o prompt para geração de respostas
    prompt = ChatPromptTemplate.from_messages([
        ('system', 'Responda às perguntas do usuário com base no contexto abaixo:\n\n{context}'),
        MessagesPlaceholder(variable_name='chat_history'),
        ('user', '{input}')
    ])

    # Cria a cadeia para processar documentos
    cadeia_processar_documentos = create_stuff_documents_chain(modelo, prompt)

    # Retorna a cadeia de recuperação integrada
    return create_retrieval_chain(cadeia_recuperador, cadeia_processar_documentos)

def obter_resposta(entrada_usuario):
    """Processa a entrada do usuário e retorna a resposta do chatbot."""
    # Obtém as cadeias de recuperação e RAG
    cadeia_recuperador = obter_cadeia_recuperador_contexto(st.session_state.base_vetores)
    cadeia_rag_conversacional = obter_cadeia_rag_conversacional(cadeia_recuperador)

    # Processa a entrada do usuário e retorna a resposta gerada
    resposta = cadeia_rag_conversacional.invoke({
        'chat_history': st.session_state.historico_chat,
        'input': entrada_usuario
    })

    return resposta['answer']

def main():
    """
    Função principal para configurar e executar a interface da aplicação Streamlit.
    """
    # Configura o título e ícone da página
    st.set_page_config(page_title='Chat com websites', page_icon='🤖')
    st.title('Chat com websites')

    # Configura a barra lateral para entrada da URL
    with st.sidebar:
        st.header('Configurações')
        url_website = st.text_input('URL do website')

    # Verifica se a URL foi inserida
    if url_website is None or url_website == '':
        st.info('Por favor, insira a URL de um website')
    else:
         # Inicializa o histórico de conversação na sessão, se necessário
        if 'historico_chat' not in st.session_state:
            st.session_state.historico_chat = [
                AIMessage(content='Olá, sou um bot. Como posso ajudar?')
            ]

        # Inicializa a base vetorial, se ainda não existente
        if 'base_vetores' not in st.session_state:
            with st.spinner('Gerando o banco vetorial. Isso pode levar alguns minutos...'):
                st.session_state.base_vetores = obter_base_vetores_da_url(url_website)
            st.success('Banco vetorial gerado com sucesso!')

        # Captura a entrada do usuário
        consulta_usuario = st.chat_input('Digite sua mensagem aqui...')

        # Processa e armazena a mensagem do usuário e a resposta do chatbot
        if consulta_usuario is not None and consulta_usuario != '':
            resposta = obter_resposta(consulta_usuario)
            st.session_state.historico_chat.append(HumanMessage(content=consulta_usuario))
            st.session_state.historico_chat.append(AIMessage(content=resposta))

        # Exibe o histórico da conversa no chat
        for mensagem in st.session_state.historico_chat:
            if isinstance(mensagem, AIMessage):
                with st.chat_message('ai'):
                    st.write(mensagem.content)
            elif isinstance(mensagem, HumanMessage):
                with st.chat_message('human'):
                    st.write(mensagem.content)

# Executa a aplicação se o script for chamado diretamente
if __name__ == '__main__':
    main()