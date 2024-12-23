# Chat com Websites

Este é um projeto de chatbot conversacional que utiliza técnicas de Recuperação de Respostas Baseada em Conhecimento (“RAG” - Retrieval-Augmented Generation) para interagir com conteúdos extraídos de websites. A interface é implementada em Streamlit, tornando-o fácil de usar e executar.

## Funcionalidades

* Permite que o usuário insira a URL de um website e carregue seu conteúdo.

* Divide o texto do website em pedaços menores para um processamento mais eficiente.

* Gera uma base vetorial de embeddings utilizando o OpenAI.

* Mantém um histórico de mensagens para facilitar a continuidade do contexto na conversa.

* Responde às consultas do usuário com base no conteúdo recuperado e no contexto da conversa.