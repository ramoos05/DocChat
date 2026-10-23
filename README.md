# 📄 DocChat 

Aplicação desenvolvida em **Python** que utiliza **Retrieval-Augmented Generation (RAG)** para permitir conversas com documentos PDF por meio de Inteligência Artificial.

Construída com **Streamlit, LangChain, ChromaDB e OpenAI API**, oferece busca semântica, histórico de conversas, respostas em streaming e referências às fontes consultadas.

## 🎥 Demonstração



https://github.com/user-attachments/assets/69229350-1286-4d99-bae3-6d0eb70af624


### Interface da aplicação

<img width="1917" height="870" alt="Captura de tela 2026-10-22 215509" src="https://github.com/user-attachments/assets/97a25598-74fd-48b8-9efb-5b96eaedb81e" />

<img width="1917" height="871" alt="Captura de tela 2026-10-22 215450" src="https://github.com/user-attachments/assets/ff24c7f3-0eb2-47c2-826a-a108c8ba4e50" />

## 🛠️ Tecnologias

**Python 3.13** • **Streamlit** • **LangChain** • **ChromaDB** • **OpenAI API** • **Pytest** • **GitHub Actions**

## ✨ Funcionalidades

- Upload e processamento de múltiplos PDFs.
- Indexação e busca semântica com embeddings.
- Respostas em streaming com histórico de conversa.
- Exibição dos documentos, páginas e trechos recuperados.
- Seleção do modelo de IA e quantidade de trechos consultados.
- Gerenciamento de documentos e isolamento da base vetorial por sessão.
- Testes automatizados e integração contínua.

## 🧠 Como funciona?

O DocChat utiliza uma arquitetura RAG para recuperar informações relevantes antes de gerar respostas:

```text
Upload de PDFs
      ↓
Extração e divisão do texto
      ↓
OpenAI Embeddings
      ↓
ChromaDB (Banco Vetorial)
      ↓
Pergunta → Busca Semântica
      ↓
Contexto + Histórico
      ↓
OpenAI LLM
      ↓
Resposta + Fontes
```

## 🚀 Como executar

**1. Clone o repositório**

```bash
git clone https://github.com/ramoos05/DocChat.git
cd DocChat
```

**2. Crie o ambiente virtual e instale as dependências (Windows)**

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

**3. Execute a aplicação**

```powershell
.\venv\Scripts\python.exe -m streamlit run app.py
```

Informe sua chave da OpenAI na barra lateral ou configure `OPENAI_API_KEY` em um arquivo `.env`, utilizando `.env.example` como referência.

## 🧪 Testes

O projeto utiliza **Pytest** com embeddings locais e um modelo simulado, permitindo executar os testes sem chamadas pagas à OpenAI.

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements_dev.txt
.\venv\Scripts\python.exe -m pytest -q
```

O **GitHub Actions** executa os testes em pushes e pull requests.

## ⚠️ Observações

- PDFs digitalizados precisam passar por OCR.
- O conteúdo dos documentos é enviado à OpenAI para processamento.
- Os textos e embeddings ficam armazenados no servidor; fechar o navegador não os exclui.
- O isolamento por sessão não substitui autenticação para documentos confidenciais.
- Respostas geradas por IA devem ser verificadas nas fontes originais.
- A utilização da API da OpenAI pode gerar custos.

## 👨‍💻 Desenvolvedor

**Patrick Ramos** — Estudante de Análise e Desenvolvimento de Sistemas, com foco em Backend Python e Inteligência Artificial.

[GitHub](https://github.com/ramoos05) • [LinkedIn](https://www.linkedin.com/in/ramos-dev/)
