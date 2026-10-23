# DocChat

Aplicação em Python para conversar em português com documentos PDF. As respostas aparecem em tempo real, acompanhadas dos arquivos, páginas e trechos recuperados na busca.

Construída com Streamlit, LangChain, ChromaDB e a API da OpenAI.

## Recursos

- Upload de vários PDFs com texto selecionável.
- Indexação sem repetir arquivos com o mesmo nome na sessão.
- Perguntas de continuação com histórico da conversa.
- Respostas com streaming e painel de fontes recuperadas.
- Remoção de documentos e limpeza da conversa.
- Escolha do modelo e da quantidade de trechos consultados.
- Bases de documentos separadas por sessão do navegador.

## Requisitos

Python 3.13 e uma chave da API da OpenAI com acesso ao modelo escolhido e saldo disponível.

## Instalação

Baixe ou clone este repositório e abra um terminal na pasta do projeto.

No Windows (PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -m streamlit run app.py
```

No Linux ou macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

Informe a chave na barra lateral. Para uso local, também é possível copiar `.env.example` para `.env` e preencher `OPENAI_API_KEY`.

## Uso e privacidade

Envie os PDFs pela barra lateral e faça uma pergunta. Para substituir um arquivo por outra versão com o mesmo nome, remova o documento anterior antes de enviar a nova versão.

O texto dos PDFs é enviado à OpenAI para criar embeddings. A pergunta, o histórico e os trechos recuperados são enviados para gerar a resposta. As chamadas podem gerar cobranças na conta associada à chave utilizada.

A chave digitada fica na memória da sessão. Os textos e embeddings ficam no servidor, em `db/sessions/`, em uma base separada para cada sessão. Uma nova sessão não recupera os documentos da anterior. Encerrar o navegador não apaga os arquivos do disco; remova os documentos pelo aplicativo antes de sair se precisar excluir o conteúdo indexado. A base antiga em `db/` é preservada e não é carregada automaticamente pela interface.

O `.gitignore` exclui variáveis de ambiente, segredos do Streamlit, bases locais, PDFs, uploads e ambientes virtuais. O arquivo `.env.example` contém apenas o nome da variável, sem credenciais.

## Limitações

PDFs escaneados precisam de OCR antes do upload. As fontes exibidas mostram os trechos recuperados, não uma garantia de que cada afirmação foi verificada. A busca consulta um número limitado de trechos; resumos e listas podem omitir informações de documentos longos. Revise as respostas no documento original.

A separação por sessão não substitui autenticação e controle de acesso para hospedar documentos confidenciais. Uma chave configurada no servidor estará disponível para as chamadas de todos os usuários com acesso ao aplicativo.

## Testes

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements_dev.txt
.\venv\Scripts\python.exe -m pytest -q
```

Os testes usam PDFs em memória, embeddings locais e um modelo simulado, sem chamadas pagas à OpenAI. O GitHub Actions executa a suíte em pushes e pull requests.

## Estrutura

| Arquivo | Responsabilidade |
| --- | --- |
| `app.py` | Interface, configuração e estado de sessão |
| `rag.py` | Leitura de PDFs, indexação, recuperação e respostas |
| `tests/` | Testes automatizados |
| `.streamlit/config.toml` | Tema e configuração da interface |
| `.env.example` | Exemplo de configuração local |

## Publicação no GitHub

Crie um repositório vazio no GitHub. Nesta pasta, execute:

```powershell
git init
git add .
git status --short
git diff --cached --stat
git commit -m "Prepare DocChat for publication"
git branch -M main
```

Depois, use os comandos de `git remote add origin` e `git push` fornecidos pelo GitHub para o seu repositório. Confira os arquivos preparados antes do commit: `.env`, `db/`, PDFs e `venv/` não devem aparecer. O `.gitignore` não remove arquivos que já tenham sido versionados em outro repositório.
