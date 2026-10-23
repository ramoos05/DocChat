from html import escape
from pathlib import Path
from uuid import uuid4

import streamlit as st
from decouple import config

import rag


st.set_page_config(page_title='DocChat', page_icon='📄', layout='centered')

MODEL_OPTIONS = ['gpt-4o-mini', 'gpt-4o', 'gpt-4-turbo']

QUICK_QUESTIONS = {
    '📝 Resumo geral': 'Faça um resumo geral dos documentos, organizado por tópicos.',
    '🎯 Pontos principais': 'Quais são os pontos mais importantes dos documentos?',
    '📋 Listar itens': 'Liste todos os itens, nomes ou datas importantes que aparecem nos documentos.',
}

CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
  html, body, [class*="css"], .stMarkdown, button, input, textarea { font-family: 'Inter', sans-serif; }
  #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { visibility: hidden; }
  header[data-testid="stHeader"] { background: transparent; }
  [data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"] { visibility: visible; color: #e9e9f5; }

  .stApp { background-color: #07070d;
           background-image: radial-gradient(700px 420px at 12% 0%, rgba(91,45,200,.40), transparent),
                             radial-gradient(620px 420px at 100% 100%, rgba(15,143,184,.30), transparent);
           background-attachment: fixed; }
  .block-container { padding-top: 2rem; padding-bottom: 6rem; max-width: 860px; }

  .hero { text-align: center; margin: 18px 0 28px; }
  .hero h1 { margin: 0; font-size: 2.8rem; font-weight: 800; letter-spacing: -.03em; padding: 0;
             background: linear-gradient(90deg, #ffffff, #b9a6ff 60%, #6fd3ff);
             -webkit-background-clip: text; background-clip: text; color: transparent; }
  .hero p { margin: 10px 0 14px; color: #a9aac6; font-size: 1.05rem; }
  .chip { display: inline-block; font-size: .75rem; font-weight: 600; color: #cfd0ea; padding: 5px 13px; margin: 0 3px;
          border-radius: 999px; background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.14); }

  .glass, .step { background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.10); border-radius: 20px;
                  backdrop-filter: blur(10px); }
  .step { padding: 20px; height: 100%; transition: border-color .2s, transform .2s; }
  .step:hover { border-color: rgba(124,92,255,.8); transform: translateY(-2px); }
  .num { display: inline-flex; align-items: center; justify-content: center; width: 30px; height: 30px; border-radius: 50%;
         background: linear-gradient(135deg, #7c5cff, #38bdf8); color: #fff; font-weight: 700; font-size: .85rem; margin-bottom: 12px; }
  .step b { display: block; color: #f4f4fb; font-size: 1.02rem; margin-bottom: 5px; }
  .step span { color: #a9aac6; font-size: .9rem; line-height: 1.5; }

  [data-testid="stChatMessage"] { border-radius: 18px; border: 1px solid rgba(255,255,255,.10); background: rgba(255,255,255,.05);
                                  backdrop-filter: blur(10px); padding: 14px 16px; }
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) { background: rgba(124,92,255,.14); border-color: rgba(124,92,255,.35); }
  [data-testid="stChatInput"] { border-radius: 18px; background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.14); }
  [data-testid="stBottom"] > div { background: transparent; }
  [data-testid="stExpander"] { border: 1px solid rgba(255,255,255,.10); border-radius: 14px; background: rgba(255,255,255,.03); }

  [data-testid="stSidebar"] { background: rgba(10,10,20,.85); border-right: 1px solid rgba(255,255,255,.08); backdrop-filter: blur(12px); }
  [data-testid="stFileUploaderDropzone"] { border: 1.5px dashed rgba(255,255,255,.28); background: rgba(255,255,255,.04); border-radius: 14px; }
  .doc { display: flex; justify-content: space-between; align-items: center; gap: 8px; font-size: .85rem; color: #dcdcf0;
         padding: 9px 12px; border-radius: 12px; background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.10); overflow: hidden; }
  .doc span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .doc em { color: #9fa0bd; font-style: normal; white-space: nowrap; font-size: .75rem; }

  .stButton > button { border-radius: 14px; border: 1px solid rgba(255,255,255,.14); background: rgba(255,255,255,.06); transition: all .2s; padding: .6rem .8rem; }
  .stButton > button:hover { border-color: #7c5cff; background: rgba(124,92,255,.18); color: #fff; }

  .src { font-size: .85rem; color: #b4b5d0; line-height: 1.5; padding: 10px 12px; border-left: 3px solid #7c5cff;
         background: rgba(255,255,255,.04); border-radius: 0 12px 12px 0; }
  .src code { background: rgba(124,92,255,.22); padding: 2px 9px; border-radius: 999px; color: #dcd2ff; font-size: .78rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'pending' not in st.session_state:
    st.session_state['pending'] = None


def load_vector_store(api_key):
    if 'vector_store' not in st.session_state or st.session_state.get('api_key') != api_key:
        session_id = st.session_state.setdefault('document_session', uuid4().hex)
        directory = str(Path(rag.PERSIST_DIRECTORY) / 'sessions' / session_id)
        st.session_state['vector_store'] = rag.get_vector_store(
            rag.make_embeddings(api_key), persist_directory=directory,
        )
        st.session_state['api_key'] = api_key
    return st.session_state['vector_store']


with st.sidebar:
    st.header('⚙️ Configuração')

    env_key = config('OPENAI_API_KEY', default='')
    typed_key = st.text_input(
        'Chave da OpenAI',
        type='password',
        placeholder='sk-... (ou use o arquivo .env)',
        help='A chave fica só nesta sessão e não é salva em nenhum arquivo.',
    )
    api_key = (typed_key or env_key).strip()

    selected_model = st.selectbox('Modelo', MODEL_OPTIONS)
    top_k = st.slider(
        'Trechos consultados',
        min_value=2, max_value=12, value=6,
        help='Quantos trechos dos documentos são enviados ao modelo a cada pergunta.',
    )

    st.divider()
    st.header('📄 Documentos')

    if not api_key:
        st.info('Informe a chave da OpenAI para começar.')
        vector_store = None
    else:
        try:
            vector_store = load_vector_store(api_key)
        except Exception:
            st.error('Não foi possível abrir a base de documentos. Verifique a configuração local.')
            st.stop()

        uploaded_files = st.file_uploader(
            'Envie PDFs', type=['pdf'], accept_multiple_files=True,
            label_visibility='collapsed',
            key=f"uploads-{st.session_state.get('upload_version', 0)}",
        )
        if uploaded_files:
            for uploaded_file in uploaded_files:
                try:
                    with st.spinner(f'Lendo {uploaded_file.name}...'):
                        added = rag.index_pdf(vector_store, uploaded_file.getvalue(), uploaded_file.name)
                    if added:
                        st.toast(f'{uploaded_file.name}: {added} trechos indexados', icon='✅')
                except ValueError as error:
                    st.warning(str(error))
                except Exception:
                    st.error(f'Não consegui indexar {uploaded_file.name}. Verifique o PDF, a chave e a conexão.')

        sources = rag.list_sources(vector_store)
        if sources:
            for name, count in sources.items():
                col_a, col_b = st.columns([5, 1])
                col_a.markdown(f'<div class="doc"><span>{escape(name)}</span><em>{count} trechos</em></div>',
                               unsafe_allow_html=True)
                if col_b.button('', icon=':material/delete:', key=f'del-{name}', help=f'Remover {name}'):
                    rag.delete_source(vector_store, name)
                    st.session_state['upload_version'] = st.session_state.get('upload_version', 0) + 1
                    st.session_state['messages'] = []
                    st.session_state['pending'] = None
                    st.rerun()
        else:
            st.caption('Nenhum documento indexado ainda.')

    st.divider()
    if st.button('🧹 Limpar conversa', use_container_width=True, disabled=not st.session_state['messages']):
        st.session_state['messages'] = []
        st.rerun()


st.markdown(
    '<div class="hero"><h1>Pergunte aos seus PDFs</h1>'
    '<p>Respostas só com base nos documentos, sempre com a página de origem.</p>'
    f'<span class="chip">{selected_model}</span><span class="chip">{top_k} trechos</span>'
    '<span class="chip">Respostas em tempo real</span></div>',
    unsafe_allow_html=True,
)


def show_sources(sources):
    if not sources:
        return
    with st.expander(f'📚 Fontes ({len(sources)})'):
        for src in sources:
            page = f' · página {src["page"]}' if src['page'] else ''
            st.markdown(f'<div class="src"><code>{escape(src["file"])}</code>{page}<br>{escape(src["snippet"])}</div>',
                        unsafe_allow_html=True)
            st.write('')


has_docs = bool(vector_store and rag.list_sources(vector_store))

if not api_key or not has_docs:
    col1, col2, col3 = st.columns(3)
    col1.markdown('<div class="step"><div class="num">1</div><b>Chave</b><span>Cole sua chave da OpenAI na barra lateral.</span></div>',
                  unsafe_allow_html=True)
    col2.markdown('<div class="step"><div class="num">2</div><b>PDFs</b><span>Envie um ou mais arquivos para indexar.</span></div>',
                  unsafe_allow_html=True)
    col3.markdown('<div class="step"><div class="num">3</div><b>Pergunte</b><span>Receba respostas com as páginas de origem.</span></div>',
                  unsafe_allow_html=True)
    st.stop()


for message in st.session_state['messages']:
    with st.chat_message(message['role'], avatar='🧑‍💻' if message['role'] == 'user' else '🤖'):
        st.markdown(message['content'])
        if message['role'] == 'ai':
            show_sources(message.get('sources'))

question = st.chat_input('Pergunte algo sobre os documentos...') or st.session_state['pending']
st.session_state['pending'] = None

if not st.session_state['messages'] and not question:
    st.caption('Sugestões para começar:')
    columns = st.columns(len(QUICK_QUESTIONS))
    for column, (label, text) in zip(columns, QUICK_QUESTIONS.items()):
        if column.button(label, use_container_width=True):
            st.session_state['pending'] = text
            st.rerun()

if question:
    history = list(st.session_state['messages'])

    with st.chat_message('user', avatar='🧑‍💻'):
        st.markdown(question)

    with st.chat_message('ai', avatar='🤖'):
        try:
            llm = rag.make_llm(selected_model, api_key)
            collected = {'sources': []}

            def generate():
                for kind, value in rag.stream_answer(llm, vector_store, question, history, k=top_k):
                    if kind == 'answer':
                        yield value
                    else:
                        collected['sources'] = value

            response = st.write_stream(generate())
            show_sources(collected['sources'])
        except Exception:
            response, collected = None, {'sources': []}
            st.error('Não consegui responder. Verifique a chave, o acesso ao modelo e a conexão e tente novamente.')

    if response:
        st.session_state['messages'].append({'role': 'user', 'content': question})
        st.session_state['messages'].append({'role': 'ai', 'content': response, 'sources': collected['sources']})
