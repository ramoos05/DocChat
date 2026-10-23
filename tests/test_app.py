from pathlib import Path

from langchain_core.embeddings import DeterministicFakeEmbedding
from streamlit.testing.v1 import AppTest

import rag


APP_PATH = str(Path(__file__).resolve().parents[1] / 'app.py')


def test_interface_sem_chave_nao_abre_base(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', '')
    app = AppTest.from_file(APP_PATH).run(timeout=15)

    assert not app.exception
    assert 'Informe a chave da OpenAI para começar.' in [item.value for item in app.info]
    assert 'vector_store' not in app.session_state


def test_interface_isola_documentos_por_sessao(monkeypatch, tmp_path):
    monkeypatch.setenv('OPENAI_API_KEY', 'test-placeholder')
    monkeypatch.setattr(rag, 'PERSIST_DIRECTORY', str(tmp_path))
    monkeypatch.setattr(rag, 'make_embeddings', lambda key: DeterministicFakeEmbedding(size=32))

    first = AppTest.from_file(APP_PATH).run(timeout=15)
    second = AppTest.from_file(APP_PATH).run(timeout=15)

    assert not first.exception
    assert not second.exception
    assert first.session_state['document_session'] != second.session_state['document_session']
    first_store = first.session_state['vector_store']
    first.run(timeout=15)
    assert first.session_state['vector_store'] is first_store
