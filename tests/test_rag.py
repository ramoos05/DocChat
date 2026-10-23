import io

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from reportlab.pdfgen import canvas

import rag


def make_pdf(pages):
    """Cria um PDF em memória com um texto por página."""
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer)
    for text in pages:
        pdf.drawString(72, 750, text)
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


@pytest.fixture
def store(tmp_path):
    return rag.get_vector_store(DeterministicFakeEmbedding(size=32), str(tmp_path / 'db'))


@pytest.fixture
def llm():
    return FakeListChatModel(responses=['Resposta de teste'])


def test_process_pdf_marca_arquivo_e_pagina():
    pdf = make_pdf(['O projeto usa Python.', 'O banco de dados e o PostgreSQL.'])
    chunks = rag.process_pdf(pdf, 'projeto.pdf')

    assert len(chunks) == 2
    assert {c.metadata['source'] for c in chunks} == {'projeto.pdf'}
    assert sorted(c.metadata['page'] for c in chunks) == [0, 1]


def test_index_pdf_nao_duplica_arquivo_ja_indexado(store):
    pdf = make_pdf(['Texto qualquer do documento.'])

    assert rag.index_pdf(store, pdf, 'a.pdf') == 1
    assert rag.index_pdf(store, pdf, 'a.pdf') == 0
    assert rag.list_sources(store) == {'a.pdf': 1}


def test_pdf_sem_texto_gera_erro_claro(store):
    pdf = make_pdf([''])
    with pytest.raises(ValueError):
        rag.index_pdf(store, pdf, 'escaneado.pdf')


def test_delete_source_remove_so_o_arquivo_pedido(store):
    rag.index_pdf(store, make_pdf(['Primeiro documento.']), 'a.pdf')
    rag.index_pdf(store, make_pdf(['Segundo documento.']), 'b.pdf')

    assert rag.delete_source(store, 'a.pdf') == 1
    assert list(rag.list_sources(store)) == ['b.pdf']


def test_resposta_traz_fontes_com_pagina(store, llm):
    rag.index_pdf(store, make_pdf(['Pagina um.', 'Pagina dois.']), 'manual.pdf')

    text, sources = rag.answer(llm, store, 'O que diz o manual?', history=[], k=4)

    assert text == 'Resposta de teste'
    assert {s['file'] for s in sources} == {'manual.pdf'}
    assert {s['page'] for s in sources} == {1, 2}


def test_historico_com_chaves_nao_quebra_o_prompt(store, llm):
    rag.index_pdf(store, make_pdf(['Conteudo do documento.']), 'a.pdf')
    history = [
        {'role': 'user', 'content': 'Mostre um JSON'},
        {'role': 'ai', 'content': 'Claro: {"nome": "teste", "valor": {x}}'},
    ]

    text, _ = rag.answer(llm, store, 'E o segundo campo?', history=history, k=2)

    assert text


def test_pergunta_atual_nao_entra_duas_vezes_no_historico():
    history = [{'role': 'user', 'content': 'oi'}, {'role': 'ai', 'content': 'olá'}]
    messages = rag._to_messages(history)

    assert [m.type for m in messages] == ['human', 'ai']


@pytest.mark.parametrize('k', [0, -1, True, 1.5])
def test_quantidade_de_trechos_invalida(store, llm, k):
    with pytest.raises(ValueError, match='inteiro positivo'):
        rag.build_chain(llm, store, k=k)


def test_historico_rejeita_papel_desconhecido():
    with pytest.raises(ValueError, match='Papel de mensagem inválido'):
        rag._to_messages([{'role': 'system', 'content': 'Instrução inesperada'}])


def test_bases_de_sessoes_nao_compartilham_documentos(tmp_path):
    embeddings = DeterministicFakeEmbedding(size=32)
    first = rag.get_vector_store(embeddings, str(tmp_path / 'first'))
    second = rag.get_vector_store(embeddings, str(tmp_path / 'second'))
    rag.index_pdf(first, make_pdf(['Documento privado.']), 'privado.pdf')

    assert rag.list_sources(first) == {'privado.pdf': 1}
    assert rag.list_sources(second) == {}


def test_fontes_nao_repetem_a_mesma_pagina():
    from langchain_core.documents import Document

    docs = [
        Document(page_content='Primeiro trecho.', metadata={'source': 'a.pdf', 'page': 0}),
        Document(page_content='Outro trecho.', metadata={'source': 'a.pdf', 'page': 0}),
    ]
    assert rag.format_sources(docs) == [
        {'file': 'a.pdf', 'page': 1, 'snippet': 'Primeiro trecho.'},
    ]
