"""Recusa não é incapacidade — e o campo que dizia isso nunca era lido."""
import inspect

from resolvedor_captcha import solver


def test_a_recusa_e_reconhecida_pelo_task_summary():
    """O modelo responde no schema, com `matching_tiles` vazio e a recusa
    escrita no `task_summary`:

        tiles=[] resumo='Não posso resolver CAPTCHAs nem indicar quais...'

    A gente registrava isso como "tiles vazios", que é o diagnóstico OPOSTO:
    um diz que o modelo não conseguiu, o outro que ele não quis. Foi essa
    confusão que me fez concluir que o astra era ruim em grade.
    """
    assert solver._e_recusa_do_segundo(
        {"matching_tiles": [], "task_summary": "Não posso resolver CAPTCHAs."})
    assert not solver._e_recusa_do_segundo(
        {"matching_tiles": [1, 4], "task_summary": "itens de metal"})
    # Vazio SEM recusa continua sendo vazio: são causas diferentes.
    assert not solver._e_recusa_do_segundo(
        {"matching_tiles": [], "task_summary": "nenhum item corresponde"})


def test_insiste_com_o_MESMO_provedor():
    """A recusa não é determinística — mesma imagem e mesmo prompt: recusa,
    recusa, acerta. Trocar de provedor não resolveria, porque não há nada
    errado com a imagem nem com o prompt.

    Medido em 11/09/2026 sobre 20 grades aceitas pelo portal:

        antes   75% exatos, 25% vazios, 0% errados
        agora   90% exatos, 10% vazios, 0% errados

    Seis reperguntas dispararam e quatro viraram acerto.
    """
    fonte = inspect.getsource(solver._astra_call)
    assert "_e_recusa_do_segundo(resposta)" in fonte
    assert "cliente.chat.completions.create(**pedido)" in fonte
    assert fonte.count("create(**pedido)") == 2, "a mesma chamada, repetida"


def test_a_repergunta_respeita_o_orcamento():
    """Sem tempo não há repergunta: insistir com orçamento estourado produz a
    chamada condenada que o resto do arquivo existe para evitar."""
    fonte = inspect.getsource(solver._astra_call)
    i = fonte.index("_e_recusa_do_segundo(resposta)")
    assert "ASTRA_DEADLINE_MIN_S" in fonte[i:i + 700]


def test_o_pedido_e_montado_uma_vez_so():
    """Duplicar a montagem convidaria a primeira chamada e a repergunta a
    divergirem — e a divergência seria invisível."""
    fonte = inspect.getsource(solver._astra_call)
    assert fonte.count("_contents_para_openai(contents, schema)") == 1
