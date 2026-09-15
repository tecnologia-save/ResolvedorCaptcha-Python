"""Erro ou imprecisão — se o Gemini não fechou, entra o segundo goleiro."""
import inspect

from resolvedor_captcha import solver


def test_resposta_INUTIL_vai_ao_segundo_provedor():
    """Até 11/09/2026 o astra só entrava quando havia ERRO de chamada.

    Resposta com `confidence=low` ou sem tiles voltava para o Gemini com
    outro modelo — e o log dizia "retentando Gemini". Do ponto de vista de
    quem espera a resposta, erro e imprecisão são a mesma coisa: ele não
    fechou.

    Girar modelo dentro do mesmo provedor muda pouco. Com `temperature=0.0` o
    Gemini tende a repetir a própria resposta; quem muda de verdade é trocar
    de PROVEDOR.
    """
    fonte = inspect.getsource(solver._solve_grade)
    assert "direto_ao_segundo=(attempt > 1)" in fonte
    assert "retentando Gemini" not in fonte
    assert "indo ao segundo" in fonte


def test_o_atalho_pula_o_gemini_de_verdade():
    """`direto_ao_segundo` não pode ser só um rótulo: tem de sair antes do
    laço de modelos, senão o Gemini é chamado assim mesmo."""
    fonte = inspect.getsource(solver._gemini_call)
    # `not astra_recusou` desde 15/09/2026: quem acabou de recusar não é
    # chamado de novo na mesma chamada — ver test_astra_que_recusa_devolve_ao_gemini.
    i = fonte.index("if direto_ao_segundo and not astra_recusou and _astra_configurado():")
    j = fonte.index("for mi, model in enumerate(ativos):")
    assert i < j, "o atalho vem antes do laço de modelos"
    trecho = fonte[i:j]
    assert "resposta = _astra_call(" in trecho
    assert trecho.index("resposta = _astra_call(") < trecho.index("return resposta")


def test_sem_segundo_provedor_o_atalho_nao_se_aplica():
    """Sem astra configurado não há goleiro reserva — o caminho normal segue
    valendo, senão a resposta ruim viraria falha sem alternativa nenhuma."""
    fonte = inspect.getsource(solver._gemini_call)
    i = fonte.index("if direto_ao_segundo")
    assert "_astra_configurado()" in fonte[i:i + 120]


def test_um_erro_de_chamada_tambem_troca_de_provedor():
    """A outra metade da mesma política, medida na RUN-385b9699: três modelos
    do Gemini queimaram 45s disputando entre si e o astra foi recusado por
    0,8s."""
    assert solver.MODELOS_GEMINI_ANTES_DO_SEGUNDO == 1
