"""Timeout não custa o mesmo que 503, e o descanso tem de refletir isso."""
import inspect
import time

from resolvedor_captcha import solver
from resolvedor_captcha.solver import (
    CATEGORIA_CARA,
    DEGRAU_FALHA_CARA,
    _DESCANSOS,
)


def _limpar():
    solver._BANCO.clear()


def test_timeout_na_PRIMEIRA_falha_ja_pula_uma_empresa():
    """RUN-9634657b, BACUTIA COMERCIAL, 11/09/2026:

        19:40:08  'gemini-3.5-flash-lite' descansa 1min (1ª falha: ReadTimeout)
        19:40:12  segundo provedor resolveu
        19:42:25  novo captcha — flash-lite já voltou e joga PRIMEIRO
        19:42:52  falha na chamada | gemini-3.5-flash-lite | ReadTimeout
        19:43:16  solver não resolveu — empresa para validação manual

    21 dos 55s do orçamento foram para um modelo que tinha acabado de pendurar
    uma chamada. Uma empresa leva ~8min: com 1min de descanso o modelo volta
    SEMPRE a tempo de pegar o próximo captcha, que é o caso exato acima.
    """
    _limpar()
    solver._penalizar("m", "ReadTimeout", categoria=CATEGORIA_CARA)
    falta = solver._voltar_em("m") - time.monotonic()
    assert falta > 60, f"descanso de {falta:.0f}s deixa o modelo pegar o próximo captcha"
    assert falta <= _DESCANSOS[DEGRAU_FALHA_CARA - 1] + 1


def test_falha_BARATA_continua_no_primeiro_degrau():
    """Um 503 volta em ~1s. Reinsistir nele é barato, e endurecer aqui só
    esvaziaria a bancada — erro que este arquivo já cometeu na direção oposta.
    """
    _limpar()
    solver._penalizar("m", "503", categoria="indisponivel")
    falta = solver._voltar_em("m") - time.monotonic()
    assert falta <= _DESCANSOS[0] + 1


def test_sem_categoria_o_comportamento_e_o_de_sempre():
    """A sondagem de startup chama sem categoria, e o comentário dela diz por
    quê: um pico ali não pode custar o modelo mais rápido pela execução inteira.
    """
    _limpar()
    solver._penalizar("m", "não respondeu à sondagem")
    falta = solver._voltar_em("m") - time.monotonic()
    assert falta <= _DESCANSOS[0] + 1


def test_um_acerto_ainda_zera_a_ficha():
    """Não é banimento. Se fosse, um pico tiraria o modelo primário da execução
    inteira — e a alternativa medida é 0,5s mais lenta, não melhor.
    """
    _limpar()
    solver._penalizar("m", "ReadTimeout", categoria=CATEGORIA_CARA)
    solver._premiar("m")
    assert solver._pode_jogar("m")
    assert "m" not in solver._BANCO


def test_a_categoria_chega_de_verdade_ao_banco():
    """O defeito recorrente deste código é o valor que se calcula e não se usa:
    `_penalizar` aceitar `categoria` e o único chamador real não passar nada
    deixaria todos os testes acima verdes com a produção inalterada.
    """
    fonte = inspect.getsource(solver._gemini_call)
    assert "categoria=_categoria_do_erro(last_exc)" in fonte


def test_o_degrau_maior_nao_apaga_a_contagem_real():
    """O log diz "2ª falha seguida" e o operador conta em cima disso. O degrau
    do descanso pode ser maior que a contagem; a contagem não pode mentir.
    """
    _limpar()
    solver._penalizar("m", "ReadTimeout", categoria=CATEGORIA_CARA)
    assert solver._BANCO["m"][0] == 1, "a 1ª falha continua sendo a 1ª"
    solver._penalizar("m", "ReadTimeout", categoria=CATEGORIA_CARA)
    assert solver._BANCO["m"][0] == 2
    falta = solver._voltar_em("m") - time.monotonic()
    assert falta <= _DESCANSOS[1] + 1, (
        "a 2ª falha cara não pode saltar para o degrau de 15min")
