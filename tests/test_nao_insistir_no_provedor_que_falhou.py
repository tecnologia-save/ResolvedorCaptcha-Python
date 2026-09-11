"""Insistir no provedor que acabou de falhar é a ordem errada."""
import time

from resolvedor_captcha import solver
from resolvedor_captcha.solver import (
    ASTRA_DEADLINE_MIN_S,
    GEMINI_DEADLINE_MIN_MS,
    MODELOS_GEMINI_ANTES_DO_SEGUNDO,
    PoliticaLatencia,
)

RESERVA_MS = int(ASTRA_DEADLINE_MIN_S * 1000)


def test_um_modelo_e_o_bastante_antes_de_trocar_de_PROVEDOR():
    """Medido em 11/09/2026, LEONARDO VIEIRA RESTAURANTE:

        'gemini-3.5-flash'      descansa 1min (ReadTimeout)
        'gemini-3.1-flash-lite' descansa 1min (ReadTimeout)
        segundo provedor NAO chamado: restam 9.2s e o minimo viavel e 10s

    O astra foi recusado por oito décimos de segundo. Os dois modelos do
    Gemini consumiram o orçamento disputando entre si, e quem podia responder
    não foi perguntado. A empresa terminou como "exige validação manual".
    """
    assert MODELOS_GEMINI_ANTES_DO_SEGUNDO == 1

    import inspect
    fonte = inspect.getsource(solver._gemini_call)
    assert "mi >= MODELOS_GEMINI_ANTES_DO_SEGUNDO" in fonte
    # Só quando há para quem ir: sem segundo provedor a rotação é tudo o que
    # existe, e encurtar tiraria tentativa sem dar nada em troca.
    i = fonte.index("mi >= MODELOS_GEMINI_ANTES_DO_SEGUNDO")
    assert "reserva_ms and" in fonte[max(0, i - 60):i]


def test_a_reserva_sai_do_TETO_e_nao_so_da_decisao_de_continuar():
    """O guard de reserva roda uma vez por MODELO; dentro de cada um cabem duas
    tentativas, e elas gastavam para dentro da reserva. Por isso a decisão de
    parar chegava certa e tarde — com 9,2s para um provedor que precisa de 10.
    """
    p = PoliticaLatencia(timeout_ms=40_000, fim=time.monotonic() + 30)
    com = p.timeout_efetivo_ms(piso_ms=GEMINI_DEADLINE_MIN_MS,
                               reserva_ms=RESERVA_MS)
    sem = p.timeout_efetivo_ms(piso_ms=GEMINI_DEADLINE_MIN_MS)
    assert com < sem, "a reserva tem de encolher o teto desta chamada"
    assert com <= 30_000 - RESERVA_MS + 1


def test_a_reserva_nao_empurra_abaixo_do_minimo_da_API():
    """Mesma regra do `piso_ms`: guardar para o outro não pode inviabilizar
    esta chamada, que já tem a imagem fresca na mão."""
    p = PoliticaLatencia(timeout_ms=40_000, fim=time.monotonic() + 12)
    assert p.timeout_efetivo_ms(piso_ms=GEMINI_DEADLINE_MIN_MS,
                                reserva_ms=RESERVA_MS) >= GEMINI_DEADLINE_MIN_MS


def test_sem_reserva_nada_muda():
    """`reserva_ms=0` é o caminho de quem não tem segundo provedor."""
    p = PoliticaLatencia(timeout_ms=20_000, fim=time.monotonic() + 30)
    assert (p.timeout_efetivo_ms(reserva_ms=0)
            == p.timeout_efetivo_ms())
