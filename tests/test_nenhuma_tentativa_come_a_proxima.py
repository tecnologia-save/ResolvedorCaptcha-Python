"""Uma requisição pendurada não pode inviabilizar todas as outras."""
import time

from resolvedor_captcha.solver import (
    ASTRA_DEADLINE_MIN_S,
    GEMINI_DEADLINE_MIN_MS,
    PoliticaLatencia,
)


def _politica(orcamento_s, teto_ms=40_000):
    return PoliticaLatencia(timeout_ms=teto_ms, fim=time.monotonic() + orcamento_s)


def test_o_cenario_da_PREMIUM_TEXTIL_passa_a_ter_segunda_chance():
    """RUN-ef3f4b9d, 11/09/2026, orçamento 55s e teto de 40s para grade.

        Teto por chamada ajustado ao tipo: 10s -> 40s (grade)
        falha na chamada | gemini-3.5-flash-lite | ReadTimeout
        parando a cadeia do Gemini com 12s

    Uma requisição pendurada levou os 40. Os 12 restantes não cabem em
    ninguém: o mínimo viável são 10s para o Gemini e 10s para o segundo
    provedor. O captcha era "clique em todos os objetos feitos principalmente
    de metal", com dois baldes óbvios — nenhum modelo chegou a ver a imagem, e
    a empresa foi marcada como "exige validação manual".
    """
    p = _politica(55)
    primeira = p.timeout_efetivo_ms(preservar_retentativa=True)
    assert primeira < 40_000, "o teto do tipo não pode valer sozinho"

    # A pendurada consome o teto inteiro. O que sobra ainda serve?
    sobra_ms = 55_000 - primeira
    assert sobra_ms >= GEMINI_DEADLINE_MIN_MS + ASTRA_DEADLINE_MIN_S * 1000, (
        "depois da primeira travar, ainda tem de caber Gemini E astra")


def test_o_teto_nao_corta_chamada_BOA():
    """12s foi a primeira tentativa de corte, em 26/08/2026, e era cedo demais:
    uma chamada boa levou 19,8s na mesma medição. O teto fica logo ACIMA da
    pior resposta boa observada, nunca abaixo."""
    assert _politica(55).timeout_efetivo_ms(preservar_retentativa=True) > 19_800


def test_a_ultima_chance_usa_tudo():
    """Sem próxima para proteger, reservar é só desperdiçar."""
    p = _politica(30)
    assert (p.timeout_efetivo_ms(preservar_retentativa=False)
            > p.timeout_efetivo_ms(preservar_retentativa=True))


def test_sem_orcamento_total_nada_muda():
    """`fim=None` é o consumidor sem prazo — o comportamento de sempre."""
    p = PoliticaLatencia(timeout_ms=20_000, fim=None)
    assert p.timeout_efetivo_ms(preservar_retentativa=True) == 20_000


def test_a_fracao_se_ajusta_ao_orcamento_de_cada_consumidor():
    """Metade, e não um número fixo: o login tem 300s de orçamento e a
    representação 55s. Um fixo bom para um é ruim para o outro."""
    assert (_politica(300).timeout_efetivo_ms(preservar_retentativa=True)
            > _politica(55).timeout_efetivo_ms(preservar_retentativa=True))


def test_a_reserva_nao_fabrica_chamada_condenada():
    """A fração não pode empurrar abaixo do piso de quem vai ser chamado.

    Um teste que já existia — a run real reproduzida, orçamento 30s e teto de
    10s — mostrou a terceira chamada caindo para 5s com a primeira versão
    desta reserva. Cinco segundos é abaixo dos 10s que a API do Gemini exige
    ("Manually set deadline 5s is too short"): a reserva fabricaria exatamente
    a chamada condenada que veio eliminar.

    Quando não cabem os dois, a preferência é a tentativa de AGORA — ela tem
    uma imagem fresca na mão, e a próxima ainda pode nem acontecer.
    """
    p = _politica(10, teto_ms=10_000)   # sobra 10s, teto 10s, metade = 5s
    assert p.timeout_efetivo_ms(
        preservar_retentativa=True, piso_ms=GEMINI_DEADLINE_MIN_MS) == 10_000


def test_com_folga_a_reserva_continua_valendo():
    """O piso é limite inferior, não desligamento: com orçamento grande a
    fração segue mandando."""
    p = _politica(55)
    assert p.timeout_efetivo_ms(
        preservar_retentativa=True, piso_ms=GEMINI_DEADLINE_MIN_MS) < 40_000
