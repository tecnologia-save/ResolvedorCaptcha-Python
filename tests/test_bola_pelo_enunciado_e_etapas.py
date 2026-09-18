"""Bola só quando o enunciado pede, e o preparo da rodada tem nome.

18/09/2026: o portal trocou o desafio da representação para "o objeto que
consegue rolar numa superfície plana". Não estava na lista de clique em ponto, e
duas empresas caíram no resolvedor da bola quando o fundo estava animado. O log
também passou a mostrar "Astra respondeu em 4s · preparo e espera 9s" sem dizer
onde iam os 9s.
"""
import inspect

import pytest

from resolvedor_captcha import solver


@pytest.mark.parametrize("instrucao", [
    "clique na bola que nunca toca a linha",
    "clique na flor em que a abelha nunca pousa",
    "selecione o animal que não toca o chão",
])
def test_mecanica_de_sequencia_vai_para_a_bola(instrucao):
    assert solver._bola_pelo_enunciado(instrucao)


@pytest.mark.parametrize("instrucao", [
    "identifique o objeto que consegue rolar numa superfície plana",
    "selecione o objeto que consegue rolar numa superfície plana",
    "encontre a falha na corrente",
    "encontre o local que precisa de ser ligado",
])
def test_desafio_novo_nao_vai_para_a_bola_so_por_animacao(instrucao):
    """Não precisa estar em lista nenhuma: basta não pedir a mecânica da bola."""
    assert not solver._bola_pelo_enunciado(instrucao)


def test_enunciado_ilegivel_mantem_o_comportamento_antigo():
    assert solver._bola_pelo_enunciado("")
    assert solver._bola_pelo_enunciado(None)


def test_a_classificacao_usa_a_regra_nova():
    assert "_area_do_desafio_se_move(page) and _bola_pelo_enunciado(" in inspect.getsource(solver)


def test_etapas_contam_o_tempo_de_cada_passo(monkeypatch):
    agora = [0.0]
    monkeypatch.setattr(solver.time, "monotonic", lambda: agora[0])
    solver._ULTIMA_CHAMADA.update(provedor="Astra", segundos=4.0)
    etapas = solver._Etapas()
    agora[0] = 1.5
    etapas.marcar("enunciado")
    agora[0] = 4.0
    etapas.marcar("captura")
    agora[0] = 10.0
    etapas.marcar("chamada")
    assert etapas.texto() == "preparo: enunciado 1.5s, captura 2.5s, montagem da chamada 2.0s"


def test_etapas_sem_marcas_nao_escrevem_nada():
    assert solver._Etapas().texto() == ""


@pytest.mark.parametrize("funcao", ["_solve_imagem", "_solve_grade_fused"])
def test_a_resposta_da_rodada_traz_o_preparo(funcao):
    fonte = inspect.getsource(getattr(solver, funcao))
    assert "etapas = _Etapas()" in fonte
    assert "{etapas.texto()}" in fonte
    assert 'etapas.marcar("chamada")' in fonte
