"""O desafio "clique no vão" não vai para o resolvedor da bola, e a 2ª rodada
ouve o outro provedor.

16/09/2026: o portal passou a servir este desafio na representação — 55 das 57
classificações do dia, sempre "encontre a falha na corrente" ou "clique no ponto
partido da corrente". Duas coisas quebravam:

  - com o fundo animado, a sonda de movimento mandava o captcha para o resolvedor
    da BOLA, que pergunta por animais: "animais=[] tocados=[] confidence=low";
  - 103 respostas no formato, todas do Astra, 39 captchas fechados (~38% por
    tentativa). A rodada 2 repetia o Astra com a mesma imagem.
"""
import json

import pytest

from resolvedor_captcha import solver


@pytest.mark.parametrize("instrucao", [
    "encontre a falha na corrente",
    "clique no ponto partido da corrente",
    "encontre o local que precisa de ser ligado",
    "clique onde falta a ligação entre os elos",
])
def test_enunciado_de_clique_em_ponto_e_reconhecido(instrucao):
    assert solver._e_clique_em_ponto(instrucao)


@pytest.mark.parametrize("instrucao", [
    "clique na flor em que a abelha nunca pousa",
    "selecione os itens feitos principalmente de madeira natural",
    "selecione tudo o que cabe num bolso",
    "",
])
def test_outros_enunciados_nao_sao_clique_em_ponto(instrucao):
    assert not solver._e_clique_em_ponto(instrucao)


def test_a_sonda_de_movimento_nao_vence_o_enunciado():
    """Fundo animado não manda mais este desafio ao resolvedor da bola."""
    import inspect
    fonte = inspect.getsource(solver)
    # Desde 18/09/2026 a regra é mais forte: bola só quando o enunciado pede.
    assert "_area_do_desafio_se_move(page) and _bola_pelo_enunciado(" in fonte


def test_alternancia_por_rodada():
    assert solver._alternar_provedor(1) is None, "rodada 1 mantém a ordem do tipo"
    assert solver._alternar_provedor(2) == 99, "rodada 2 ouve o Gemini primeiro"
    assert solver._alternar_provedor(3) is None
    assert solver._alternar_provedor(4) == 99


class _Resposta:
    text = json.dumps({"x": 10, "y": 20, "confidence": "high", "action": "click"})


class _ClienteGemini:
    def __init__(self):
        self.chamadas = 0

        class _Modelos:
            def generate_content(inner, **_k):
                self.chamadas += 1
                return _Resposta()

        self.models = _Modelos()


@pytest.fixture
def provedores(monkeypatch):
    estado = {"astra": 0, "cliente": _ClienteGemini()}

    def astra(contents, schema, tag, politica):
        estado["astra"] += 1
        return {"x": 1, "y": 2, "confidence": "high"}

    monkeypatch.setattr(solver, "_astra_configurado", lambda: True)
    monkeypatch.setattr(solver, "_astra_call", astra)
    monkeypatch.setattr(solver, "_get_client", lambda _k: estado["cliente"])
    return estado


def test_rodada_impar_continua_perguntando_ao_astra(provedores):
    solver._gemini_call(["x"], {}, "chave", "imagem",
                        rodizio_segundo_provedor=solver._alternar_provedor(1))
    assert provedores["astra"] == 1
    assert provedores["cliente"].chamadas == 0


def test_rodada_par_pergunta_ao_gemini_primeiro(provedores):
    solver._gemini_call(["x"], {}, "chave", "imagem",
                        rodizio_segundo_provedor=solver._alternar_provedor(2))
    assert provedores["cliente"].chamadas == 1
    assert provedores["astra"] == 0


def test_a_memoria_de_falha_do_gemini_ainda_vence_a_alternancia(provedores):
    """Se ele falhou DE VERDADE neste captcha, não adianta insistir."""
    solver._marcar_gemini_nao_fechou("imagem")
    solver._gemini_call(["x"], {}, "chave", "imagem",
                        rodizio_segundo_provedor=solver._alternar_provedor(2))
    assert provedores["astra"] == 1
    assert provedores["cliente"].chamadas == 0


@pytest.mark.parametrize("funcao", ["_solve_grade_fused", "_solve_imagem"])
def test_os_dois_resolvedores_de_ponto_alternam(funcao):
    import inspect
    fonte = inspect.getsource(getattr(solver, funcao))
    assert "rodizio_segundo_provedor=_alternar_provedor(rnd)" in fonte, funcao
