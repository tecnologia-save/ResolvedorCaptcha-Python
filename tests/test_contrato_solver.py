"""Contrato do solver que ja existia — a rede de seguranca da correcao.

Estes testes nao descrevem comportamento novo: descrevem o que o solver ja fazia
antes do freshness guard, para que a correcao possa ser feita sem quebrar a
deteccao de frame ativo, o fallback de modelo ou o clique nos tiles.

Sem hCaptcha real, sem Gemini real, sem rede.
"""
import json

import pytest
from fakes import Captcha, Desafio, FakePage

from resolvedor_captcha import solver

# ── Deteccao do frame ATIVO entre varios pre-carregados ──────────────────────

def test_encontra_o_frame_ativo_e_ignora_os_pre_carregados(page, captcha):
    """O hCaptcha deixa varios `frame=challenge` no DOM; so um vale."""
    assert captcha.n_iframes == 2
    frame = solver._get_challenge_frame(page)
    assert frame is not None
    assert solver._get_active_iframe_index(page) == captcha.idx_ativo


def test_sem_desafio_ativo_nao_ha_frame(page, captcha):
    captcha.resolver()
    assert solver._get_challenge_frame(page) is None
    assert solver._challenge_visible(page) is False


def test_challenge_visible_segue_o_estado_do_captcha(page, captcha):
    assert solver._challenge_visible(page) is True
    captcha.resolver()
    assert solver._challenge_visible(page) is False


# ── Classificacao de erro do provedor ────────────────────────────────────────

@pytest.mark.parametrize("mensagem", [
    "503 UNAVAILABLE", "model is overloaded", "429 RESOURCE_EXHAUSTED",
    "404 model not found", "no longer available",
])
def test_erros_que_justificam_trocar_de_modelo(mensagem):
    assert solver._is_overloaded_error(Exception(mensagem)) is True


@pytest.mark.parametrize("mensagem", [
    "400 INVALID_ARGUMENT", "401 unauthorized", "API key not valid",
])
def test_erros_que_nao_justificam_trocar_de_modelo(mensagem):
    assert solver._is_overloaded_error(Exception(mensagem)) is False


# ── Fallback entre modelos ───────────────────────────────────────────────────

class _Resposta:
    def __init__(self, texto):
        self.text = texto


def _cliente(comportamento):
    """Cliente falso do google.genai. `comportamento` mapeia modelo -> acao."""
    chamadas = []

    class _Models:
        def generate_content(self, model, contents, config):
            chamadas.append(model)
            acao = comportamento.get(model)
            if isinstance(acao, BaseException):
                raise acao
            return _Resposta(json.dumps(acao if acao is not None else {"ok": True}))

    class _Cliente:
        models = _Models()

    return _Cliente(), chamadas


def test_sucesso_no_primeiro_modelo_nao_tenta_os_demais(monkeypatch):
    cliente, chamadas = _cliente({solver.GEMINI_MODELS[0]: {"ok": 1}})
    monkeypatch.setattr(solver, "_get_client", lambda _k: cliente)
    assert solver._gemini_call([], {}, "k", "grade") == {"ok": 1}
    assert chamadas == [solver.GEMINI_MODELS[0]]


def test_modelo_sobrecarregado_cai_para_o_proximo(monkeypatch):
    cliente, chamadas = _cliente({
        solver.GEMINI_MODELS[0]: RuntimeError("503 UNAVAILABLE"),
        solver.GEMINI_MODELS[1]: {"ok": 2},
    })
    monkeypatch.setattr(solver, "_get_client", lambda _k: cliente)
    assert solver._gemini_call([], {}, "k", "grade") == {"ok": 2}
    assert chamadas[0] == solver.GEMINI_MODELS[0]
    assert solver.GEMINI_MODELS[1] in chamadas


def test_erro_nao_sobrecarga_nao_percorre_todos_os_modelos(monkeypatch):
    """Chave invalida nao melhora trocando de modelo — e nao deve circular."""
    cliente, chamadas = _cliente({m: RuntimeError("400 INVALID_ARGUMENT")
                                  for m in solver.GEMINI_MODELS})
    monkeypatch.setattr(solver, "_get_client", lambda _k: cliente)
    with pytest.raises(RuntimeError):
        solver._gemini_call([], {}, "k", "grade")
    assert set(chamadas) == {solver.GEMINI_MODELS[0]}


def test_todos_indisponiveis_levanta(monkeypatch):
    cliente, _ = _cliente({m: RuntimeError("503 UNAVAILABLE")
                           for m in solver.GEMINI_MODELS})
    monkeypatch.setattr(solver, "_get_client", lambda _k: cliente)
    with pytest.raises(RuntimeError):
        solver._gemini_call([], {}, "k", "grade")


# ── Clique nos tiles ─────────────────────────────────────────────────────────

def test_clica_os_tiles_pedidos_uma_vez_cada(page, captcha):
    solver._click_grade_tiles(page, [4, 0, 4, 8])
    assert [idx for _d, idx in captcha.tiles_clicados] == [0, 4, 8]


def test_lista_vazia_nao_clica(page, captcha):
    solver._click_grade_tiles(page, [])
    assert captcha.tiles_clicados == []


def test_clique_fused_usa_o_centro_de_cada_celula(page, captcha):
    bbox = {"x": 0.0, "y": 0.0, "width": 300.0, "height": 300.0}
    solver._click_fused_grade_tiles(page, [0, 8], bbox)
    coords = [(x, y) for _d, x, y in captcha.cliques_pixel]
    assert coords == [(50.0, 50.0), (250.0, 250.0)]


# ── Caminho feliz completo ───────────────────────────────────────────────────

def test_solve_grade_clica_e_submete_quando_o_desafio_nao_muda(
        page, captcha, gemini):
    """O caminho que precisa continuar funcionando depois da correcao."""
    gemini["resposta"] = {"task_summary": "onibus", "matching_tiles": [1, 3],
                          "confidence": "high"}
    solver._solve_grade(page, "chave-de-teste", max_rounds=1)
    assert [idx for _d, idx in captcha.tiles_clicados] == [1, 3]
    assert captcha.submits >= 1
    # E clicou no desafio CERTO — o mesmo objeto que gerou a captura.
    assert all(d is captcha.desafio for d, _i in captcha.tiles_clicados)


def test_confianca_baixa_nao_clica(page, captcha, gemini):
    gemini["resposta"] = {"task_summary": "?", "matching_tiles": [1],
                          "confidence": "low"}
    solver._solve_grade(page, "chave-de-teste", max_rounds=1)
    assert captcha.tiles_clicados == []


def test_desafio_ja_resolvido_encerra_sem_clicar(page, captcha, gemini):
    captcha.resolver()
    assert solver._solve_grade(page, "chave-de-teste", max_rounds=1) is True
    assert captcha.tiles_clicados == []
    assert gemini["chamadas"] == 0


# ── Utilitario de texto ──────────────────────────────────────────────────────

def test_limpar_texto_colapsa_e_trunca():
    assert solver._limpar_texto("a\n\n   b") == "a b"
    assert solver._limpar_texto("x" * 300, max_len=10) == "x" * 10 + "…"


def test_fake_recusa_js_nao_modelado(captcha):
    """A honestidade do duble: script desconhecido falha alto."""
    frame = FakePage(captcha).frames[captcha.idx_ativo]
    with pytest.raises(AssertionError):
        frame.evaluate("() => document.querySelector('.inventado')")


def test_desafios_distintos_tem_pixels_distintos():
    """Premissa do freshness guard, registrada como contrato do duble."""
    a, b = Desafio(pixels=b"A"), Desafio(pixels=b"B")
    assert a.pixels != b.pixels
    c = Captcha(a)
    c.trocar_desafio(b)
    assert c.desafio is b and c.historico == [a, b]


# ── Deteccao publica, sem resolucao ──────────────────────────────────────────

def test_captcha_presente_detecta_desafio_aberto(page, captcha):
    assert solver.captcha_presente(page) is True


def test_captcha_presente_e_falso_quando_nao_ha_desafio(page, captcha):
    captcha.resolver()
    captcha.n_iframes = 0
    assert solver.captcha_presente(page) is False


def test_captcha_presente_detecta_o_widget_checkbox_fechado(page, captcha):
    """Desafio ainda nao aberto tambem e' fluxo parado esperando alguem."""
    captcha.resolver()             # nenhum frame=challenge ativo
    captcha.checkbox_presente = 1
    assert solver.captcha_presente(page) is True


def test_captcha_presente_nao_chama_o_solver(page, captcha, monkeypatch):
    """DETECCAO nao pode virar resolucao: nada de gastar chamada ao modelo
    nem clicar tile que ninguem pediu."""
    def proibido(*_a, **_k):
        raise AssertionError("captcha_presente nao pode resolver")

    monkeypatch.setattr(solver, "solve_hcaptcha", proibido)
    monkeypatch.setattr(solver, "_gemini_grade", proibido)
    monkeypatch.setattr(solver, "_click_grade_tiles", proibido)
    solver.captcha_presente(page)
    assert captcha.tiles_clicados == [] and captcha.submits == 0


def test_captcha_presente_nunca_levanta(monkeypatch):
    """Quem pergunta esta num estado incerto; excecao aqui vira ruido."""
    class _PaginaQuebrada:
        frames = property(lambda self: (_ for _ in ()).throw(RuntimeError("x")))

        def locator(self, _s):
            raise RuntimeError("x")

    assert solver.captcha_presente(_PaginaQuebrada()) is False


def test_detector_esta_na_api_publica():
    import resolvedor_captcha
    assert "captcha_presente" in resolvedor_captcha.__all__
    assert resolvedor_captcha.captcha_presente is solver.captcha_presente
