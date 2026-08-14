"""Freshness guard — uma resposta so pode clicar no desafio que a originou.

Defeito confirmado em producao de QA na AUT-0076:

    captura A -> Gemini analisa A -> hCaptcha muda para B
              -> resposta de A chega -> solver clica os indices de A em B.

A janela e larga porque uma unica chamada ao modelo chegou a levar ~2 minutos
(503 no primeiro modelo, fallback no segundo). Nesse intervalo o hCaptcha troca
o desafio sozinho — reload manual so torna o defeito mais facil de reproduzir,
nao e condicao para ele.

O que torna o bug SILENCIOSO: `_click_grade_tiles` recria o locator a cada
chamada, e locators do Playwright resolvem na hora do clique. O locator "fresco"
aplica os indices velhos a grade nova sem levantar exececao alguma. Por isso
nenhum destes testes aceita "o locator resolveu" como prova de identidade — a
prova e o OBJETO `Desafio` em que o clique caiu.

Regra: na duvida, nao clicar. Falha segura e preferivel a clique errado.
"""
import pytest
from fakes import CAIXA_PADRAO, Desafio, FakePage

from resolvedor_captcha import solver

RESPOSTA_A = {"task_summary": "onibus", "matching_tiles": [0, 4, 8],
              "confidence": "high"}


def _desafio_b():
    return Desafio(prompt="selecione todas as imagens com barco",
                   pixels=b"PNG-DESAFIO-B")


# ── A · desafio trocou enquanto o modelo pensava ─────────────────────────────

def test_a_desafio_mudou_durante_a_analise_zero_cliques(page, captcha, gemini):
    original = captcha.desafio
    gemini["resposta"] = RESPOSTA_A
    gemini["ao_chamar"] = lambda: captcha.trocar_desafio(_desafio_b())

    solver._solve_grade(page, "chave-de-teste", max_rounds=1)

    assert captcha.tiles_clicados == []
    assert captcha.cliques_pixel == []
    assert captcha.submits == 0
    assert captcha.desafio is not original


def test_a_resposta_obsoleta_e_descartada_e_nao_adaptada(page, captcha, gemini):
    """Nunca remapear indices antigos para a grade nova."""
    gemini["resposta"] = RESPOSTA_A
    gemini["ao_chamar"] = lambda: captcha.trocar_desafio(_desafio_b())
    solver._solve_grade(page, "chave-de-teste", max_rounds=1)
    assert not captcha.tiles_clicados


# ── B · reload / substituicao de frame ───────────────────────────────────────

def test_b_reload_substitui_o_frame_zero_cliques(page, captcha, gemini):
    gemini["resposta"] = RESPOSTA_A
    gemini["ao_chamar"] = captcha.recarregar

    solver._solve_grade(page, "chave-de-teste", max_rounds=1)

    assert captcha.tiles_clicados == []
    assert captcha.submits == 0
    # O frame novo EXISTE e resolve normalmente — e justamente por isso que
    # "locator resolveu" nao serve como prova de identidade.
    assert solver._get_challenge_frame(page) is not None


def test_b_indice_do_iframe_ativo_mudou_mas_isso_nao_autoriza_clique(
        page, captcha, gemini):
    gemini["resposta"] = RESPOSTA_A
    gemini["ao_chamar"] = captcha.recarregar
    solver._solve_grade(page, "chave-de-teste", max_rounds=1)
    assert captcha.idx_ativo == 0          # era 1 antes do reload
    assert captcha.tiles_clicados == []


# ── C · desafio permaneceu o mesmo ───────────────────────────────────────────

def test_c_desafio_identico_clica_normalmente(page, captcha, gemini):
    gemini["resposta"] = {"task_summary": "onibus", "matching_tiles": [2, 5],
                          "confidence": "high"}
    solver._solve_grade(page, "chave-de-teste", max_rounds=1)
    assert [i for _d, i in captcha.tiles_clicados] == [2, 5]
    assert all(d is captcha.desafio for d, _i in captcha.tiles_clicados)
    assert captcha.submits >= 1


def test_c_guard_nao_introduz_clique_extra(page, captcha, gemini):
    gemini["resposta"] = {"task_summary": "onibus", "matching_tiles": [7],
                          "confidence": "high"}
    solver._solve_grade(page, "chave-de-teste", max_rounds=1)
    assert len(captcha.tiles_clicados) == 1


# ── D · tiles atualizados logo antes do primeiro clique ──────────────────────

def test_d_tiles_trocaram_com_o_mesmo_enunciado_zero_cliques(
        page, captcha, gemini):
    """Pior caso: o enunciado nao muda, so as imagens.

    Se a identidade fosse so o texto do prompt, este caso passaria batido — e
    e o mais comum no hCaptcha, que reusa o enunciado entre rodadas.
    """
    mesmo_prompt = captcha.desafio.prompt
    gemini["resposta"] = RESPOSTA_A
    gemini["ao_chamar"] = lambda: captcha.trocar_desafio(
        Desafio(prompt=mesmo_prompt, pixels=b"PNG-OUTROS-TILES"))

    solver._solve_grade(page, "chave-de-teste", max_rounds=1)

    assert captcha.tiles_clicados == []
    assert captcha.desafio.prompt == mesmo_prompt   # o texto continua igual


# ── E · nova rodada depois do submit ─────────────────────────────────────────

def test_e_cada_rodada_recaptura_e_clica_no_seu_proprio_desafio(
        page, captcha, gemini):
    """Resposta da rodada anterior nao pode sobreviver para a proxima."""
    primeiro = captcha.desafio
    segundo = _desafio_b()
    captcha.ao_submeter = lambda c: (c.trocar_desafio(segundo)
                                     if c.desafio is primeiro else c.resolver())
    gemini["resposta"] = {"task_summary": "x", "matching_tiles": [3],
                          "confidence": "high"}

    solver._solve_grade(page, "chave-de-teste", max_rounds=2)

    alvos = [d for d, _i in captcha.tiles_clicados]
    assert primeiro in alvos and segundo in alvos
    assert gemini["chamadas"] >= 2          # recapturou e reanalisou


def test_e_desafio_novo_recebe_analise_nova(page, captcha, gemini):
    captcha.ao_submeter = lambda c: c.trocar_desafio(_desafio_b())
    gemini["resposta"] = {"task_summary": "x", "matching_tiles": [1],
                          "confidence": "high"}
    solver._solve_grade(page, "chave-de-teste", max_rounds=2)
    assert gemini["chamadas"] == 2
    assert len({id(d) for d, _i in captcha.tiles_clicados}) == 2


# ── Caminho fused: identidade + geometria ────────────────────────────────────

def test_fused_desafio_mudou_zero_cliques_de_pixel(page, captcha, gemini):
    gemini["resposta"] = RESPOSTA_A
    gemini["ao_chamar"] = lambda: captcha.trocar_desafio(_desafio_b())
    captcha.desafio.n_tiles = 1      # forca o caminho de clique por pixel

    solver._solve_grade_fused(page, "chave-de-teste", max_rounds=1)

    assert captcha.cliques_pixel == []
    assert captcha.submits == 0


def test_fused_geometria_deslocada_zero_cliques(page, captcha, gemini):
    """A bbox e calculada ANTES do Gemini e usada em coordenadas absolutas.

    Se o iframe rolou ou mudou de lugar, clicar naquelas coordenadas acerta
    pixel arbitrario da pagina — pior que nao clicar.
    """
    caixa_deslocada = dict(CAIXA_PADRAO, y=CAIXA_PADRAO["y"] + 250)
    gemini["resposta"] = RESPOSTA_A
    gemini["ao_chamar"] = lambda: captcha.desafio.caixa.update(caixa_deslocada)
    captcha.desafio.n_tiles = 1

    solver._solve_grade_fused(page, "chave-de-teste", max_rounds=1)

    assert captcha.cliques_pixel == []


def test_fused_desafio_estavel_clica(page, captcha, gemini):
    gemini["resposta"] = {"task_summary": "x", "matching_tiles": [0],
                          "confidence": "high"}
    captcha.desafio.n_tiles = 1
    solver._solve_grade_fused(page, "chave-de-teste", max_rounds=1)
    assert len(captcha.cliques_pixel) == 1


# ── A funcao de guarda, isolada ──────────────────────────────────────────────

def test_fingerprint_muda_quando_os_pixels_mudam(page, captcha):
    antes = solver._fingerprint_desafio(page, captcha.desafio.pixels)
    captcha.trocar_desafio(_desafio_b())
    depois = solver._fingerprint_desafio(page, captcha.desafio.pixels)
    assert antes and depois and antes != depois


def test_fingerprint_muda_quando_so_o_enunciado_muda(page, captcha):
    antes = solver._fingerprint_desafio(page, captcha.desafio.pixels)
    captcha.trocar_desafio(Desafio(prompt="outro enunciado",
                                   pixels=captcha.desafio.pixels))
    depois = solver._fingerprint_desafio(page, captcha.desafio.pixels)
    assert antes != depois


def test_fingerprint_estavel_para_o_mesmo_desafio(page, captcha):
    a = solver._fingerprint_desafio(page, captcha.desafio.pixels)
    b = solver._fingerprint_desafio(page, captcha.desafio.pixels)
    assert a == b


@pytest.mark.parametrize("preparar", [
    pytest.param(lambda c: c.resolver(), id="desafio-sumiu"),
    pytest.param(lambda c: c.trocar_desafio(_desafio_b()), id="desafio-trocou"),
    pytest.param(lambda c: c.recarregar(), id="reload"),
])
def test_indeterminado_ou_diferente_nao_autoriza_clique(
        page, captcha, preparar):
    fp = solver._fingerprint_desafio(page, captcha.desafio.pixels)
    preparar(captcha)
    assert solver._desafio_ainda_e_o_mesmo(page, fp) is False


def test_fingerprint_de_origem_ausente_nao_autoriza_clique(page):
    """Se nem sabemos o que analisamos, nao ha o que confirmar."""
    assert solver._desafio_ainda_e_o_mesmo(page, None) is False
    assert solver._desafio_ainda_e_o_mesmo(page, "") is False


def test_recaptura_falhando_nao_autoriza_clique(page, captcha, monkeypatch):
    fp = solver._fingerprint_desafio(page, captcha.desafio.pixels)

    def screenshot_quebrado(*_a, **_k):
        raise RuntimeError("elemento destacado do DOM")

    monkeypatch.setattr(FakePage, "screenshot", screenshot_quebrado)
    monkeypatch.setattr(solver, "_capturar_desafio", lambda _p: (None, None))
    assert solver._desafio_ainda_e_o_mesmo(page, fp) is False


# ── Higiene: o fingerprint nao vaza para o log ───────────────────────────────

def test_descarte_nao_loga_fingerprint(page, captcha, gemini, capsys):
    gemini["resposta"] = RESPOSTA_A
    gemini["ao_chamar"] = lambda: captcha.trocar_desafio(_desafio_b())
    solver._solve_grade(page, "chave-de-teste", max_rounds=1)

    saida = capsys.readouterr().out
    assert "resposta descartada: desafio mudou" in saida
    fp = solver._fingerprint_desafio(page, captcha.desafio.pixels)
    assert fp and fp not in saida
    assert captcha.desafio.prompt not in saida


def test_solve_grade_isolado_de_captcha_e_gemini_reais(page, captcha, gemini):
    """Guarda-corpo do proprio arquivo: nada aqui toca rede."""
    solver._solve_grade(page, "chave-de-teste", max_rounds=1)
    assert isinstance(page, FakePage)
    assert gemini["chamadas"] >= 1
