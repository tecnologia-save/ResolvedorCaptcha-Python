"""Higiene dos logs — nada que venha de fora entra em texto no log.

O stdout deste solver vira log de execucao na plataforma que hospeda a
automacao. `str(e)` de um erro do google.genai carrega o JSON cru da resposta;
erros do Playwright embutem seletor, URL do frame e trechos do DOM. Os dois
estavam sendo impressos.

A regra: sai apenas o que e NOSSO — categoria de vocabulario fechado, nome de
modelo da nossa lista, status numerico e o nome da classe. Ler `str(e)` para
CLASSIFICAR continua permitido; o que nao pode e imprimi-lo.
"""
import ast
import json
import pathlib

import pytest

from resolvedor_captcha import solver

FONTE = pathlib.Path(solver.__file__).read_text(encoding="utf-8")

# Corpo tipico de um erro do provedor, com o formato que apareceu no log real.
CORPO_PROVEDOR = json.dumps({
    "error": {
        "code": 503,
        "message": "The model is overloaded. Please try again later.",
        "status": "UNAVAILABLE",
        "details": [{"@type": "type.googleapis.com/google.rpc.DebugInfo",
                     "detail": "SEGREDO-DO-CORPO-QUE-NAO-PODE-VAZAR"}],
    }
})


class _ErroProvedor(Exception):
    def __init__(self, corpo, code=503):
        super().__init__(corpo)
        self.code = code


# ── Gate estrutural: nenhum print interpola excecao capturada ────────────────

def test_nenhum_print_interpola_a_excecao_capturada():
    """Varre o fonte inteiro, nao so os pontos que ja conhecemos."""
    arvore = ast.parse(FONTE)
    ofensores = []
    for h in [n for n in ast.walk(arvore) if isinstance(n, ast.ExceptHandler)]:
        if not h.name:
            continue
        for no in ast.walk(h):
            if not (isinstance(no, ast.Call)
                    and getattr(no.func, "id", None) == "print"):
                continue
            texto = ast.unparse(no)
            for forma in (f"{{{h.name}}}", f"str({h.name})", f"repr({h.name})",
                          f"_limpar_texto({h.name}", f"{h.name}.args"):
                if forma in texto:
                    ofensores.append(f"linha {no.lineno}: {forma}")
    assert ofensores == []


def test_nenhum_print_usa_corpo_de_resposta():
    arvore = ast.parse(FONTE)
    for no in ast.walk(arvore):
        if isinstance(no, ast.Call) and getattr(no.func, "id", None) == "print":
            texto = ast.unparse(no)
            for proibido in ("response.text", "response.content", ".headers",
                             "resp.text", "b64", "base64", "api_key"):
                assert proibido not in texto, f"linha {no.lineno}: {proibido}"


def test_api_key_nunca_e_impressa():
    """A chave circula por parametro em todo o modulo — mas nao por print."""
    arvore = ast.parse(FONTE)
    for no in ast.walk(arvore):
        if isinstance(no, ast.Call) and getattr(no.func, "id", None) == "print":
            assert "api_key" not in ast.unparse(no)


def test_screenshot_nunca_vai_para_o_log():
    arvore = ast.parse(FONTE)
    for no in ast.walk(arvore):
        if isinstance(no, ast.Call) and getattr(no.func, "id", None) == "print":
            texto = ast.unparse(no)
            for proibido in ("{png", "{iframe_png", "{tiles_png", "{png_raw",
                             "{ref_img"):
                assert proibido not in texto, f"linha {no.lineno}"


# ── Diagnostico: so campos nossos ────────────────────────────────────────────

def test_diagnostico_nao_contem_o_corpo_do_provedor():
    erro = _ErroProvedor(CORPO_PROVEDOR)
    linha = solver._diagnostico_erro(erro, solver.GEMINI_MODELS[0])
    assert "SEGREDO-DO-CORPO-QUE-NAO-PODE-VAZAR" not in linha
    assert "The model is overloaded" not in linha
    assert "googleapis" not in linha
    assert "{" not in linha and "}" not in linha


def test_diagnostico_traz_modelo_categoria_tipo_e_status():
    erro = _ErroProvedor(CORPO_PROVEDOR)
    linha = solver._diagnostico_erro(erro, solver.GEMINI_MODELS[0])
    assert f"modelo={solver.GEMINI_MODELS[0]}" in linha
    assert "categoria=indisponivel" in linha
    assert "tipo=_ErroProvedor" in linha
    assert "status=503" in linha


def test_modelo_fora_da_nossa_lista_nao_entra_no_log():
    """Nome de modelo so e nosso se veio da nossa lista."""
    linha = solver._diagnostico_erro(RuntimeError("503"), "modelo-de-fora")
    assert "modelo-de-fora" not in linha
    assert "modelo=" not in linha


def test_diagnostico_sem_modelo_continua_valido():
    linha = solver._diagnostico_erro(TimeoutError("deadline exceeded"))
    assert "categoria=tempo_esgotado" in linha
    assert "tipo=TimeoutError" in linha


@pytest.mark.parametrize(("mensagem", "esperada"), [
    ("503 UNAVAILABLE", "indisponivel"),
    ("429 RESOURCE_EXHAUSTED", "limite_de_uso"),
    ("404 model not found", "modelo_ausente"),
    ("deadline exceeded", "tempo_esgotado"),
    ("401 unauthorized", "credencial"),
    ("400 INVALID_ARGUMENT", "requisicao_invalida"),
    ("algo totalmente novo", "desconhecido"),
])
def test_categorias_sao_vocabulario_fechado(mensagem, esperada):
    assert solver._categoria_do_erro(Exception(mensagem)) == esperada


def test_categoria_nunca_devolve_texto_do_provedor():
    fechado = {c for c, _m in solver._CATEGORIAS_ERRO} | {"desconhecido"}
    assert solver._categoria_do_erro(Exception(CORPO_PROVEDOR)) in fechado


# ── Status: inteiro, e so ────────────────────────────────────────────────────

def test_status_vem_do_atributo_quando_existe():
    assert solver._status_do_erro(_ErroProvedor("x", code=403)) == 403


def test_status_ausente_devolve_none():
    assert solver._status_do_erro(RuntimeError("sem numero algum")) is None


@pytest.mark.parametrize("valor", [True, False, "503", 99, 600, None])
def test_status_recusa_valor_que_nao_e_status(valor):
    erro = RuntimeError("sem numero")
    erro.code = valor
    assert solver._status_do_erro(erro) is None


# ── Comportamento no fluxo real ──────────────────────────────────────────────

def test_falha_do_modelo_loga_sem_corpo(monkeypatch, capsys):
    class _Models:
        def generate_content(self, model, contents, config):
            raise _ErroProvedor(CORPO_PROVEDOR)

    class _Cliente:
        models = _Models()

    monkeypatch.setattr(solver, "_get_client", lambda _k: _Cliente())
    with pytest.raises(RuntimeError) as exc:
        solver._gemini_call([], {}, "chave", "grade")

    saida = capsys.readouterr().out
    assert "SEGREDO-DO-CORPO-QUE-NAO-PODE-VAZAR" not in saida
    assert "The model is overloaded" not in saida
    assert "status=503" in saida and "categoria=indisponivel" in saida
    # A mensagem da propria excecao tambem vira log em quem a captura.
    assert "SEGREDO-DO-CORPO-QUE-NAO-PODE-VAZAR" not in str(exc.value)


def test_falha_no_solver_loga_sem_corpo(page, captcha, monkeypatch, capsys):
    def explode(*_a, **_k):
        raise _ErroProvedor(CORPO_PROVEDOR)

    monkeypatch.setattr(solver, "_gemini_grade", explode)
    solver._solve_grade(page, "chave", max_rounds=1)

    saida = capsys.readouterr().out
    assert "SEGREDO-DO-CORPO-QUE-NAO-PODE-VAZAR" not in saida
    assert "categoria=indisponivel" in saida
    assert captcha.tiles_clicados == []


def test_falha_de_clique_loga_so_o_tipo(page, captcha, monkeypatch, capsys):
    """Erro do Playwright embute seletor e URL do frame."""
    from fakes import FakeLocator

    def click_quebrado(*_a, **_k):
        raise RuntimeError("Timeout 2000ms exceeded. waiting for "
                           "frame.locator('SELETOR-INTERNO') at "
                           "https://newassets.hcaptcha.com/SEGREDO-URL")

    monkeypatch.setattr(FakeLocator, "click", click_quebrado)
    solver._click_grade_tiles(page, [0, 1])

    saida = capsys.readouterr().out
    assert "SELETOR-INTERNO" not in saida
    assert "SEGREDO-URL" not in saida
    assert "hcaptcha.com" not in saida
    assert "RuntimeError" in saida


def test_limpar_texto_segue_valendo_para_texto_do_modelo():
    """`_limpar_texto` nao foi removido: ele serve ao task_summary, que e
    conteudo pedido ao modelo, nao mensagem de erro do provedor.
    """
    assert solver._limpar_texto("resumo   da\n tarefa") == "resumo da tarefa"
