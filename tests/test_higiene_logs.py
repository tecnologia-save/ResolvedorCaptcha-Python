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


# ── Coleta de amostras: util, mas com freio ─────────────────────────────────
#
# Uma coleta anterior tirava 40 screenshots a cada classificacao, dentro de
# `detectar_tipo_captcha` — que a lib de login chama REPETIDAMENTE enquanto
# aguarda o desfecho. A run ficava capturando sem parar, e os 20s saiam ANTES
# de o relogio do orcamento comecar a contar.

def test_sem_a_variavel_nao_guarda_nada(monkeypatch, tmp_path):
    monkeypatch.delenv("CAPTCHA_DEBUG_AMOSTRAS_DIR", raising=False)
    solver._AMOSTRAS_GUARDADAS.clear()
    solver._guardar_amostra(object(), "grade_fused")
    assert list(tmp_path.iterdir()) == []


def test_guarda_UMA_por_tipo_por_processo(monkeypatch, tmp_path):
    """Uma por tipo — nao uma por rodada. O laco chama isto a cada iteracao."""
    monkeypatch.setenv("CAPTCHA_DEBUG_AMOSTRAS_DIR", str(tmp_path))
    monkeypatch.setattr(solver, "_capturar_desafio", lambda _p: (b"png", None))
    monkeypatch.setattr(solver, "_extrair_instrucao", lambda _p: "quebra o padrao")
    solver._AMOSTRAS_GUARDADAS.clear()
    for _ in range(5):
        solver._guardar_amostra(object(), "grade_fused")
    pngs = list(tmp_path.glob("*.png"))
    assert len(pngs) == 1, [p.name for p in pngs]


def test_tipos_diferentes_geram_amostras_diferentes(monkeypatch, tmp_path):
    monkeypatch.setenv("CAPTCHA_DEBUG_AMOSTRAS_DIR", str(tmp_path))
    monkeypatch.setattr(solver, "_capturar_desafio", lambda _p: (b"png", None))
    monkeypatch.setattr(solver, "_extrair_instrucao", lambda _p: "x")
    solver._AMOSTRAS_GUARDADAS.clear()
    solver._guardar_amostra(object(), "grade_fused")
    solver._guardar_amostra(object(), "bola_em_movimento")
    assert len(list(tmp_path.glob("*.png"))) == 2


def test_o_enunciado_vai_junto(monkeypatch, tmp_path):
    """E o enunciado que indexa o catalogo: a imagem muda, o texto se repete."""
    monkeypatch.setenv("CAPTCHA_DEBUG_AMOSTRAS_DIR", str(tmp_path))
    monkeypatch.setattr(solver, "_capturar_desafio", lambda _p: (b"png", None))
    monkeypatch.setattr(solver, "_extrair_instrucao",
                        lambda _p: "clique no icone que quebra o padrao")
    solver._AMOSTRAS_GUARDADAS.clear()
    solver._guardar_amostra(object(), "grade_fused")
    txts = list(tmp_path.glob("*.txt"))
    assert len(txts) == 1
    assert "quebra o padrao" in txts[0].read_text(encoding="utf-8")


def test_falha_na_coleta_nao_derruba_a_resolucao(monkeypatch, tmp_path):
    monkeypatch.setenv("CAPTCHA_DEBUG_AMOSTRAS_DIR", str(tmp_path))

    def explode(_p):
        raise RuntimeError("captura falhou")

    monkeypatch.setattr(solver, "_capturar_desafio", explode)
    solver._AMOSTRAS_GUARDADAS.clear()
    solver._guardar_amostra(object(), "grade_fused")   # nao pode levantar


# ── "Sumiu" e "resolvi" nao podem ser a mesma frase ─────────────────────────
#
# Em 08/09/2026 o Jean resolveu um captcha A MAO e o log registrou
# "Captcha resolvido na iteracao 1!". Eu li como resolucao automatica e
# reportei a ele como o primeiro sucesso ponta a ponta. Nao era.
#
# Os resolvedores checam `_challenge_visible` no inicio de cada rodada e, se o
# desafio sumiu, devolvem True — o desfecho FUNCIONAL e o mesmo, mas a causa
# nao. Numa run acompanhada por alguem, isso torna todo sucesso ambiguo.

def test_sem_submissao_nossa_o_log_diz_isso(capsys):
    marca = solver._SUBMISSOES
    assert solver._sumiu("grade", marca) is True
    saida = capsys.readouterr().out
    assert "SEM submissão nossa" in saida
    assert "fora da automação" in saida


def test_com_submissao_nossa_o_log_afirma_resolucao(capsys, monkeypatch):
    marca = solver._SUBMISSOES
    monkeypatch.setattr(solver, "_SUBMISSOES", marca + 1)
    assert solver._sumiu("grade", marca) is True
    saida = capsys.readouterr().out
    assert "resolvido!" in saida
    assert "SEM submissão nossa" not in saida


def test_o_desfecho_funcional_nao_muda(capsys):
    """Nos dois casos nao ha mais desafio: quem chamou segue igual."""
    assert solver._sumiu("bola", solver._SUBMISSOES) is True
    assert solver._sumiu("bola", solver._SUBMISSOES - 1) is True


def test_submeter_conta(monkeypatch):
    """Sem o contador subir, todo desaparecimento pareceria alheio."""
    monkeypatch.setattr(solver, "_get_challenge_frame", lambda _p: None)

    class Pagina:
        def wait_for_timeout(self, _ms):
            pass

        def __getattr__(self, _n):
            raise RuntimeError("sem navegador")

    antes = solver._SUBMISSOES
    try:
        solver._submit_captcha(Pagina())
    except Exception:
        pass
    assert solver._SUBMISSOES == antes + 1, "a submissao nao foi contabilizada"


# ── Triagem: "nao consegui" vira "nao consegui, e o que vi foi isto" ────────
#
# Uma amostra sozinha obriga alguem a abrir a imagem e adivinhar a mecanica.
# Com a triagem, um formato novo chega descrito: se e animado, quantos cliques
# pede, e se cai numa familia que ja temos resolvedor.

def test_sem_a_variavel_nao_diagnostica(monkeypatch):
    monkeypatch.delenv("CAPTCHA_DEBUG_AMOSTRAS_DIR", raising=False)
    chamou = {"n": 0}
    monkeypatch.setattr(solver, "_gemini_call",
                        lambda *a, **k: chamou.__setitem__("n", chamou["n"] + 1))
    solver._diagnosticar_desafio(object(), "chave", "grade_fused")
    assert chamou["n"] == 0


def test_sem_chave_nao_diagnostica(monkeypatch, tmp_path):
    monkeypatch.setenv("CAPTCHA_DEBUG_AMOSTRAS_DIR", str(tmp_path))
    chamou = {"n": 0}
    monkeypatch.setattr(solver, "_gemini_call",
                        lambda *a, **k: chamou.__setitem__("n", chamou["n"] + 1))
    solver._diagnosticar_desafio(object(), "", "grade_fused")
    assert chamou["n"] == 0


def test_a_triagem_grava_o_que_decide_o_proximo_passo(monkeypatch, tmp_path):
    """O arquivo tem de responder: e formato novo? o que ele pede?"""
    monkeypatch.setenv("CAPTCHA_DEBUG_AMOSTRAS_DIR", str(tmp_path))
    monkeypatch.setattr(solver, "_capturar_desafio", lambda _p: (b"png", None))
    monkeypatch.setattr(solver, "_parte_imagem", lambda _b: "imagem")
    monkeypatch.setattr(solver, "_gemini_call", lambda *a, **k: {
        "instrucao_lida": "Arraste a peça para o lugar certo",
        "mecanica": "encaixar uma peca deslizante",
        "alvos": "pecas de quebra-cabeca",
        "acao_necessaria": "arrastar",
        "e_animado": False,
        "quantos_cliques": 0,
        "familia_conhecida": False,
        "por_que_falhou": "a automacao so sabe clicar",
    })
    solver._diagnosticar_desafio(object(), "chave", "desconhecido")
    arquivos = list(tmp_path.glob("*-triagem.md"))
    assert len(arquivos) == 1
    texto = arquivos[0].read_text(encoding="utf-8")
    assert "arrastar" in texto
    assert "familia conhecida : False" in texto
    assert "so sabe clicar" in texto


def test_falha_na_triagem_nao_derruba_nada(monkeypatch, tmp_path):
    monkeypatch.setenv("CAPTCHA_DEBUG_AMOSTRAS_DIR", str(tmp_path))
    monkeypatch.setattr(solver, "_capturar_desafio", lambda _p: (b"png", None))
    monkeypatch.setattr(solver, "_parte_imagem", lambda _b: "imagem")

    def explode(*_a, **_k):
        raise RuntimeError("modelo fora do ar")

    monkeypatch.setattr(solver, "_gemini_call", explode)
    solver._diagnosticar_desafio(object(), "chave", "grade")   # nao pode levantar


def test_a_triagem_nao_roda_dentro_do_laco_de_rodadas():
    """Ali ainda ha orcamento a proteger, e um desafio que pode ser resolvido
    nao precisa de autopsia."""
    import inspect
    fonte = inspect.getsource(solver.solve_hcaptcha)
    pos_chamada = fonte.index("_diagnosticar_desafio(")
    pos_limite = fonte.index("Limite de {max_rounds} iterações atingido")
    assert pos_chamada > pos_limite, "a triagem entrou no meio do laço"
