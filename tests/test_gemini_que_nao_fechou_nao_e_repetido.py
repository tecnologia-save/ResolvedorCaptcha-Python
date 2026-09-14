"""Gemini que não fechou não é perguntado de novo — pedido do Jean, 14/09/2026.

"O Gemini falha, você chama o Astra. Por que na segunda etapa não chama direto
o Astra, em vez de refazer a chamada ao Gemini que já deu erro?"

Vale onde o Gemini vem PRIMEIRO — grade e grade_fused (`ORDEM_DO_SEGUNDO_PROVEDOR`
= 2). Em imagem, bola e cartão o segundo provedor já é o primeiro a ser ouvido.

Estes testes EXECUTAM `_gemini_call` com dublês — nenhuma chamada real a Gemini
nem Astra. A memória é zerada entre testes pelo `conftest.py`.
"""
import inspect

import pytest

from resolvedor_captcha import solver


class _Relogio:
    def __init__(self):
        self.agora = 1000.0

    def __call__(self):
        return self.agora


class _ClienteGemini:
    def __init__(self):
        self.chamadas = 0

        class _Modelos:
            def generate_content(inner, **_k):
                self.chamadas += 1
                raise RuntimeError("503 UNAVAILABLE: the model is overloaded")

        self.models = _Modelos()


@pytest.fixture
def provedores(monkeypatch):
    estado = {"astra": 0, "astra_erro": None, "cliente": _ClienteGemini()}

    def astra(contents, schema, tag, politica):
        estado["astra"] += 1
        if estado["astra_erro"] is not None:
            erro, estado["astra_erro"] = estado["astra_erro"], None
            raise erro
        return {"ok": "astra"}

    monkeypatch.setattr(solver, "_astra_configurado", lambda: True)
    monkeypatch.setattr(solver, "_astra_call", astra)
    monkeypatch.setattr(solver, "_get_client", lambda _k: estado["cliente"])
    return estado


# ── memória ───────────────────────────────────────────────────────────────────

def test_falhou_neste_captcha_vale_para_qualquer_passo_dele():
    """O segundo passo que o hCaptcha emenda pode ser de outro tipo."""
    solver._marcar_gemini_nao_fechou("grade")
    assert solver._motivo_para_pular_gemini("grade")
    assert solver._motivo_para_pular_gemini("grade_fused"), "o passo seguinte deste captcha também"


def test_no_proximo_captcha_so_o_mesmo_tipo_pula_o_gemini():
    solver._marcar_gemini_nao_fechou("grade_fused")
    solver._novo_captcha()
    assert solver._motivo_para_pular_gemini("grade_fused")
    assert solver._motivo_para_pular_gemini("grade") == ""


def test_a_memoria_do_tipo_vence_em_15_minutos(monkeypatch):
    relogio = _Relogio()
    monkeypatch.setattr(solver.time, "monotonic", relogio)
    solver._marcar_gemini_nao_fechou("grade")
    solver._novo_captcha()
    relogio.agora += solver.MEMORIA_GEMINI_FALHOU_S - 1
    assert solver._motivo_para_pular_gemini("grade")
    relogio.agora += 2
    assert solver._motivo_para_pular_gemini("grade") == ""


def test_triagem_nao_entra_na_memoria():
    solver._marcar_gemini_nao_fechou("triagem")
    assert solver._motivo_para_pular_gemini("grade") == ""
    assert solver._motivo_para_pular_gemini("triagem") == ""


# ── _gemini_call, na grade (Gemini primeiro) ─────────────────────────────────

def test_sem_falha_anterior_o_gemini_continua_primeiro(provedores):
    """A regra não inverte a ordem: só pula quem acabou de falhar."""
    solver._gemini_call(["x"], {}, "chave", "grade")
    assert provedores["cliente"].chamadas >= 1


def test_a_falha_real_do_gemini_marca_e_a_etapa_seguinte_vai_direto(provedores):
    assert solver._gemini_call(["x"], {}, "chave", "grade") == {"ok": "astra"}
    antes = provedores["cliente"].chamadas
    assert antes >= 1
    assert solver._motivo_para_pular_gemini("grade")
    assert solver._gemini_call(["x"], {}, "chave", "grade") == {"ok": "astra"}
    assert provedores["cliente"].chamadas == antes, "segunda etapa: direto ao Astra, sem Gemini"


def test_depois_da_falha_o_gemini_nao_e_chamado_de_novo(provedores):
    solver._marcar_gemini_nao_fechou("grade")
    assert solver._gemini_call(["x"], {}, "chave", "grade") == {"ok": "astra"}
    assert provedores["astra"] == 1
    assert provedores["cliente"].chamadas == 0


def test_o_atalho_da_memoria_nao_renova_a_propria_memoria(provedores, monkeypatch):
    relogio = _Relogio()
    monkeypatch.setattr(solver.time, "monotonic", relogio)
    solver._marcar_gemini_nao_fechou("grade")
    solver._novo_captcha()
    relogio.agora += 14 * 60
    solver._gemini_call(["x"], {}, "chave", "grade")          # vai pela memória
    relogio.agora += 2 * 60                                   # 16 min da falha real
    assert solver._motivo_para_pular_gemini("grade") == "", "senão o Gemini nunca mais seria ouvido"


def test_astra_que_erra_no_atalho_da_memoria_devolve_a_vez_ao_gemini(provedores):
    solver._marcar_gemini_nao_fechou("grade")
    provedores["astra_erro"] = RuntimeError("APIConnectionError")
    solver._gemini_call(["x"], {}, "chave", "grade")
    assert provedores["cliente"].chamadas >= 1, "memória não pode deixar o captcha sem provedor"


def test_astra_sem_orcamento_no_atalho_nao_cai_no_gemini(provedores):
    """Sem tempo para o Astra também não há para o Gemini: sobe como antes."""
    solver._marcar_gemini_nao_fechou("grade")
    provedores["astra_erro"] = solver.SegundoProvedorSemOrcamento("restam 3s")
    with pytest.raises(solver.SegundoProvedorSemOrcamento):
        solver._gemini_call(["x"], {}, "chave", "grade")
    assert provedores["cliente"].chamadas == 0


def test_orcamento_esgotado_nao_usa_o_atalho(provedores):
    solver._marcar_gemini_nao_fechou("grade")
    esgotada = solver.PoliticaLatencia(timeout_ms=1000, fim=solver.time.monotonic() - 1)
    with pytest.raises(RuntimeError):
        solver._gemini_call(["x"], {}, "chave", "grade", esgotada)
    assert provedores["astra"] == 0


# ── pontos de amarração ───────────────────────────────────────────────────────

def test_cada_captcha_comeca_com_a_memoria_dele_limpa():
    fonte = inspect.getsource(solver.solve_hcaptcha)
    assert "_novo_captcha()" in fonte
    assert fonte.index("_novo_captcha()") < fonte.index("for rnd in range")


@pytest.mark.parametrize("funcao", ["_solve_imagem", "_solve_grade", "_solve_grade_fused"])
def test_rodada_sem_orcamento_nao_comeca(funcao):
    fonte = inspect.getsource(getattr(solver, funcao))
    assert "sem tempo" in fonte and "para outra rodada" in fonte
    assert "rnd > 1 and politica is not None" in fonte, "a 1ª rodada sempre tenta"
