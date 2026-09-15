"""Astra que recusa mesmo insistindo devolve a vez ao Gemini — 15/09/2026.

ESQUADROMIL, RUN-2258caa2, grade, rodada 2: a memória mandou direto ao Astra,
ele recusou 3/3, e o "tiles vazios" mandou ao Astra mais duas vezes. Nove
recusas, ~40 s, rodada perdida, e o Gemini não foi ouvido nenhuma vez.

Estes testes EXECUTAM `_gemini_call` com dublês — nenhuma chamada real. A
memória é zerada entre testes pelo `conftest.py`.
"""
import json

import pytest

from resolvedor_captcha import solver

RECUSA = {"matching_tiles": [], "confidence": "high",
          "task_summary": "Não posso resolver captcha, desculpe."}
TILES_DO_GEMINI = {"matching_tiles": [1, 4], "confidence": "high", "task_summary": "gemini"}
TILES_DO_ASTRA = {"matching_tiles": [3], "confidence": "high", "task_summary": "astra"}


class _Resposta:
    def __init__(self, dados):
        self.text = json.dumps(dados)


class _ClienteGemini:
    def __init__(self, falha=False):
        self.chamadas = 0
        self._falha = falha

        class _Modelos:
            def generate_content(inner, **_k):
                self.chamadas += 1
                if self._falha:
                    raise RuntimeError("503 UNAVAILABLE: the model is overloaded")
                return _Resposta(TILES_DO_GEMINI)

        self.models = _Modelos()


@pytest.fixture
def provedores(monkeypatch):
    estado = {"astra": 0, "resposta_astra": RECUSA, "cliente": _ClienteGemini()}

    def astra(contents, schema, tag, politica):
        estado["astra"] += 1
        return estado["resposta_astra"]

    monkeypatch.setattr(solver, "_astra_configurado", lambda: True)
    monkeypatch.setattr(solver, "_astra_call", astra)
    monkeypatch.setattr(solver, "_get_client", lambda _k: estado["cliente"])
    return estado


def test_atalho_da_memoria_com_recusa_pergunta_ao_gemini(provedores):
    """O caso da ESQUADROMIL."""
    solver._marcar_gemini_nao_fechou("grade")
    assert solver._gemini_call(["x"], {}, "chave", "grade") == TILES_DO_GEMINI
    assert provedores["astra"] == 1
    assert provedores["cliente"].chamadas == 1


def test_a_recusa_esquece_a_memoria_para_a_proxima_chamada(provedores):
    """Senão a chamada seguinte (o "tiles vazios" da grade) iria ao Astra de novo."""
    solver._marcar_gemini_nao_fechou("grade")
    solver._gemini_call(["x"], {}, "chave", "grade")
    assert solver._motivo_para_pular_gemini("grade") == ""


def test_chamador_que_pede_o_astra_e_recebe_recusa_ouve_o_gemini(provedores):
    """É o `direto_ao_segundo=(attempt > 1)` da grade."""
    r = solver._gemini_call(["x"], {}, "chave", "grade", direto_ao_segundo=True)
    assert r == TILES_DO_GEMINI
    assert provedores["astra"] == 1


def test_rodizio_esgotado_com_recusa_ouve_o_gemini(provedores):
    limite = solver._rodizio_do_segundo_provedor("grade")
    r = solver._gemini_call(["x"], {}, "chave", "grade", rodizio=limite)
    assert r == TILES_DO_GEMINI
    assert provedores["astra"] == 1, "a recusa no rodízio não pode cair no atalho e chamar de novo"


def test_quem_acabou_de_recusar_nao_e_o_ultimo_recurso(provedores):
    """Gemini fora e Astra recusando: a chamada falha em vez de pagar mais três recusas."""
    provedores["cliente"] = _ClienteGemini(falha=True)
    solver._marcar_gemini_nao_fechou("grade")
    with pytest.raises(Exception):
        solver._gemini_call(["x"], {}, "chave", "grade")
    assert provedores["astra"] == 1


def test_resposta_de_verdade_do_astra_continua_valendo(provedores):
    provedores["resposta_astra"] = TILES_DO_ASTRA
    solver._marcar_gemini_nao_fechou("grade")
    assert solver._gemini_call(["x"], {}, "chave", "grade") == TILES_DO_ASTRA
    assert provedores["cliente"].chamadas == 0


def test_vazio_sem_recusa_nao_devolve_a_vez():
    """Vazio sem a frase de recusa é incapacidade, não recusa: fica com o chamador."""
    assert solver._astra_recusou_insistindo("grade", {"matching_tiles": [], "task_summary": "nada"}) is False
