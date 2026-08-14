"""Fixtures do duble deterministico. As classes estao em `fakes.py`."""
import pytest
from fakes import Captcha, FakePage

from resolvedor_captcha import solver

# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def sem_sleep(monkeypatch):
    """O solver dorme entre tentativas; nos testes isso e so lentidao."""
    monkeypatch.setattr(solver.time, "sleep", lambda *_a, **_k: None)


@pytest.fixture
def captcha():
    return Captcha()


@pytest.fixture
def page(captcha):
    return FakePage(captcha)


@pytest.fixture
def gemini(monkeypatch):
    """Substitui as chamadas ao modelo. Nenhuma rede, nenhuma API key.

    `resposta` e o dict que o modelo devolveria; `ao_chamar` roda ANTES de
    responder — e o gancho que simula "o desafio mudou enquanto o modelo
    pensava", sem depender de tempo real.
    """
    estado = {
        "resposta": {"task_summary": "onibus", "matching_tiles": [0, 4, 8],
                     "confidence": "high"},
        "ao_chamar": None,
        "chamadas": 0,
    }

    def falso(*_a, **_k):
        estado["chamadas"] += 1
        if callable(estado["ao_chamar"]):
            estado["ao_chamar"]()
        resposta = estado["resposta"]
        if isinstance(resposta, BaseException):
            raise resposta
        return resposta

    monkeypatch.setattr(solver, "_gemini_grade", falso)
    monkeypatch.setattr(solver, "_gemini_grade_fused", falso)
    return estado
