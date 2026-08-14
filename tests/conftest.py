"""Fixtures do duble deterministico. As classes estao em `fakes.py`."""
import pytest
from fakes import Captcha, FakePage

from resolvedor_captcha import solver

# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def relogio_virtual(monkeypatch):
    """Tempo virtual: `sleep` avanca o relogio que `time` le.

    Anular so o `sleep` nao bastaria — o solver faz polling com deadline
    (`_wait_for_resolve`), e um `sleep` inerte transformaria cada espera de 3 s
    num busy-loop de 3 s REAIS. Com o relogio virtual a suite fica rapida e,
    mais importante, deterministica: nenhum teste depende de quanto a maquina
    demorou.
    """
    agora = {"t": 1_000.0}

    def dormir(segundos=0.0):
        agora["t"] += max(float(segundos or 0.0), 0.01)

    def ler():
        agora["t"] += 0.001      # avanca tambem sem sleep: nada trava
        return agora["t"]

    monkeypatch.setattr(solver.time, "sleep", dormir)
    monkeypatch.setattr(solver.time, "time", ler)


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
