"""Latencia da chamada ao modelo e politica de fallback.

O freshness guard impede o clique errado, mas cada analise que envelhece e uma
rodada jogada fora. Estes testes fecham a outra ponta: quanto tempo uma unica
tentativa pode consumir, e quando vale trocar de modelo em vez de insistir.

Aritmetica do pior caso, antes: 4 modelos x 2 tentativas x ~143 s medidos +
backoff = ~19 minutos sobre UM screenshot. Depois: 4 modelos x 1 tentativa x
30 s = 2 minutos.
"""
import json

import pytest

from resolvedor_captcha import solver


class _Resposta:
    def __init__(self, texto):
        self.text = texto


def _cliente(comportamento):
    """Cliente falso. `comportamento[modelo]` = excecao ou dict de resposta."""
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


# ── Timeout explicito ────────────────────────────────────────────────────────

def test_timeout_padrao_e_de_30_segundos():
    assert solver.GEMINI_TIMEOUT_MS == 30_000


def test_config_carrega_o_timeout_em_milissegundos():
    """Sem isto o SDK usa o default e uma tentativa dura minutos."""
    config = solver._make_config(solver._SCHEMA_GRADE, solver.GEMINI_MODELS[0])
    assert config.http_options is not None
    assert config.http_options.timeout == solver.GEMINI_TIMEOUT_MS


def test_timeout_acompanha_toda_chamada(monkeypatch):
    """O teto vale por tentativa, em qualquer modelo da lista."""
    for modelo in solver.GEMINI_MODELS:
        config = solver._make_config(solver._SCHEMA_GRADE, modelo)
        assert config.http_options.timeout == solver.GEMINI_TIMEOUT_MS


def test_config_preserva_determinismo_e_schema():
    """O timeout nao pode ter custado nenhuma garantia antiga."""
    config = solver._make_config(solver._SCHEMA_GRADE, solver.GEMINI_MODELS[0])
    assert config.temperature == 0.0
    assert config.response_mime_type == "application/json"
    assert config.response_schema is not None


# ── Politica de sobrecarga ───────────────────────────────────────────────────

def test_sobrecarga_nao_repete_o_mesmo_modelo(monkeypatch):
    """Uma tentativa por modelo sobrecarregado — o pool e dele, nao da chamada."""
    cliente, chamadas = _cliente({
        solver.GEMINI_MODELS[0]: RuntimeError("503 UNAVAILABLE"),
        solver.GEMINI_MODELS[1]: {"ok": 2},
    })
    monkeypatch.setattr(solver, "_get_client", lambda _k: cliente)

    assert solver._gemini_call([], {}, "k", "grade") == {"ok": 2}
    assert chamadas == [solver.GEMINI_MODELS[0], solver.GEMINI_MODELS[1]]
    assert chamadas.count(solver.GEMINI_MODELS[0]) == 1


def test_timeout_tambem_avanca_para_o_proximo_modelo(monkeypatch):
    """A lista esta ordenada por latencia medida: o proximo e o mais rapido
    dos que restam. Insistir no que acabou de estourar e a pior escolha.
    """
    cliente, chamadas = _cliente({
        solver.GEMINI_MODELS[0]: TimeoutError("deadline exceeded"),
        solver.GEMINI_MODELS[1]: {"ok": 3},
    })
    monkeypatch.setattr(solver, "_get_client", lambda _k: cliente)

    assert solver._gemini_call([], {}, "k", "grade") == {"ok": 3}
    assert chamadas.count(solver.GEMINI_MODELS[0]) == 1


@pytest.mark.parametrize("mensagem", ["timeout", "request timed out",
                                      "deadline exceeded"])
def test_timeout_e_classificado_como_transitorio(mensagem):
    assert solver._is_overloaded_error(Exception(mensagem)) is True


def test_todos_sobrecarregados_faz_uma_tentativa_por_modelo(monkeypatch):
    cliente, chamadas = _cliente({m: RuntimeError("503 UNAVAILABLE")
                                  for m in solver.GEMINI_MODELS})
    monkeypatch.setattr(solver, "_get_client", lambda _k: cliente)

    with pytest.raises(RuntimeError):
        solver._gemini_call([], {}, "k", "grade")
    assert chamadas == list(solver.GEMINI_MODELS)   # uma cada, na ordem


# ── Erros que NAO sao transitorios ───────────────────────────────────────────

@pytest.mark.parametrize("mensagem", [
    "API key not valid", "401 unauthorized", "400 INVALID_ARGUMENT",
    "PERMISSION_DENIED",
])
def test_erro_de_configuracao_nao_circula_pelos_modelos(monkeypatch, mensagem):
    """Chave invalida nao melhora no modelo seguinte.

    Deixar circular gastaria quatro chamadas inuteis e ainda pioraria o
    diagnostico, porque o erro reportado seria o do ultimo modelo.
    """
    cliente, chamadas = _cliente({m: RuntimeError(mensagem)
                                  for m in solver.GEMINI_MODELS})
    monkeypatch.setattr(solver, "_get_client", lambda _k: cliente)

    with pytest.raises(RuntimeError):
        solver._gemini_call([], {}, "k", "grade")
    assert set(chamadas) == {solver.GEMINI_MODELS[0]}


def test_erro_de_configuracao_ainda_repete_no_mesmo_modelo(monkeypatch):
    """Retry local continua valendo para o que pode ser intermitente."""
    cliente, chamadas = _cliente({m: RuntimeError("400 INVALID_ARGUMENT")
                                  for m in solver.GEMINI_MODELS})
    monkeypatch.setattr(solver, "_get_client", lambda _k: cliente)

    with pytest.raises(RuntimeError):
        solver._gemini_call([], {}, "k", "grade")
    assert len(chamadas) == solver.GEMINI_TRIES_PER_MODEL


# ── Caminhos de sucesso ──────────────────────────────────────────────────────

def test_sucesso_no_primeiro_modelo_faz_uma_chamada(monkeypatch):
    cliente, chamadas = _cliente({solver.GEMINI_MODELS[0]: {"ok": 1}})
    monkeypatch.setattr(solver, "_get_client", lambda _k: cliente)
    assert solver._gemini_call([], {}, "k", "grade") == {"ok": 1}
    assert chamadas == [solver.GEMINI_MODELS[0]]


def test_ordem_dos_modelos_preservada():
    """A ordem veio de medicao (17/08/2026); este commit nao a altera."""
    assert solver.GEMINI_MODELS == [
        "gemini-3.5-flash",
        "gemini-3-flash-preview",
        "gemini-3.5-flash-lite",
        "gemini-flash-lite-latest",
    ]
    assert solver.GEMINI_MODEL == solver.GEMINI_MODELS[0]
    assert len(solver.GEMINI_MODELS) == 4


# Modelos reprovados por MEDICAO — nao voltam ao caminho quente sem nova medida.
# Ou o pool vive saturado (timeout/503 na maioria das chamadas) ou o ID ja foi
# aposentado pelo Google (404). Ver o comentario de GEMINI_MODELS em solver.py.
MODELOS_REPROVADOS = (
    "gemini-pro-latest",       # 3/16 — 11 timeouts
    "gemini-3.1-pro-preview",  # 4/16 — 12 timeouts
    "gemini-flash-latest",     # 3/16 — 9 timeouts, 4x 503
    "gemini-3.6-flash",        # 3/8
    "gemini-3.7-flash",        # 2/8
    "gemini-2.5-flash",        # 0/6 — 404 "no longer available"
    "gemini-2.0-flash",        # 404 "no longer available"
)


def test_nenhum_modelo_reprovado_no_caminho_quente():
    """Esta automacao nao roda com modelo instavel: a lista padrao e so aprovado."""
    reprovados_na_lista = [m for m in solver.GEMINI_MODELS if m in MODELOS_REPROVADOS]
    assert reprovados_na_lista == []


def test_pior_caso_de_chamadas_e_um_por_modelo(monkeypatch):
    """Teto de chamadas quando tudo esta sobrecarregado."""
    cliente, chamadas = _cliente({m: RuntimeError("overloaded")
                                  for m in solver.GEMINI_MODELS})
    monkeypatch.setattr(solver, "_get_client", lambda _k: cliente)
    with pytest.raises(RuntimeError):
        solver._gemini_call([], {}, "k", "grade")
    assert len(chamadas) == len(solver.GEMINI_MODELS)
