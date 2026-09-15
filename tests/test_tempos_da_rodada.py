"""A linha da resposta de cada rodada diz quem respondeu e em quanto tempo.

Pedido do Jean em 15/09/2026, junto com a limpeza do log. As linhas de
screenshot e recorte saíram, e eram elas que mediam o tempo parado antes de o
provedor ser chamado (~12 s na ALEX ROCHA, investigação ainda aberta). No lugar
delas, a resposta de cada rodada traz:

    'itens que cabem num bolso' | high | tiles=[2, 4, 5] · Astra respondeu em 9s · preparo e espera 12s
"""
import ast
import json
import pathlib

from resolvedor_captcha import solver


class _Resposta:
    text = json.dumps({"matching_tiles": [2], "confidence": "high"})


class _Cliente:
    class models:  # noqa: N801 — imita o SDK
        @staticmethod
        def generate_content(**_k):
            return _Resposta()


def test_sem_chamada_respondida_diz_so_o_tempo_da_rodada(monkeypatch):
    agora = [100.0]
    monkeypatch.setattr(solver.time, "monotonic", lambda: agora[0])
    inicio = solver._comecar_rodada()
    agora[0] += 7
    assert solver._tempos_da_rodada(inicio) == "rodada em 7s"


def test_resposta_do_gemini_fica_registrada_com_o_modelo(monkeypatch):
    monkeypatch.setattr(solver, "_astra_configurado", lambda: False)
    monkeypatch.setattr(solver, "_get_client", lambda _k: _Cliente())
    modelo = solver.modelos_ativos()[0]
    inicio = solver._comecar_rodada()
    solver._gemini_call(["x"], {}, "chave", "grade")
    assert solver._ULTIMA_CHAMADA["provedor"] == f"Gemini ({modelo})"
    texto = solver._tempos_da_rodada(inicio)
    assert texto.startswith(f"Gemini ({modelo}) respondeu em ")
    assert "preparo e espera" in texto


def test_preparo_e_o_que_nao_foi_a_chamada(monkeypatch):
    agora = [0.0]
    monkeypatch.setattr(solver.time, "monotonic", lambda: agora[0])
    inicio = solver._comecar_rodada()
    agora[0] = 21.0
    solver._ULTIMA_CHAMADA.update(provedor="Astra", segundos=9.0)
    assert solver._tempos_da_rodada(inicio) == "Astra respondeu em 9s · preparo e espera 12s"


def test_o_astra_registra_a_propria_resposta():
    arvore = ast.parse(pathlib.Path(solver.__file__).read_text(encoding="utf-8"))
    astra = next(n for n in ast.walk(arvore)
                 if isinstance(n, ast.FunctionDef) and n.name == "_astra_call")
    # `ast.unparse` normaliza as aspas para simples.
    assert "_ULTIMA_CHAMADA.update(provedor='Astra'" in ast.unparse(astra)


def test_as_tres_respostas_de_rodada_trazem_os_tempos():
    arvore = ast.parse(pathlib.Path(solver.__file__).read_text(encoding="utf-8"))
    for nome in ("_solve_grade", "_solve_grade_fused", "_solve_imagem"):
        funcao = next(n for n in ast.walk(arvore)
                      if isinstance(n, ast.FunctionDef) and n.name == nome)
        fonte = ast.unparse(funcao)
        assert "_inicio_rodada = _comecar_rodada()" in fonte, nome
        assert "_tempos_da_rodada(_inicio_rodada)" in fonte, nome
