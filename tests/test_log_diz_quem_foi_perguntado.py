"""O log diz QUEM foi perguntado — pedido do Jean, 15/09/2026.

"Na minha cabeça era o Gemini sendo chamado, não resolvendo e o Astra entrando
em cena." A frase

    [captcha/imagem] Gemini não fechou — perguntando ao segundo provedor.

saía em TODA chamada ao Astra, inclusive nos tipos em que ele é o primeiro a
ser perguntado (imagem, grade de pontos, bola, cartão). O log contava uma
história que não tinha acontecido.

As frases proibidas são procuradas nas CHAMADAS de print, por AST: os
comentários citam logs antigos de propósito, e testar o texto cru daria falso
positivo com eles.
"""
import ast
import json
import pathlib

import pytest

from resolvedor_captcha import solver


def _mensagens_impressas() -> list[str]:
    arvore = ast.parse(pathlib.Path(solver.__file__).read_text(encoding="utf-8"))
    return [ast.unparse(n) for n in ast.walk(arvore)
            if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "print"]


def test_nenhuma_mensagem_chama_o_astra_de_segundo_provedor():
    """A única exceção é o aviso de chave ausente, que fala da configuração."""
    ruins = [m for m in _mensagens_impressas()
             if "segundo provedor" in m and "SEM segundo provedor" not in m]
    assert ruins == []


def test_a_frase_que_enganava_nao_volta():
    assert not [m for m in _mensagens_impressas() if "Gemini não fechou" in m]


def test_a_chamada_ao_astra_se_anuncia_pelo_nome():
    assert any("perguntando ao Astra." in m for m in _mensagens_impressas())


class _Resposta:
    text = json.dumps({"matching_tiles": [2], "confidence": "high"})


class _Cliente:
    class models:  # noqa: N801 — imita o SDK
        @staticmethod
        def generate_content(**_k):
            return _Resposta()


def test_a_chamada_ao_gemini_diz_o_modelo(monkeypatch, capsys):
    """Executa `_gemini_call` com um cliente dublê e lê o que foi impresso."""
    monkeypatch.setattr(solver, "_astra_configurado", lambda: False)
    monkeypatch.setattr(solver, "_get_client", lambda _k: _Cliente())
    modelo = solver.modelos_ativos()[0]
    solver._gemini_call(["x"], {}, "chave", "grade")
    saida = capsys.readouterr().out
    assert f"[captcha/grade] perguntando ao Gemini ({modelo})." in saida
    assert "Astra" not in saida


def test_o_atalho_da_memoria_diz_que_pulou_o_gemini(monkeypatch, capsys):
    monkeypatch.setattr(solver, "_astra_configurado", lambda: True)
    monkeypatch.setattr(solver, "_astra_call",
                        lambda *a, **k: {"matching_tiles": [1], "confidence": "high"})
    solver._marcar_gemini_nao_fechou("grade")
    solver._gemini_call(["x"], {}, "chave", "grade")
    saida = capsys.readouterr().out
    assert "[captcha/grade] Gemini pulado:" in saida
    assert "perguntando ao Gemini" not in saida


@pytest.mark.parametrize("narracao", [
    "Tile {idx} clicado",
    "Submetendo desafio...",
    "Screenshot capturado",
    "Tiles recortados",
    "Via DOM bounds",
    "Nenhum captcha na página",
])
def test_narracao_de_passo_que_deu_certo_saiu(narracao):
    """Contagem das runs de 14 e 15/09/2026: 123 "Tile N clicado", 102
    "Submetendo desafio...", 120 de screenshot/recorte."""
    assert not [m for m in _mensagens_impressas() if narracao in m]
