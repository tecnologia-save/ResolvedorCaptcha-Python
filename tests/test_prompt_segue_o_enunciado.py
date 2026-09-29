"""O prompt do tipo `imagem` obedece ao enunciado, em vez de sobrepor o seu.

29/09/2026, captcha trazido pelo Jean: "Clique no caractere parcialmente coberto
por uma linha", quatro bichos sobre fundo quadriculado. O prompt citava o
enunciado e, na frase seguinte, mandava "compare as figuras ENTRE SI e escolha a
que destoa das demais". Entre tigre, capivara, polvo e ovelha nenhuma destoa: o
critério estava na tela e na instrução, e o modelo recebia ordem de ignorar os
dois.

A tarefa fixa não some — ela foi medida contra as amostras de "figura diferente"
em 09/09/2026 e continua valendo para elas, e para o caso sem enunciado legível.
"""
import pytest

from resolvedor_captcha import solver


COBERTO = "Clique no caractere parcialmente coberto por uma linha"
DIFERENTE = "Por favor, clique na figura diferente"
PADRAO = "Por favor, clique no ícone que quebra o padrão"


def _prompt(monkeypatch, instrucao):
    """Monta o prompt real de `_gemini_pixel`, sem chamar provedor nenhum."""
    capturado = {}

    def falso_call(partes, *a, **k):
        capturado["texto"] = next(p for p in partes if isinstance(p, str))
        return {"x": 1, "y": 1}

    monkeypatch.setattr(solver, "_gemini_call", falso_call)
    monkeypatch.setattr(solver, "_parte_imagem", lambda png: {"img": True})
    monkeypatch.setattr(solver, "_dimensoes_png", lambda png: (400, 300))
    solver._gemini_pixel(b"png", instrucao, "chave-de-teste")
    return capturado["texto"]


# ══ 1 · O enunciado manda ═══════════════════════════════════════════════════

def test_enunciado_com_criterio_proprio_vira_a_tarefa(monkeypatch):
    p = _prompt(monkeypatch, COBERTO)
    assert COBERTO in p
    assert "Faça exatamente o que o enunciado pede" in p
    assert "destoa das demais" not in p, "a tarefa antiga contradizia o enunciado"


def test_o_modelo_e_avisado_a_nao_procurar_a_diferente(monkeypatch):
    """Sem isto o modelo cai no hábito: "figura diferente" é o captcha mais
    comum da família, e o prompt antigo o treinava a responder sempre isso."""
    p = _prompt(monkeypatch, COBERTO)
    assert "não procure a figura" in p


# ══ 2 · A família medida não regride ════════════════════════════════════════

@pytest.mark.parametrize("instrucao", [DIFERENTE, PADRAO,
                                       "Clique na imagem que não pertence ao grupo"])
def test_figura_diferente_mantem_a_tarefa_medida(monkeypatch, instrucao):
    p = _prompt(monkeypatch, instrucao)
    assert "Compare as figuras ENTRE SI" in p
    assert "destoa das demais" in p


def test_sem_enunciado_legivel_mantem_o_comportamento_antigo(monkeypatch):
    """Sem texto não há critério, e "a que destoa" é o palpite já medido."""
    p = _prompt(monkeypatch, "")
    assert "Compare as figuras ENTRE SI" in p


@pytest.mark.parametrize("instrucao, diferente", [
    (DIFERENTE, True),
    (PADRAO, True),
    ("Clique na figura fora do padrão", True),
    (COBERTO, False),
    ("Clique no animal que a bola nunca toca", False),
    ("Clique no objeto que consegue rolar numa superfície plana", False),
    ("", False),
])
def test_a_familia_e_reconhecida_por_marca_curta(instrucao, diferente):
    assert solver._pede_a_figura_diferente(instrucao) is diferente


# ══ 3 · O resto do prompt continua de pé ════════════════════════════════════

def test_o_formato_da_resposta_nao_mudou(monkeypatch):
    """As coordenadas e o limite de um clique valem para os dois caminhos."""
    for instrucao in (COBERTO, DIFERENTE):
        p = _prompt(monkeypatch, instrucao)
        assert "400x300 pixels" in p
        assert "x de 0 a 399" in p and "y de 0 a 299" in p
        assert "0,0 no canto superior esquerdo" in p
        assert "Um ponto só" in p
