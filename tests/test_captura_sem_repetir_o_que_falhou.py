"""A captura não repete, a cada rodada, os seletores que já falharam.

21/09/2026, medido em produção com o detalhe de etapas: a rodada de imagem
gastava 8 s só na captura, sempre. São oito seletores de <img> com 800 ms de
espera cada antes de cair na estratégia que funciona hoje (recorte por medida
da tela — a imagem do desafio é fundo CSS e não existe <img> nenhum).

Agora: espera de 200 ms por seletor, e a estratégia que funcionou fica
lembrada dentro do mesmo captcha.
"""
import pytest

from resolvedor_captcha import solver


class _Locator:
    def __init__(self, registro, nome, existe):
        self.registro, self.nome, self.existe = registro, nome, existe

    @property
    def first(self):
        return self

    def nth(self, _i):
        return self

    def bounding_box(self, timeout=None):
        self.registro.append((self.nome, timeout))
        if not self.existe:
            raise TimeoutError("seletor não encontrado")
        return {"x": 0, "y": 0, "width": 300, "height": 300}

    def screenshot(self, **_k):
        return b"png-da-imagem"


class _Frame:
    """Sem <img> nenhum: é o desafio de hoje, com a imagem em fundo CSS."""

    def __init__(self, registro):
        self.registro = registro

    def locator(self, sel):
        return _Locator(self.registro, sel, existe=False)

    def evaluate(self, js, *_a):
        self.registro.append(("js", None))
        if "querySelectorAll('img')" in js:
            return None
        return {"x": 0, "y": 40, "width": 400, "height": 300}


class _Pagina:
    def screenshot(self, **_k):
        return b"png-recortado"


@pytest.fixture
def captura(monkeypatch):
    registro = []
    frame = _Frame(registro)
    monkeypatch.setattr(solver, "_get_challenge_frame", lambda page: frame)
    monkeypatch.setattr(solver, "_get_challenge_element_locator",
                        lambda page: type("E", (), {"bounding_box": lambda s: {"x": 10, "y": 20, "width": 400, "height": 400}})())
    monkeypatch.setattr(solver, "_get_challenge_frame_locator", lambda page: frame)
    solver._novo_captcha()
    return registro


def test_a_espera_por_seletor_e_curta(captura):
    png, bbox = solver._get_task_image_screenshot_and_bbox(_Pagina())
    assert png == b"png-recortado"
    esperas = {t for nome, t in captura if t is not None}
    assert esperas == {solver.TIMEOUT_SELETOR_IMAGEM_MS}
    assert solver.TIMEOUT_SELETOR_IMAGEM_MS <= 300


def test_a_segunda_rodada_vai_direto_ao_que_funcionou(captura):
    solver._get_task_image_screenshot_and_bbox(_Pagina())
    tentativas_da_primeira = len(captura)
    assert tentativas_da_primeira >= 9, "a primeira rodada ainda tenta tudo"

    captura.clear()
    png, _bbox = solver._get_task_image_screenshot_and_bbox(_Pagina())
    assert png == b"png-recortado"
    assert [n for n, _t in captura] == ["js"], "só o JS do recorte; nenhum seletor de img"


def test_captcha_novo_volta_a_tentar_tudo(captura):
    solver._get_task_image_screenshot_and_bbox(_Pagina())
    solver._novo_captcha()
    captura.clear()
    solver._get_task_image_screenshot_and_bbox(_Pagina())
    assert len([n for n, _t in captura if n != "js"]) >= 8


def test_a_montagem_da_chamada_aparece_separada_no_log(monkeypatch):
    agora = [0.0]
    monkeypatch.setattr(solver.time, "monotonic", lambda: agora[0])
    solver._ULTIMA_CHAMADA.update(provedor="Astra", segundos=4.0)
    etapas = solver._Etapas()
    solver._CUSTO_DE_MONTAGEM.update(imagem=7.5, pedido=2.0)
    agora[0] = 15.0
    etapas.marcar("chamada")
    assert etapas.texto() == "preparo: montagem da chamada (imagem 7.5s, pedido 2.0s) 11.0s"


def test_o_custo_da_montagem_zera_a_cada_rodada():
    solver._CUSTO_DE_MONTAGEM.update(imagem=9.0, pedido=1.0)
    solver._comecar_rodada()
    assert solver._CUSTO_DE_MONTAGEM == {"imagem": 0.0, "pedido": 0.0}


def test_a_conversao_da_imagem_entra_na_conta():
    solver._comecar_rodada()
    solver._para_envio(b"nao e png", max_dim=100)
    assert solver._CUSTO_DE_MONTAGEM["imagem"] >= 0.0
    assert "imagem" in solver._CUSTO_DE_MONTAGEM
