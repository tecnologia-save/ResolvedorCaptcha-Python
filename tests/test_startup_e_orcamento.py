"""Inicio da resolucao e orcamento de tempo — sem rede, sem navegador.

Execucao real na representacao de CNPJ: a representacao foi pedida as 16:17:16,
o desafio grade apareceu as 16:17:27, dois modelos consecutivos deram
ReadTimeout, o terceiro respondeu, e o captcha so terminou perto das 16:18:45.
O portal entao nao confirmou a representacao.

Duas causas de tempo, independentes:

  1. `solve_hcaptcha` esperava o checkbox por ate 10 s ANTES de procurar
     desafio ativo. Numa grade que ja vem aberta, sao 10 s por nada;
  2. o teto de 30 s por chamada e o certo para o captcha de login e caro demais
     para a representacao — dois timeouts consecutivos consomem um minuto antes
     de o terceiro modelo ser tentado.

Dados sinteticos. Nenhuma pagina, chave ou resposta de cliente.
"""
import inspect
import json

import pytest

from resolvedor_captcha import solver

# ── Pagina falsa ─────────────────────────────────────────────────────────────

class _Locator:
    """Locator com N iframes de checkbox, cada um com sua visibilidade."""

    def __init__(self, visiveis, idx=None):
        # visiveis: lista de bool, um por iframe
        self._visiveis = list(visiveis)
        self._idx = idx

    @property
    def first(self):
        return self.nth(0)

    def nth(self, i):
        return _Locator(self._visiveis, idx=i)

    def count(self):
        return len(self._visiveis)

    def is_visible(self):
        i = 0 if self._idx is None else self._idx
        return bool(self._visiveis[i]) if i < len(self._visiveis) else False

    def wait_for(self, **_k):
        if not any(self._visiveis):
            raise TimeoutError("nao apareceu")

    def click(self, **_k):
        pass


class _FrameLocator:
    """`frame_locator(...).nth(i)` — registra QUAL iframe foi clicado."""

    def __init__(self, pagina, idx=None):
        self.pagina, self.idx = pagina, idx

    def nth(self, i):
        return _FrameLocator(self.pagina, idx=i)

    @property
    def first(self):
        return self

    def locator(self, _sel):
        return _BotaoDoFrame(self.pagina, self.idx)


class _BotaoDoFrame:
    def __init__(self, pagina, idx):
        self.pagina, self.idx = pagina, idx

    @property
    def first(self):
        return self

    def click(self, **_k):
        self.pagina.clicados.append(self.idx)


class _Pagina:
    """So o que o inicio da resolucao consulta."""

    def __init__(self, *, desafio=False, checkbox=False, desafio_apos=None,
                 checkboxes=None):
        self.desafio = desafio
        # `checkboxes` = visibilidade de cada iframe; `checkbox` e o atalho de
        # um so.
        self.checkboxes = (list(checkboxes) if checkboxes is not None
                           else ([True] if checkbox else []))
        self.clicados = []
        self.desafio_apos = desafio_apos     # aparece apos N consultas
        self.consultas_de_desafio = 0

    def locator(self, _sel):
        return _Locator(self.checkboxes)

    def frame_locator(self, _sel):
        return _FrameLocator(self)

    def wait_for_timeout(self, _ms):
        pass


@pytest.fixture
def pagina_observavel(monkeypatch):
    """`_get_challenge_frame` responde conforme o estado da pagina falsa."""
    def frame(page):
        page.consultas_de_desafio += 1
        if page.desafio:
            return object()
        if (page.desafio_apos is not None
                and page.consultas_de_desafio > page.desafio_apos):
            page.desafio = True
            return object()
        return None

    monkeypatch.setattr(solver, "_get_challenge_frame", frame)
    monkeypatch.setattr(solver.time, "sleep", lambda _s: None)


# ══ 1 · Checkbox OU desafio, no mesmo prazo ═════════════════════════════════

def test_desafio_ja_ativo_e_reconhecido_de_imediato(pagina_observavel):
    """CASO A: grade aberta, sem checkbox. Nao ha 10 s a perder."""
    pagina = _Pagina(desafio=True)
    assert solver._aguardar_desafio_ou_checkbox(pagina) == solver.INICIO_DESAFIO
    assert pagina.consultas_de_desafio == 1     # a primeira consulta ja decidiu


def test_checkbox_visivel_sem_desafio(pagina_observavel):
    """CASO B: o widget esta la e o desafio ainda nao."""
    pagina = _Pagina(checkbox=True)
    assert solver._aguardar_desafio_ou_checkbox(pagina) == solver.INICIO_CHECKBOX


def test_desafio_vence_o_checkbox_quando_ambos_existem(pagina_observavel):
    """CASO C: clicar num widget antigo com o desafio aberto nao adianta."""
    pagina = _Pagina(desafio=True, checkbox=True)
    assert solver._aguardar_desafio_ou_checkbox(pagina) == solver.INICIO_DESAFIO


def test_nenhum_dos_dois_expira_um_unico_prazo(pagina_observavel):
    """CASO D: um prazo, nao dois em sequencia."""
    pagina = _Pagina()
    assert solver._aguardar_desafio_ou_checkbox(
        pagina, timeout_ms=300) == solver.INICIO_NENHUM


def test_desafio_que_aparece_durante_a_espera_e_pego(pagina_observavel):
    pagina = _Pagina(desafio_apos=3)
    assert solver._aguardar_desafio_ou_checkbox(pagina) == solver.INICIO_DESAFIO


def test_o_startup_nao_espera_checkbox_antes_de_procurar_desafio():
    """Gate estrutural: `solve_hcaptcha` nao pode voltar a serializar.

    O defeito era uma linha — `_click_checkbox_widget(page, timeout_ms=10_000)`
    incondicional, antes de qualquer deteccao.
    """
    fonte = inspect.getsource(solver.solve_hcaptcha)
    assert "_aguardar_desafio_ou_checkbox" in fonte
    assert "_click_checkbox_widget(page, timeout_ms=10_000)" not in fonte
    # O clique so acontece DEPOIS da observacao, no ramo do checkbox.
    assert (fonte.index("_click_checkbox_widget")
            > fonte.index("_aguardar_desafio_ou_checkbox"))


# ══ 2 · A prioridade da classificacao nao mudou ═════════════════════════════

def test_grade_continua_sendo_a_primeira_hipotese():
    """9+ `.task` decide antes de qualquer outro sinal."""
    fonte = inspect.getsource(solver._detect_challenge_type)
    pos_grade = fonte.index('return "grade"')
    assert pos_grade < fonte.index('return "cartao_animal"')
    assert pos_grade < fonte.index('return "grade_fused"')


def test_cartao_animal_ainda_e_checado_antes_de_fused():
    """PROTECAO PRESERVADA: cartao_animal comeca com 0 tiles.

    Se `grade_fused` fosse avaliado antes, um cartao_animal com 1-8 elementos
    viraria fused — falso positivo que "grade primeiro" nao autoriza.
    """
    fonte = inspect.getsource(solver._detect_challenge_type)
    assert fonte.index('return "cartao_animal"') < fonte.index('return "grade_fused"')


# ══ 3 · Orcamento de tempo ══════════════════════════════════════════════════

def test_sem_parametros_a_politica_e_a_de_sempre():
    p = solver.PoliticaLatencia()
    assert p.timeout_ms == solver.GEMINI_TIMEOUT_MS
    assert p.fim is None
    assert p.esgotado is False
    assert p.timeout_efetivo_ms() == solver.GEMINI_TIMEOUT_MS


def test_o_deadline_total_manda_no_timeout_individual(monkeypatch):
    """Com 8 s sobrando, nenhuma chamada pode reservar 30 s."""
    agora = [1000.0]
    monkeypatch.setattr(solver.time, "monotonic", lambda: agora[0])
    p = solver.PoliticaLatencia(timeout_ms=30_000, fim=agora[0] + 8.0)
    assert p.timeout_efetivo_ms() == 8_000


def test_timeout_individual_menor_que_o_deadline_prevalece(monkeypatch):
    agora = [1000.0]
    monkeypatch.setattr(solver.time, "monotonic", lambda: agora[0])
    p = solver.PoliticaLatencia(timeout_ms=10_000, fim=agora[0] + 25.0)
    assert p.timeout_efetivo_ms() == 10_000


def test_orcamento_esgotado_e_reconhecido(monkeypatch):
    agora = [1000.0]
    monkeypatch.setattr(solver.time, "monotonic", lambda: agora[0])
    p = solver.PoliticaLatencia(timeout_ms=10_000, fim=agora[0] + 5.0)
    assert p.esgotado is False
    agora[0] += 6.0
    assert p.esgotado is True
    assert p.timeout_efetivo_ms() == 1        # nunca zero nem negativo


def test_a_config_carrega_o_timeout_da_politica():
    config = solver._make_config(solver._SCHEMA_GRADE, solver.GEMINI_MODELS[0],
                                 timeout_ms=10_000)
    assert config.http_options.timeout == 10_000


def test_a_config_sem_timeout_explicito_mantem_o_padrao():
    config = solver._make_config(solver._SCHEMA_GRADE, solver.GEMINI_MODELS[0])
    assert config.http_options.timeout == solver.GEMINI_TIMEOUT_MS


# ── A cadeia de modelos respeita o orcamento ─────────────────────────────────

class _Resposta:
    def __init__(self, texto):
        self.text = texto


def _cliente(comportamento, relogio, custo_s=0.0):
    """Cliente falso; cada chamada consome `custo_s` do relogio virtual."""
    chamadas = []

    class _Models:
        def generate_content(self, model, contents, config):
            chamadas.append((model, config.http_options.timeout))
            relogio[0] += custo_s
            acao = comportamento.get(model)
            if isinstance(acao, BaseException):
                raise acao
            return _Resposta(json.dumps(acao if acao is not None else {"ok": True}))

    class _Cliente:
        models = _Models()

    return _Cliente(), chamadas


@pytest.fixture
def relogio(monkeypatch):
    agora = [1000.0]
    monkeypatch.setattr(solver.time, "monotonic", lambda: agora[0])
    monkeypatch.setattr(solver.time, "sleep",
                        lambda s: agora.__setitem__(0, agora[0] + s))
    return agora


def test_a_run_real_reproduzida_dois_timeouts_e_o_terceiro_responde(monkeypatch, relogio):
    """Com a politica rapida, os tres modelos cabem no orcamento.

    Cada timeout consome o teto individual (10 s), nao os 30 s do padrao — e o
    terceiro so ganha o que ainda sobra do total.
    """
    cliente, chamadas = _cliente({
        solver.GEMINI_MODELS[0]: TimeoutError("ReadTimeout"),
        solver.GEMINI_MODELS[1]: TimeoutError("ReadTimeout"),
        solver.GEMINI_MODELS[2]: {"ok": 3},
    }, relogio, custo_s=10.0)
    monkeypatch.setattr(solver, "_get_client", lambda _k: cliente)

    inicio = relogio[0]
    politica = solver.PoliticaLatencia(timeout_ms=10_000, fim=inicio + 25.0)
    assert solver._gemini_call([], {}, "k", "grade", politica) == {"ok": 3}
    assert [t for _m, t in chamadas] == [10_000, 10_000, 5_000]
    assert relogio[0] - inicio <= 30.0


def test_o_orcamento_esgotado_interrompe_a_cadeia(monkeypatch, relogio):
    """Nao adianta ir ao proximo modelo com o screenshot ja velho."""
    cliente, chamadas = _cliente(
        {m: TimeoutError("ReadTimeout") for m in solver.GEMINI_MODELS},
        relogio, custo_s=10.0)
    monkeypatch.setattr(solver, "_get_client", lambda _k: cliente)

    politica = solver.PoliticaLatencia(timeout_ms=10_000, fim=relogio[0] + 15.0)
    with pytest.raises(RuntimeError):
        solver._gemini_call([], {}, "k", "grade", politica)
    assert len(chamadas) == 2          # o terceiro nao chegou a ser tentado


def test_sem_politica_a_cadeia_usa_o_teto_de_sempre(monkeypatch, relogio):
    """Os demais consumidores nao mudam de comportamento."""
    cliente, chamadas = _cliente({solver.GEMINI_MODELS[0]: {"ok": 1}}, relogio)
    monkeypatch.setattr(solver, "_get_client", lambda _k: cliente)
    assert solver._gemini_call([], {}, "k", "grade") == {"ok": 1}
    assert chamadas == [(solver.GEMINI_MODELS[0], solver.GEMINI_TIMEOUT_MS)]


def test_solve_hcaptcha_sem_parametros_nao_impoe_deadline():
    """Assinatura retrocompativel: o default continua sendo o de antes."""
    parametros = inspect.signature(solver.solve_hcaptcha).parameters
    assert parametros["gemini_timeout_ms"].default is None
    assert parametros["deadline_s"].default is None
    assert parametros["gemini_timeout_ms"].kind == inspect.Parameter.KEYWORD_ONLY
    assert parametros["deadline_s"].kind == inspect.Parameter.KEYWORD_ONLY


def test_a_lista_de_modelos_nao_foi_reordenada():
    """Uma run nao derruba uma campanha de medicao."""
    assert solver.GEMINI_MODELS == [
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash",
        "gemini-3.1-flash-lite",
    ]


# ══ 4 · O freshness guard nao foi tocado ════════════════════════════════════

def test_o_freshness_guard_continua_no_lugar():
    for nome in ("_fingerprint_desafio", "_desafio_ainda_e_o_mesmo",
                 "_capturar_desafio"):
        assert hasattr(solver, nome), nome
    assert "_desafio_ainda_e_o_mesmo" in inspect.getsource(solver._solve_grade)


# ══ 5 · Presenca de captcha exige VISIBILIDADE ══════════════════════════════
#
# `captcha_presente` respondia `page.locator(CHECKBOX_SEL).count() > 0`: existir
# no DOM bastava. O hCaptcha deixa seus iframes para tras, e no portal Servicos
# RF um captcha antecede o outro — login e depois representacao. O widget da
# etapa anterior continua no documento, e o integrador era mandado para um ramo
# de captcha que nao existia mais.
#
# Nao ha prova de que foi isso que derrubou a run de 16:56; ha prova de que o
# mecanismo existe e produz exatamente aquele sintoma silencioso.

class _CheckboxLocator:
    """N iframes de checkbox, cada um com sua propria visibilidade."""

    def __init__(self, visiveis, idx=None):
        self._visiveis = list(visiveis)
        self._idx = idx

    @property
    def first(self):
        return self.nth(0)

    def nth(self, i):
        return _CheckboxLocator(self._visiveis, idx=i)

    def count(self):
        return len(self._visiveis)

    def is_visible(self):
        i = 0 if self._idx is None else self._idx
        return bool(self._visiveis[i]) if i < len(self._visiveis) else False


class _PaginaCaptcha:
    def __init__(self, *, challenge=False, checkbox_existe=False,
                 checkbox_visivel=False, checkboxes=None):
        self.challenge = challenge
        if checkboxes is not None:
            self.checkboxes = list(checkboxes)
        elif checkbox_existe:
            self.checkboxes = [bool(checkbox_visivel)]
        else:
            self.checkboxes = []

    def locator(self, _sel):
        return _CheckboxLocator(self.checkboxes)


@pytest.fixture
def challenge_conforme(monkeypatch):
    monkeypatch.setattr(solver, "_challenge_visible", lambda p: p.challenge)


def test_challenge_ativo_e_captcha_presente(challenge_conforme):
    assert solver.captcha_presente(_PaginaCaptcha(challenge=True)) is True


def test_checkbox_visivel_e_captcha_presente(challenge_conforme):
    assert solver.captcha_presente(
        _PaginaCaptcha(checkbox_existe=True, checkbox_visivel=True)) is True


def test_checkbox_existente_porem_oculto_nao_e_captcha(challenge_conforme):
    """O caso do iframe deixado para tras pelo captcha anterior."""
    assert solver.captcha_presente(
        _PaginaCaptcha(checkbox_existe=True, checkbox_visivel=False)) is False


def test_pagina_sem_captcha_algum(challenge_conforme):
    assert solver.captcha_presente(_PaginaCaptcha()) is False


def test_challenge_ativo_vence_checkbox_oculto(challenge_conforme):
    """Desafio aberto decide antes de qualquer coisa sobre o widget."""
    assert solver.captcha_presente(
        _PaginaCaptcha(challenge=True, checkbox_existe=True,
                       checkbox_visivel=False)) is True


def test_presenca_nao_se_apoia_mais_em_count():
    """Gate: `count() > 0` nao pode voltar a ser a prova."""
    fonte = inspect.getsource(solver.captcha_presente)
    corpo = fonte[fonte.rindex('"""') + 3:]        # so o codigo, sem o docstring
    assert "count()" not in corpo
    assert "_indice_checkbox_visivel" in corpo


# ══ 6 · Checkbox stale ANTES do atual — o `.first` de novo ══════════════════
#
# A correcao do 1.0.7 trocou `count() > 0` por `.first.is_visible()`, e isso
# ainda erra no proprio cenario que a motivou: o hCaptcha mantem mais de um
# iframe de widget, e o obsoleto pode vir PRIMEIRO na ordem do documento.
# `.first` responde pelo oculto e conclui "nao ha captcha" com um widget real
# esperando na tela.

STALE_E_ATUAL = [False, True]      # [0] obsoleto e oculto, [1] atual e visivel


def test_checkbox_stale_na_frente_nao_esconde_o_atual(challenge_conforme):
    """RED A: o segundo iframe esta visivel — ha captcha."""
    assert solver.captcha_presente(
        _PaginaCaptcha(checkboxes=STALE_E_ATUAL)) is True


def test_o_indice_devolvido_e_o_do_iframe_visivel(challenge_conforme):
    assert solver._indice_checkbox_visivel(
        _PaginaCaptcha(checkboxes=STALE_E_ATUAL)) == 1


def test_todos_os_iframes_ocultos_nao_sao_captcha(challenge_conforme):
    """RED D."""
    assert solver.captcha_presente(
        _PaginaCaptcha(checkboxes=[False, False])) is False
    assert solver._indice_checkbox_visivel(
        _PaginaCaptcha(checkboxes=[False, False])) is None


def test_challenge_ativo_vence_checkbox_stale(challenge_conforme):
    """RED E: desafio aberto decide antes de qualquer coisa sobre widgets."""
    assert solver.captcha_presente(
        _PaginaCaptcha(challenge=True, checkboxes=STALE_E_ATUAL)) is True


def test_o_startup_ve_o_checkbox_atual_mesmo_com_stale_na_frente(pagina_observavel):
    """RED B: o inicio responde CHECKBOX, e nao 'nenhum'."""
    pagina = _Pagina(checkboxes=STALE_E_ATUAL)
    assert solver._aguardar_desafio_ou_checkbox(pagina) == solver.INICIO_CHECKBOX


def test_o_clique_acontece_no_iframe_detectado(pagina_observavel):
    """RED C: detectar um iframe e clicar outro seria trocar um erro por outro."""
    pagina = _Pagina(checkboxes=STALE_E_ATUAL)
    assert solver._click_checkbox_widget(pagina) is True
    assert pagina.clicados == [1]


def test_o_clique_usa_o_primeiro_visivel_quando_ha_varios(pagina_observavel):
    """Politica deterministica: primeiro visivel em ordem de documento."""
    pagina = _Pagina(checkboxes=[False, True, True])
    solver._click_checkbox_widget(pagina)
    assert pagina.clicados == [1]


def test_nenhuma_deteccao_de_checkbox_se_apoia_em_first():
    """Gate: `.first` nao pode voltar a governar o widget."""
    for funcao in (solver.captcha_presente, solver._checkbox_visivel,
                   solver._indice_checkbox_visivel,
                   solver._click_checkbox_widget):
        fonte = inspect.getsource(funcao)
        corpo = fonte[fonte.rindex('"""') + 3:] if '"""' in fonte else fonte
        assert "CHECKBOX_SEL).first" not in corpo, funcao.__name__


def test_as_tres_funcoes_usam_a_mesma_resolucao():
    """Uma unica fonte de verdade sobre qual widget vale."""
    for funcao in (solver.captcha_presente, solver._checkbox_visivel,
                   solver._click_checkbox_widget):
        assert "_indice_checkbox_visivel" in inspect.getsource(funcao), \
            funcao.__name__


# ══ 7 · `abrir_desafio` — abrir nao e resolver ══════════════════════════════

def test_abrir_desafio_com_challenge_ja_ativo_nao_clica(pagina_observavel):
    pagina = _Pagina(desafio=True, checkboxes=[True])
    assert solver.abrir_desafio(pagina) is True
    assert pagina.clicados == []


def test_abrir_desafio_clica_o_widget_e_espera_o_challenge(pagina_observavel):
    pagina = _Pagina(checkboxes=STALE_E_ATUAL, desafio_apos=1)
    assert solver.abrir_desafio(pagina) is True
    assert pagina.clicados == [1]


def test_abrir_desafio_sem_widget_algum_e_falso(pagina_observavel):
    assert solver.abrir_desafio(_Pagina(), timeout_ms=200) is False


def test_abrir_desafio_nao_resolve_nada(monkeypatch, pagina_observavel):
    """A fronteira e o ponto: nenhuma chamada ao modelo, nenhum tile clicado."""
    def proibido(*_a, **_k):
        raise AssertionError("abrir_desafio nao pode resolver")

    monkeypatch.setattr(solver, "_gemini_call", proibido)
    monkeypatch.setattr(solver, "_solve_grade", proibido)
    monkeypatch.setattr(solver, "_solve_cartao_animal", proibido)
    pagina = _Pagina(checkboxes=[True], desafio_apos=1)
    assert solver.abrir_desafio(pagina) is True


def test_abrir_desafio_e_publico():
    import resolvedor_captcha as pacote
    assert "abrir_desafio" in pacote.__all__
    assert pacote.abrir_desafio is solver.abrir_desafio
