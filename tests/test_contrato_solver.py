"""Contrato do solver que ja existia — a rede de seguranca da correcao.

Estes testes nao descrevem comportamento novo: descrevem o que o solver ja fazia
antes do freshness guard, para que a correcao possa ser feita sem quebrar a
deteccao de frame ativo, o fallback de modelo ou o clique nos tiles.

Sem hCaptcha real, sem Gemini real, sem rede.
"""
import json

import pytest
from fakes import Captcha, Desafio, FakePage

from resolvedor_captcha import solver

# ── Deteccao do frame ATIVO entre varios pre-carregados ──────────────────────

def test_encontra_o_frame_ativo_e_ignora_os_pre_carregados(page, captcha):
    """O hCaptcha deixa varios `frame=challenge` no DOM; so um vale."""
    assert captcha.n_iframes == 2
    frame = solver._get_challenge_frame(page)
    assert frame is not None
    assert solver._get_active_iframe_index(page) == captcha.idx_ativo


def test_sem_desafio_ativo_nao_ha_frame(page, captcha):
    captcha.resolver()
    assert solver._get_challenge_frame(page) is None
    assert solver._challenge_visible(page) is False


def test_challenge_visible_segue_o_estado_do_captcha(page, captcha):
    assert solver._challenge_visible(page) is True
    captcha.resolver()
    assert solver._challenge_visible(page) is False


# ── Classificacao de erro do provedor ────────────────────────────────────────

@pytest.mark.parametrize("mensagem", [
    "503 UNAVAILABLE", "model is overloaded", "429 RESOURCE_EXHAUSTED",
    "404 model not found", "no longer available",
])
def test_erros_que_justificam_trocar_de_modelo(mensagem):
    assert solver._is_overloaded_error(Exception(mensagem)) is True


@pytest.mark.parametrize("mensagem", [
    "400 INVALID_ARGUMENT", "401 unauthorized", "API key not valid",
])
def test_erros_que_nao_justificam_trocar_de_modelo(mensagem):
    assert solver._is_overloaded_error(Exception(mensagem)) is False


# ── Fallback entre modelos ───────────────────────────────────────────────────

class _Resposta:
    def __init__(self, texto):
        self.text = texto


def _cliente(comportamento):
    """Cliente falso do google.genai. `comportamento` mapeia modelo -> acao."""
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


def test_sucesso_no_primeiro_modelo_nao_tenta_os_demais(monkeypatch):
    cliente, chamadas = _cliente({solver.GEMINI_MODELS[0]: {"ok": 1}})
    monkeypatch.setattr(solver, "_get_client", lambda _k: cliente)
    assert solver._gemini_call([], {}, "k", "grade") == {"ok": 1}
    assert chamadas == [solver.GEMINI_MODELS[0]]


def test_modelo_sobrecarregado_cai_para_o_proximo(monkeypatch):
    cliente, chamadas = _cliente({
        solver.GEMINI_MODELS[0]: RuntimeError("503 UNAVAILABLE"),
        solver.GEMINI_MODELS[1]: {"ok": 2},
    })
    monkeypatch.setattr(solver, "_get_client", lambda _k: cliente)
    assert solver._gemini_call([], {}, "k", "grade") == {"ok": 2}
    assert chamadas[0] == solver.GEMINI_MODELS[0]
    assert solver.GEMINI_MODELS[1] in chamadas


def test_erro_nao_sobrecarga_nao_percorre_todos_os_modelos(monkeypatch):
    """Chave invalida nao melhora trocando de modelo — e nao deve circular."""
    cliente, chamadas = _cliente({m: RuntimeError("400 INVALID_ARGUMENT")
                                  for m in solver.GEMINI_MODELS})
    monkeypatch.setattr(solver, "_get_client", lambda _k: cliente)
    with pytest.raises(RuntimeError):
        solver._gemini_call([], {}, "k", "grade")
    assert set(chamadas) == {solver.GEMINI_MODELS[0]}


def test_todos_indisponiveis_levanta(monkeypatch):
    cliente, _ = _cliente({m: RuntimeError("503 UNAVAILABLE")
                           for m in solver.GEMINI_MODELS})
    monkeypatch.setattr(solver, "_get_client", lambda _k: cliente)
    with pytest.raises(RuntimeError):
        solver._gemini_call([], {}, "k", "grade")


# ── Clique nos tiles ─────────────────────────────────────────────────────────

def test_clica_os_tiles_pedidos_uma_vez_cada(page, captcha):
    """Uma vez cada — a ORDEM deixou de ser fixa de propósito.

    Este teste comparava a sequência exata `[0, 4, 8]`, que era a ordem
    crescente de índice. Ela nunca foi requisito: o que ele guarda, e o nome
    diz, é que cada tile pedido é clicado uma única vez.

    Desde 10/09/2026 a ordem é embaralhada — clicar sempre na ordem do DOM é
    uma das constantes que denunciam automação. Comparar sequência aqui
    passaria a reprovar o comportamento correto.
    """
    solver._click_grade_tiles(page, [4, 0, 4, 8])
    clicados = [idx for _d, idx in captcha.tiles_clicados]
    assert sorted(clicados) == [0, 4, 8]
    assert len(clicados) == len(set(clicados)), "nenhum tile clicado duas vezes"


def test_lista_vazia_nao_clica(page, captcha):
    solver._click_grade_tiles(page, [])
    assert captcha.tiles_clicados == []


def test_clique_fused_usa_o_centro_de_cada_celula(page, captcha):
    bbox = {"x": 0.0, "y": 0.0, "width": 300.0, "height": 300.0}
    solver._click_fused_grade_tiles(page, [0, 8], bbox)
    coords = [(x, y) for _d, x, y in captcha.cliques_pixel]
    assert coords == [(50.0, 50.0), (250.0, 250.0)]


# ── Caminho feliz completo ───────────────────────────────────────────────────

def test_solve_grade_clica_e_submete_quando_o_desafio_nao_muda(
        page, captcha, gemini):
    """O caminho que precisa continuar funcionando depois da correcao."""
    gemini["resposta"] = {"task_summary": "onibus", "matching_tiles": [1, 3],
                          "confidence": "high"}
    solver._solve_grade(page, "chave-de-teste", max_rounds=1)
    # `sorted`: a ordem de clique é embaralhada. Com dois tiles este assert
    # passava metade das vezes — teste que falha em uma execução a cada duas
    # é pior que teste nenhum, porque ensina a ignorar o vermelho.
    assert sorted(idx for _d, idx in captcha.tiles_clicados) == [1, 3]
    assert captcha.submits >= 1
    # E clicou no desafio CERTO — o mesmo objeto que gerou a captura.
    assert all(d is captcha.desafio for d, _i in captcha.tiles_clicados)


def test_confianca_baixa_nao_clica(page, captcha, gemini):
    gemini["resposta"] = {"task_summary": "?", "matching_tiles": [1],
                          "confidence": "low"}
    solver._solve_grade(page, "chave-de-teste", max_rounds=1)
    assert captcha.tiles_clicados == []


def test_desafio_ja_resolvido_encerra_sem_clicar(page, captcha, gemini):
    captcha.resolver()
    assert solver._solve_grade(page, "chave-de-teste", max_rounds=1) is True
    assert captcha.tiles_clicados == []
    assert gemini["chamadas"] == 0


# ── Utilitario de texto ──────────────────────────────────────────────────────

def test_limpar_texto_colapsa_e_trunca():
    assert solver._limpar_texto("a\n\n   b") == "a b"
    assert solver._limpar_texto("x" * 300, max_len=10) == "x" * 10 + "…"


def test_fake_recusa_js_nao_modelado(captcha):
    """A honestidade do duble: script desconhecido falha alto."""
    frame = FakePage(captcha).frames[captcha.idx_ativo]
    with pytest.raises(AssertionError):
        frame.evaluate("() => document.querySelector('.inventado')")


def test_desafios_distintos_tem_pixels_distintos():
    """Premissa do freshness guard, registrada como contrato do duble."""
    a, b = Desafio(pixels=b"A"), Desafio(pixels=b"B")
    assert a.pixels != b.pixels
    c = Captcha(a)
    c.trocar_desafio(b)
    assert c.desafio is b and c.historico == [a, b]


# ── Deteccao publica, sem resolucao ──────────────────────────────────────────

def test_captcha_presente_detecta_desafio_aberto(page, captcha):
    assert solver.captcha_presente(page) is True


def test_captcha_presente_e_falso_quando_nao_ha_desafio(page, captcha):
    captcha.resolver()
    captcha.n_iframes = 0
    assert solver.captcha_presente(page) is False


def test_captcha_presente_detecta_o_widget_checkbox_fechado(page, captcha):
    """Desafio ainda nao aberto tambem e' fluxo parado esperando alguem."""
    captcha.resolver()             # nenhum frame=challenge ativo
    captcha.checkbox_presente = 1
    assert solver.captcha_presente(page) is True


def test_captcha_presente_ignora_checkbox_deixado_para_tras(page, captcha):
    """Iframe do captcha ANTERIOR: existe no DOM, nao esta na tela.

    No portal Servicos RF um captcha antecede o outro — login e depois
    representacao. Chamar o widget da etapa anterior de "captcha aguardando
    interacao" manda o integrador para um ramo que nao existe mais.
    """
    captcha.resolver()
    captcha.checkbox_presente = 1
    captcha.checkbox_visivel = False
    assert solver.captcha_presente(page) is False


def test_captcha_presente_nao_chama_o_solver(page, captcha, monkeypatch):
    """DETECCAO nao pode virar resolucao: nada de gastar chamada ao modelo
    nem clicar tile que ninguem pediu."""
    def proibido(*_a, **_k):
        raise AssertionError("captcha_presente nao pode resolver")

    monkeypatch.setattr(solver, "solve_hcaptcha", proibido)
    monkeypatch.setattr(solver, "_gemini_grade", proibido)
    monkeypatch.setattr(solver, "_click_grade_tiles", proibido)
    solver.captcha_presente(page)
    assert captcha.tiles_clicados == [] and captcha.submits == 0


def test_captcha_presente_nunca_levanta(monkeypatch):
    """Quem pergunta esta num estado incerto; excecao aqui vira ruido."""
    class _PaginaQuebrada:
        frames = property(lambda self: (_ for _ in ()).throw(RuntimeError("x")))

        def locator(self, _s):
            raise RuntimeError("x")

    assert solver.captcha_presente(_PaginaQuebrada()) is False


def test_detector_esta_na_api_publica():
    import resolvedor_captcha
    assert "captcha_presente" in resolvedor_captcha.__all__
    assert resolvedor_captcha.captcha_presente is solver.captcha_presente


# ── Classificacao publica do desafio ─────────────────────────────────────────

def test_tipos_conhecidos_sao_os_valores_reais_do_detector():
    """Vocabulario fechado, batendo com o que `_detect_challenge_type` devolve."""
    assert solver.TIPOS_CONHECIDOS == (
        "nenhum", "grade", "grade_fused", "bola_em_movimento",
        "cartao_animal", "imagem", "desconhecido")


def test_sem_desafio_devolve_nenhum(page, captcha):
    captcha.resolver()
    captcha.n_iframes = 0
    assert solver.detectar_tipo_captcha(page) == solver.TIPO_NENHUM


def test_grade_e_classificada(page, captcha, monkeypatch):
    monkeypatch.setattr(solver, "_detect_challenge_type",
                        lambda *_a, **_k: "grade")
    assert solver.detectar_tipo_captcha(page) == solver.TIPO_GRADE


@pytest.mark.parametrize("tipo", ["grade_fused", "cartao_animal", "imagem"])
def test_demais_tipos_sao_classificados(page, captcha, monkeypatch, tipo):
    monkeypatch.setattr(solver, "_detect_challenge_type",
                        lambda *_a, **_k: tipo)
    assert solver.detectar_tipo_captcha(page) == tipo


def test_classificacao_falha_devolve_desconhecido_e_nao_chuta_grade(
        page, captcha, monkeypatch):
    """A diferenca que importa para quem decide politica.

    Uso interno chuta `grade` e tenta; a API de inspecao nao pode, senao um
    formato nao classificavel entraria no caminho automatico por engano.
    """
    def explode(*_a, **_k):
        raise RuntimeError("frame morreu")

    monkeypatch.setattr(solver, "_detect_challenge_type", explode)
    assert solver.detectar_tipo_captcha(page) == solver.TIPO_DESCONHECIDO


def test_ao_falhar_preserva_o_comportamento_interno(page, captcha, monkeypatch):
    """`solve_hcaptcha` continua chutando `grade` — nada mudou para ele."""
    import inspect
    assert (inspect.signature(solver._detect_challenge_type)
            .parameters["ao_falhar"].default == solver.TIPO_GRADE)


def test_valor_fora_do_vocabulario_vira_desconhecido(page, captcha, monkeypatch):
    monkeypatch.setattr(solver, "_detect_challenge_type",
                        lambda *_a, **_k: "formato_novo_do_portal")
    assert solver.detectar_tipo_captcha(page) == solver.TIPO_DESCONHECIDO


def test_classificacao_nunca_levanta():
    class _PaginaQuebrada:
        frames = property(lambda self: (_ for _ in ()).throw(RuntimeError("x")))

        def locator(self, _s):
            raise RuntimeError("x")

    assert solver.detectar_tipo_captcha(_PaginaQuebrada()) == solver.TIPO_DESCONHECIDO


def test_classificar_nao_resolve(page, captcha, monkeypatch):
    """INSPECAO, nao resolucao: nenhum clique, nenhuma chamada ao modelo."""
    def proibido(*_a, **_k):
        raise AssertionError("detectar_tipo_captcha nao pode resolver")

    monkeypatch.setattr(solver, "solve_hcaptcha", proibido)
    monkeypatch.setattr(solver, "_gemini_grade", proibido)
    monkeypatch.setattr(solver, "_click_grade_tiles", proibido)
    solver.detectar_tipo_captcha(page)
    assert captcha.tiles_clicados == [] and captcha.submits == 0


def test_classificador_esta_na_api_publica():
    import resolvedor_captcha
    assert "detectar_tipo_captcha" in resolvedor_captcha.__all__
    for nome in ("TIPO_GRADE", "TIPO_GRADE_FUSED", "TIPO_CARTAO_ANIMAL",
                 "TIPO_IMAGEM", "TIPO_NENHUM", "TIPO_DESCONHECIDO"):
        assert nome in resolvedor_captcha.__all__


# ── Sonda de movimento: o sinal FISICO que identifica a bola ────────────────
#
# A bola e uma imagem unica e quadrada, entao o fallback geometrico a
# classificava `grade_fused` e a mandava para um resolvedor que olha UM quadro.
# O desvio por palavra-chave so a pega quando o texto e legivel no DOM. A sonda
# nao depende de ler nada: movimento e a propriedade que DEFINE este desafio.

def test_limiar_de_movimento_separa_ruido_de_bola():
    """Medido: ruido de compressao 0,011%; animacao mais fraca 0,28% (abelha).

    O limiar tem de ficar entre os dois, e com folga dos dois lados — 10x acima
    do ruido e ao menos 2x abaixo do sinal mais fraco ja observado.

    O sinal fraco NAO e a bola (0,90%-2,12%). E a abelha, medida em 08/09/2026
    em 0,28%-0,42%. Enquanto a constante valia 0,3%, a abelha era classificada
    "estatico" e ia para o resolvedor de quadro unico, que nao tem como
    responder qual flor ela nunca visita. Se alguem subir este limiar de volta
    para acomodar a bola, quebra a abelha de novo — por isso o teto aqui e
    derivado da abelha.
    """
    RUIDO_MEDIDO = 0.00011
    BOLA_MAIS_FRACA_MEDIDA = 0.0028
    assert RUIDO_MEDIDO < solver.BOLA_MOVIMENTO_MIN_FRACAO < BOLA_MAIS_FRACA_MEDIDA
    assert solver.BOLA_MOVIMENTO_MIN_FRACAO >= RUIDO_MEDIDO * 10
    assert solver.BOLA_MOVIMENTO_MIN_FRACAO <= BOLA_MAIS_FRACA_MEDIDA / 2


def test_limiar_e_FRACAO_e_nao_contagem_de_pixels():
    """Areas de 651x714 e 520x402 ja foram vistas; pixel absoluto viraria
    sensibilidade diferente para cada tamanho."""
    assert 0.0 < solver.BOLA_MOVIMENTO_MIN_FRACAO < 1.0


def test_sonda_nunca_derruba_a_classificacao():
    """Qualquer falha devolve False: na duvida, mantem o que ja existia."""
    class PaginaQuebrada:
        def locator(self, *_a, **_k):
            raise RuntimeError("sem página")
    assert solver._area_do_desafio_se_move(PaginaQuebrada()) is False


def test_sonda_sem_bounding_box_nao_afirma_movimento():
    class SemCaixa:
        def locator(self, *_a, **_k):
            return self
        @property
        def first(self):
            return self
        def bounding_box(self):
            return None
    assert solver._area_do_desafio_se_move(SemCaixa()) is False


# ── Generalizacao: mecanica em vez de vocabulario ───────────────────────────
#
# Em 08/09/2026 apareceu "Clique na flor em que a abelha nunca pousa" — MESMA
# mecanica da bola, substantivos outros. O resolvedor de sequencia citava
# "bola" e "animal" 14 vezes e nunca lia o enunciado: todo o conhecimento
# estava no prompt, escrito a mao. Cada variante nova custava um dev.
#
# O `_solve_imagem` ja provava o contrario do outro lado: repassa o enunciado
# LITERAL e pergunta a celula, sem saber o que e o desafio — e por isso resolveu
# "quebra o padrao" e "figura diferente" sem ninguem mapear nada.

def test_o_prompt_de_sequencia_nao_nomeia_o_desafio():
    """Se voltar a citar 'bola'/'animal', voltou a ser catalogo de formatos."""
    prompt = solver._PROMPT_BOLA.lower()
    # "animais" sobrevive so como nome historico do campo de retorno.
    corpo = prompt.split("=== retorne ===")[0]
    assert "bola" not in corpo, "o prompt voltou a nomear o objeto movel"
    assert "animal" not in corpo, "o prompt voltou a nomear os alvos"


def test_o_prompt_de_sequencia_recebe_o_enunciado():
    """E a instrucao da tela que define a condicao, nao o texto fixo."""
    assert "{instrucao}" in solver._PROMPT_BOLA


def test_gemini_bola_aceita_e_repassa_o_enunciado(monkeypatch):
    visto = {}

    def falso(contents, schema, api_key, tag, politica=None, rodizio=0,
              rodizio_segundo_provedor=None):
        visto["texto"] = contents[0]
        visto["limite_segundo"] = rodizio_segundo_provedor
        return {"ok": 1}

    monkeypatch.setattr(solver, "_gemini_call", falso)
    solver._gemini_bola([], 1, "k", instrucao="clique na flor em que a abelha nunca pousa")
    assert "abelha nunca pousa" in visto["texto"], (
        "o enunciado da tela nao chegou ao modelo")


def test_o_resolvedor_de_sequencia_le_a_tela():
    """Sem isto o enunciado nunca sai do DOM, por mais generico que o prompt seja."""
    import inspect
    fonte = inspect.getsource(solver._solve_bola)
    assert "_extrair_instrucao(page)" in fonte
    assert "instrucao=instrucao" in fonte


def test_a_sonda_de_movimento_nao_depende_de_proporcao():
    """A bola e quadrada; a abelha veio em ~1,48 e teria escapado.

    Movimento e a unica propriedade que separa "a resposta esta neste quadro"
    de "a resposta esta na sequencia" — e ela nao tem proporcao preferida.
    """
    import inspect
    fonte = inspect.getsource(solver._detect_challenge_type)
    pos_sonda = fonte.index("_area_do_desafio_se_move(page)")
    pos_faixa = fonte.index("0.75 <= ratio <= 1.4")
    assert pos_sonda < pos_faixa, (
        "a sonda voltou para dentro da faixa de proporcao e perde formatos largos")


# ── A sonda tem de cobrir a PAUSA, nao so um instante ───────────────────────
#
# Medido em producao em 08/09/2026: o desafio "clique na flor em que a abelha
# nunca pousa" foi classificado `bola_em_movimento` UMA vez e `grade_fused`
# DUAS, sendo o mesmo desafio. A diferenca era so em que instante os dois
# screenshots caíam — a abelha POUSA nas flores, e uma janela de 0,45 s cabe
# inteira dentro de uma pausa.
#
# E a mesma propriedade que obrigou `_amostrar_frames_distintos` a existir na
# captura, e que eu tinha deixado passar na deteccao.

def test_a_sonda_tira_mais_de_duas_amostras():
    assert solver.BOLA_SONDA_AMOSTRAS >= 5, (
        "com poucas amostras a sonda cabe dentro de uma pausa e ve 'estatico'")


def test_a_janela_da_sonda_cobre_uma_fatia_util_do_ciclo():
    """Ciclo da animacao medido: ~9,9 s. A janela precisa ser grande o bastante
    para atravessar uma pausa, e pequena o bastante para nao comer o orcamento."""
    janela = (solver.BOLA_SONDA_AMOSTRAS - 1) * solver.BOLA_SONDA_INTERVALO_S
    assert 2.0 <= janela <= 4.0, f"janela de {janela:.1f}s"


def test_a_sonda_compara_com_a_PRIMEIRA_amostra():
    """Comparar so com a anterior perde o elemento que sai e volta ao mesmo
    ponto — a diferenca entre quadros vizinhos daria zero."""
    import inspect
    fonte = inspect.getsource(solver._area_do_desafio_se_move)
    assert "primeira" in fonte
    assert "ImageChops.difference(primeira, atual)" in fonte


def test_a_sonda_sai_cedo_quando_detecta():
    """O caso animado nao pode pagar a janela inteira: ele e o caso comum."""
    import inspect
    fonte = inspect.getsource(solver._area_do_desafio_se_move)
    corpo = fonte[fonte.index("for i in range(1, BOLA_SONDA_AMOSTRAS)"):]
    assert "return True" in corpo, "sem saida cedo, toda deteccao custa a janela toda"


# ── A sonda nao pode custar caro por repeticao ──────────────────────────────
#
# Ampliar a janela de 2 para 7 amostras consertou a confiabilidade e criou um
# desperdicio: `_detect_challenge_type` roda a CADA iteracao de
# `solve_hcaptcha` — sao ate 6 — entao 7 amostras viravam 42 screenshots.
# Reportado como "trocentos prints".

def test_a_sonda_lembra_o_resultado_do_mesmo_desafio(monkeypatch):
    capturas = {"n": 0}

    class Loc:
        def bounding_box(self):
            return {"x": 0, "y": 0, "width": 10, "height": 10}

    class Pagina:
        def screenshot(self, **_kw):
            capturas["n"] += 1
            import io as _io

            from PIL import Image
            buf = _io.BytesIO()
            Image.new("RGB", (10, 10)).save(buf, "PNG")
            return buf.getvalue()

    solver._SONDA_MEMORIA.clear()
    monkeypatch.setattr(solver, "_get_challenge_element_locator", lambda _p: Loc())
    monkeypatch.setattr(solver, "_prompt_do_desafio", lambda _p: "mesmo enunciado")
    monkeypatch.setattr(solver.time, "sleep", lambda _s: None)

    p = Pagina()
    solver._area_do_desafio_se_move(p)
    apos_primeira = capturas["n"]
    for _ in range(5):
        solver._area_do_desafio_se_move(p)
    assert capturas["n"] == apos_primeira, (
        "a sonda repetiu a janela inteira — e isso vira dezenas de screenshots")


def test_enunciado_diferente_sonda_de_novo(monkeypatch):
    """Se o desafio mudou, a resposta anterior nao vale mais."""
    capturas = {"n": 0}

    class Loc:
        def bounding_box(self):
            return {"x": 0, "y": 0, "width": 10, "height": 10}

    class Pagina:
        def screenshot(self, **_kw):
            capturas["n"] += 1
            import io as _io

            from PIL import Image
            buf = _io.BytesIO()
            Image.new("RGB", (10, 10)).save(buf, "PNG")
            return buf.getvalue()

    solver._SONDA_MEMORIA.clear()
    textos = iter(["primeiro desafio", "segundo desafio"])
    monkeypatch.setattr(solver, "_get_challenge_element_locator", lambda _p: Loc())
    monkeypatch.setattr(solver, "_prompt_do_desafio",
                        lambda _p: next(textos, "segundo desafio"))
    monkeypatch.setattr(solver.time, "sleep", lambda _s: None)

    p = Pagina()
    solver._area_do_desafio_se_move(p)
    n1 = capturas["n"]
    solver._area_do_desafio_se_move(p)
    assert capturas["n"] > n1, "enunciado novo tinha de disparar sonda nova"


# ── O ENUNCIADO decide a mecanica; a proporcao so desempata ────────────────
#
# Medido em 08/09/2026: o MESMO desafio "Por favor, clique na figura diferente"
# foi roteado de duas formas conforme o tamanho em que a Receita o renderizou.
#
#     605x410  ratio 1,48  ->  imagem       ->  grade 20x20  ->  RESOLVEU
#     651x714  ratio 0,91  ->  grade_fused  ->  9 tiles      ->  falhou
#
# `grade_fused` pergunta QUAIS tiles marcar — mecanica de "selecione todas as
# imagens com onibus". Num desafio com formas espalhadas nao ha tiles.

@pytest.mark.parametrize("instrucao", [
    "Clique no animal que a bola nunca toca",
    "Clique na flor em que a abelha nunca pousa",
    "Por favor, clique no ícone que quebra o padrão",
    "Por favor, clique na figura diferente",
])
def test_enunciados_reais_sao_de_clique_unico(instrucao):
    """Os quatro que apareceram em producao ate hoje."""
    assert solver._pede_um_clique_so(instrucao) is True


@pytest.mark.parametrize("instrucao", [
    "Selecione todas as imagens com ônibus",
    "Clique em todas as figuras que contenham um gato",
    "Marque cada imagem com semáforo",
    "Select all images with a bus",
])
def test_selecao_multipla_nao_e_clique_unico(instrucao):
    """A grade 3x3 de verdade continua indo para o resolvedor de tiles."""
    assert solver._pede_um_clique_so(instrucao) is False


def test_a_marca_de_varios_vence_a_de_um():
    """'clique em todas' tem 'clique', e ainda assim e plural."""
    assert solver._pede_um_clique_so("clique em todas as figuras diferentes") is False


def test_sem_enunciado_nao_afirma_clique_unico():
    """Sem texto, a decisao volta para a geometria — nao se inventa mecanica."""
    assert solver._pede_um_clique_so("") is False
    assert solver._pede_um_clique_so(None) is False


def test_o_resolvedor_de_imagem_nao_TEM_como_clicar_varias_vezes():
    """A guarda virou impossibilidade, e isso e melhor que a guarda.

    Em 08/09/2026 o modelo devolvia 4 pontos para "clique na figura diferente"
    e a automacao clicava os quatro — erro por construcao, ja que o enunciado
    EXCLUI tres deles. A correcao de entao foi recusar a lista e retentar.

    Em 09/09/2026 a malha 20x20 saiu do caminho: medido contra as amostras
    arquivadas, com as respostas marcadas na imagem, ela caia na agua vazia
    entre duas figuras enquanto o pixel direto caia em cima da certa — e em um
    terco do tempo. Com `ESQUEMA_PIXEL`, a resposta e UM ponto por construcao:
    a lista que a guarda recusava nao existe mais para ser recusada.

    Este teste afirma a propriedade nova. Se alguem devolver a malha para este
    resolvedor, a lista volta a ser possivel e ele cai.
    """
    assert "x" in solver.ESQUEMA_PIXEL["properties"]
    assert "y" in solver.ESQUEMA_PIXEL["properties"]
    assert "click_positions" not in solver.ESQUEMA_PIXEL["properties"], (
        "voltou a aceitar lista de pontos")

    import inspect
    fonte = inspect.getsource(solver._solve_imagem)
    assert "_gemini_pixel" in fonte
    assert "_overlay_grid" not in fonte, (
        "a malha voltou para o formato em que ela foi medida perdendo")


def test_a_decisao_por_enunciado_vem_antes_da_proporcao():
    import inspect
    fonte = inspect.getsource(solver._detect_challenge_type)
    pos_enunciado = fonte.index("_pede_um_clique_so(instrucao_lower)")
    pos_ratio = fonte.index("0.75 <= ratio <= 1.4")
    assert pos_enunciado < pos_ratio, (
        "a proporcao voltou a decidir o que o enunciado ja respondia")



# ── O caminho ANIMADO alcanca o segundo provedor ───────────────────────────
#
# `_solve_bola` tem 2 rodadas e o limite geral e 2, entao o segundo provedor
# nunca era alcancado ali. Foi decisao deliberada — ele mediu 0/3 contra as
# amostras da bola — e o dado de producao a derrubou: em 08/09/2026 NENHUMA
# animacao foi concluida. 0/3 dele contra 0 de N do Gemini: tentar custa uma
# chamada e nao pode ser pior que a falha certa.





def test_o_segundo_provedor_pergunta_PRIMEIRO():
    """Era o segundo modelo; em 09/09/2026 virou o primeiro.

    O placar do dia decidiu, e nao a preferencia de ninguem:

        Gemini   30 falhas  (14 + 10 + 6 nos tres modelos), o dia inteiro
                            504 DEADLINE_EXCEEDED e 503 "high demand"
        astra     2 chamadas com prazo adequado -> 2 respostas (6,9s e 17,0s)

    Manter como reserva quem responde, e como principal quem nao responde,
    gastava o orcamento inteiro para redescobrir isso a cada desafio — e o
    substituto so era chamado com o troco, quando ainda era chamado.

    O Gemini continua na cadeia logo atras: inverteu-se a ordem, nao se removeu
    ninguem. E a decisao de custo que existia aqui — nao pagar o provedor pago
    quando o gratuito resolve — foi dispensada explicitamente pelo Jean:
    "esquece custo, isso precisa estar assertivo".
    """
    assert solver.RODIZIO_DO_SEGUNDO_PROVEDOR == 0


def test_todo_resolvedor_alcanca_o_segundo_provedor():
    """Inclusive o de 2 rodadas — antes ele nunca chegava la."""
    import inspect
    for nome in ("_solve_bola", "_solve_imagem", "_solve_grade",
                 "_solve_grade_fused"):
        par = inspect.signature(getattr(solver, nome)).parameters["max_rounds"]
        assert par.default > solver.RODIZIO_DO_SEGUNDO_PROVEDOR, nome


# ── O guardiao de frescor tolera FUNDO ANIMADO ─────────────────────────────
#
# O hash byte a byte era estrito demais: ESTES desafios tem fundo animado.
# Medido em 08/09/2026 nos quadros reais, com o desafio parado, 0,27% a 0,42%
# dos pixels mudam sozinhos entre duas capturas. Um hash exato nunca bate — o
# modelo respondia certo e a resposta era descartada como "desafio mudou".
#
# Registrado em producao, tres respostas boas jogadas fora em sequencia, com
# confianca high/medium/high.

def _png(cor, tam=(100, 100)):
    import io as _io

    from PIL import Image
    buf = _io.BytesIO()
    Image.new("RGB", tam, cor).save(buf, "PNG")
    return buf.getvalue()


def _png_com_ruido(fracao, tam=(100, 100)):
    """Imagem base com `fracao` dos pixels alterados."""
    import io as _io

    from PIL import Image
    im = Image.new("RGB", tam, (10, 10, 10))
    total = tam[0] * tam[1]
    alvo = int(total * fracao)
    px = im.load()
    for i in range(alvo):
        px[i % tam[0], i // tam[0]] = (250, 250, 250)
    buf = _io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def test_o_limiar_separa_fundo_animado_de_troca_de_desafio():
    """Fundo da decimos de porcento; troca real muda a cena inteira."""
    FUNDO_ANIMADO_MEDIDO = 0.0042      # pior caso medido
    TROCA_REAL_MEDIDA = 0.09           # 42.924 px em 651x714
    assert FUNDO_ANIMADO_MEDIDO < solver.DESAFIO_MUDOU_MIN_FRACAO < TROCA_REAL_MEDIDA


def test_diferenca_pequena_conta_como_MESMO_desafio(monkeypatch):
    base = _png_com_ruido(0.0)
    quase = _png_com_ruido(0.004)      # 0,4% — fundo animado
    monkeypatch.setattr(solver, "_challenge_visible", lambda _p: True)
    monkeypatch.setattr(solver, "_capturar_desafio", lambda _p: (quase, None))
    monkeypatch.setattr(solver, "_fingerprint_desafio", lambda _p, _b: "outro")
    assert solver._desafio_ainda_e_o_mesmo(object(), "origem", base) is True


def test_diferenca_grande_conta_como_OUTRO_desafio(monkeypatch):
    base = _png_com_ruido(0.0)
    outro = _png_com_ruido(0.40)
    monkeypatch.setattr(solver, "_challenge_visible", lambda _p: True)
    monkeypatch.setattr(solver, "_capturar_desafio", lambda _p: (outro, None))
    monkeypatch.setattr(solver, "_fingerprint_desafio", lambda _p, _b: "outro")
    assert solver._desafio_ainda_e_o_mesmo(object(), "origem", base) is False


def test_sem_imagem_de_origem_continua_estrito(monkeypatch):
    """Sem com o que comparar, a duvida resolve contra clicar."""
    monkeypatch.setattr(solver, "_challenge_visible", lambda _p: True)
    monkeypatch.setattr(solver, "_capturar_desafio", lambda _p: (_png("red"), None))
    monkeypatch.setattr(solver, "_fingerprint_desafio", lambda _p, _b: "outro")
    assert solver._desafio_ainda_e_o_mesmo(object(), "origem", None) is False


def test_hash_igual_continua_sendo_caminho_rapido(monkeypatch):
    monkeypatch.setattr(solver, "_challenge_visible", lambda _p: True)
    monkeypatch.setattr(solver, "_capturar_desafio", lambda _p: (b"x", None))
    monkeypatch.setattr(solver, "_fingerprint_desafio", lambda _p, _b: "igual")
    assert solver._desafio_ainda_e_o_mesmo(object(), "igual", None) is True


def test_desafio_ausente_nunca_e_o_mesmo(monkeypatch):
    monkeypatch.setattr(solver, "_challenge_visible", lambda _p: False)
    assert solver._desafio_ainda_e_o_mesmo(object(), "origem", b"x") is False


def test_todos_os_resolvedores_passam_a_imagem_de_origem():
    """Sem ela a checagem cai no modo estrito e o defeito volta."""
    import inspect
    fonte = inspect.getsource(solver)
    assert "_desafio_ainda_e_o_mesmo(page, fingerprint)" not in fonte, (
        "algum resolvedor voltou a chamar sem a imagem de origem")
