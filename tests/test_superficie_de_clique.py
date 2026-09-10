"""O que a página observa é a sequência de eventos, não a nossa intenção."""
import inspect
import re

from resolvedor_captcha import solver


def test_o_submit_tenta_clique_REAL_antes_do_javascript():
    """A estratégia de clique real já existia — só estava atrás na fila.

    O JavaScript era a #1 e sempre vencia, então toda submissão de produção
    saía como `btn.click()` disparado por `evaluate`. O log dizia isso em cada
    desafio resolvido hoje:

        [captcha] Submit via JS/frame real.

    Clique sintético de JS chega sem `isTrusted`, sem mousedown/mouseup, sem
    coordenada e sem o mousemove que o antecede — e são esses eventos que o
    hCaptcha amostra.

    Inverter não adiciona risco: o JS continua logo abaixo como rede, para o
    botão que o locator não consegue clicar.
    """
    fonte = inspect.getsource(solver._submit_captcha)
    i = fonte.index("Submit via clique real")
    j = fonte.index("Submit via JS/frame real (fallback)")
    assert i < j, "o clique real tem de ser tentado primeiro"


def test_o_ponteiro_chega_por_trajetoria():
    """`page.mouse.click(x, y)` emite um único mousemove já no destino.

    O ponteiro nunca esteve em outro lugar — nenhum humano produz isso.

    `_mover_cursor_suave`, que já existia, mexe no cursor do SISTEMA via
    user32; a página não vê o cursor do sistema, vê os eventos injetados. São
    coisas diferentes, e só a trajetória injetada chega até ela.
    """
    fonte = inspect.getsource(solver._aproximar_do_alvo)
    assert fonte.count("page.mouse.move") == 2, (
        "duas etapas: ponto deslocado e aproximação — reta perfeita também é "
        "assinatura")
    assert "steps=" in fonte

    pixel = inspect.getsource(solver._click_pixel)
    assert "_aproximar_do_alvo(page, x, y)" in pixel
    i = pixel.index("_aproximar_do_alvo")
    j = pixel.index("page.mouse.click")
    assert i < j, "a trajetória vem antes do clique"


def test_nenhuma_constante_na_cadencia_dos_tiles():
    """Ordem crescente, 30ms fixos e 50ms de pausa eram três constantes numa
    sequência que ninguém produz à mão. A ordem em que alguém marca os
    quadrados não é a ordem do DOM."""
    fonte = inspect.getsource(solver._click_grade_tiles)
    assert "random.shuffle" in fonte
    assert "delay=30" not in fonte
    assert re.search(r"time\.sleep\(random\.uniform", fonte)


def test_random_esta_importado_no_modulo():
    """Guarda contra um erro que py_compile não pega: `random` era usado só
    dentro de `_mover_cursor_suave`, com import local. Sem o import de módulo,
    cada clique levantaria NameError em runtime — no primeiro captcha."""
    assert solver.random.__name__ == "random"
