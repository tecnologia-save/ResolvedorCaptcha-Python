"""O portal é o professor: ele já dizia se a resposta estava certa."""
import inspect

from resolvedor_captcha import solver


def test_o_veredito_tem_TRES_estados_e_nao_dois():
    """Forçar binário aqui rotularia o acerto como erro.

    O hCaptcha dá DUAS rodadas por desafio. Uma rodada 1 respondida CERTO faz
    aparecer a rodada 2 — e o desafio continua visível, então
    `_wait_for_resolve` devolve False. Um rótulo binário marcaria como errada
    exatamente a resposta que funcionou, e o acervo ensinaria o contrário do
    que aconteceu.

    `avancou` fica separado de `aceito` porque não é a mesma evidência: um é
    certeza (o desafio sumiu), o outro é inferência (mudou de desafio). Quem
    for usar o acervo escolhe o quanto quer ser exigente.
    """
    assert {solver.VEREDITO_ACEITO,
            solver.VEREDITO_AVANCOU,
            solver.VEREDITO_RECUSADO} == {"aceito", "avancou", "recusado"}

    fonte = inspect.getsource(solver._veredito_do_portal)
    assert "_desafio_ainda_e_o_mesmo" in fonte, (
        "sem comparar o desafio não há como separar 'avançou' de 'recusado'")


def test_todos_os_resolvedores_de_producao_registram():
    """Grade, grade fundida e imagem — os três que aparecem em produção.

    Sem registro num deles, o acervo fica cego justamente na família que mais
    ocorre, e a amostragem enviesa o que vier a ser aprendido.
    """
    for fn in (solver._solve_grade, solver._solve_grade_fused, solver._solve_imagem):
        assert "_registrar_licao" in inspect.getsource(fn), fn.__name__
        assert "_veredito_do_portal" in inspect.getsource(fn), fn.__name__


def test_o_acervo_e_opcional_e_nunca_derruba_a_run():
    """Sem a variável de ambiente, nada é escrito e o comportamento é o de
    sempre — é o mesmo contrato do despejo de erro, e é o que permite isto
    existir num pacote de produção."""
    fonte = inspect.getsource(solver._registrar_licao)
    assert 'os.environ.get(LICOES_DIR_ENV, "")' in fonte
    assert "if not destino" in fonte
    assert "except Exception" in fonte


def test_grava_o_que_o_modelo_VIU():
    """No resolvedor de imagem existem duas imagens: a que foi ao modelo e a de
    identidade, usada só para comparar frescor. Gravar a errada daria um acervo
    de exemplos que ninguém analisou."""
    fonte = inspect.getsource(solver._solve_imagem)
    i = fonte.index("_registrar_licao(")
    assert "png_raw" in fonte[i:i + 120]
