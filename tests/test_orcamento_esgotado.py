"""Falta de tempo não é falha do provedor, e não rende outra rodada."""
import inspect

from resolvedor_captcha import solver


def test_a_recusa_por_falta_de_tempo_tem_TIPO_proprio():
    """As duas saíam idênticas no log, e pedem ações opostas.

    `_diagnostico_erro` nunca imprime o texto do erro — regra de privacidade,
    e ela está certa. O efeito colateral é que a frase "segundo provedor não
    chamado: restam 0.0s" ficava invisível, e a recusa aparecia como

        segundo provedor também falhou | categoria=desconhecido | tipo=RuntimeError

    indistinguível de provedor quebrado. Provedor quebrado se investiga na
    chave e na API; sem orçamento se resolve dando mais tempo ou parando antes.
    """
    assert issubclass(solver.SegundoProvedorSemOrcamento, RuntimeError)
    fonte = inspect.getsource(solver._astra_call)
    assert "raise SegundoProvedorSemOrcamento(" in fonte
    assert "raise RuntimeError(\n" not in fonte.split("ASTRA_DEADLINE_MIN_S")[1][:300]


def test_o_log_da_recusa_e_montado_dos_NUMEROS_e_nao_da_excecao():
    """O gate de higiene de logs proíbe interpolar a exceção capturada, e com
    razão: texto de exceção pode carregar conteúdo do provedor.

    Minha primeira versão fez `print(f"... {e}")` e o gate reprovou. Abrir
    exceção para "esta aqui é nossa" é como a regra morre — a mensagem passou
    a ser montada dos números que já temos.
    """
    fonte = inspect.getsource(solver)
    i = fonte.index("except SegundoProvedorSemOrcamento")
    trecho = fonte[i:i + 900]
    assert "segundo provedor NAO chamado" in trecho
    assert "Nao e falha dele" in trecho
    assert "{e}" not in trecho.split("except Exception")[0]


def test_orcamento_esgotado_encerra_as_rodadas():
    """Rodada com 0s não pode dar certo, e cada uma CONTA como tentativa
    frustrada — inflando a contagem que decide "não dá para automatizar".

    Medido na RUN-ef3f4b9d: da rodada 3 em diante toda chamada já tinha 0s.
    Rodadas 3, 4 e 5 foram encenação — screenshot, "0s", erro, repete —, e o
    desfecho delas fez o fluxo concluir que o captcha era intratável. O
    enunciado era "clique em todos os objetos feitos principalmente de metal",
    com dois baldes óbvios na grade.
    """
    # A guarda do `grade_fused` passou a aceitar PONTO além de tiles, então o
    # texto que a delimita mudou. O que o teste garante continua o mesmo: entre
    # "não tenho resposta" e "vou continuar" existe uma saída por orçamento.
    for fn, marca in ((solver._solve_grade, "if not valid_tiles:"),
                      (solver._solve_grade_fused,
                       'if not (valid_tiles or (result or {}).get("ponto")):')):
        fonte = inspect.getsource(fn)
        i = fonte.index(marca)
        j = fonte.index("Continuando...", i)
        trecho = fonte[i:j]
        assert "politica.esgotado" in trecho, fn.__name__
        assert "break" in trecho, fn.__name__
