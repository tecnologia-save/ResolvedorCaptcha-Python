"""Nenhuma função do solver usa um nome que não existe.

RUN-06919de1 (15/09/2026), ALEX ROCHA, captcha da representação: o prazo
acabou entre a 1ª e a 2ª rodada, a guarda "sem tempo para outra rodada" disparou
e o print dela levantou NameError:

    print(f"    [captcha/{grade_fused}] Rodada {rnd}: restam "

`{grade_fused}` era para ser texto, não variável. O login traduz qualquer
exceção do resolvedor em FalhaDoResolvedorCaptcha, e a empresa caiu com "o
resolvedor de captcha falhou tecnicamente (NameError)". O mesmo erro estava em
`_solve_grade` e `_solve_imagem`: as três guardas vieram do mesmo script.

Os testes da guarda só conferiam o TEXTO dela no código-fonte, e a linha só
roda quando o orçamento acaba no meio do captcha. Esta varredura olha a tabela de
símbolos de cada função e pega qualquer nome global lido sem existir no módulo
nem nos builtins, inclusive em linhas que nenhum teste executa.
"""
import builtins
import inspect
import symtable

from resolvedor_captcha import solver

# Definidos pelo próprio Python em todo módulo.
DO_INTERPRETADOR = {"__file__", "__name__", "__doc__", "__spec__", "__package__"}


def _nomes_indefinidos(modulo) -> list[str]:
    fonte = inspect.getsource(modulo)
    raiz = symtable.symtable(fonte, modulo.__file__, "exec")
    achados: list[str] = []

    def varre(tabela, onde):
        for simbolo in tabela.get_symbols():
            nome = simbolo.get_name()
            if (simbolo.is_global() and simbolo.is_referenced()
                    and not simbolo.is_assigned()
                    and not hasattr(modulo, nome)
                    and not hasattr(builtins, nome)
                    and nome not in DO_INTERPRETADOR):
                achados.append(f"{onde}: {nome}")
        for filho in tabela.get_children():
            varre(filho, f"{onde}.{filho.get_name()}")

    varre(raiz, modulo.__name__)
    return achados


def test_o_solver_nao_le_nome_que_nao_existe():
    assert _nomes_indefinidos(solver) == []


def test_a_varredura_pega_o_erro_da_run():
    """Sem isto, uma varredura quebrada passaria verde para sempre."""
    import types

    falso = types.ModuleType("falso")
    falso.__file__ = "falso.py"
    codigo = (
        "def _solve_grade_fused(rnd):\n"
        "    print(f'    [captcha/{grade_fused}] Rodada {rnd}: restam ')\n"
    )
    exec(compile(codigo, "falso.py", "exec"), falso.__dict__)
    original = inspect.getsource
    try:
        inspect.getsource = lambda m: codigo if m is falso else original(m)
        assert _nomes_indefinidos(falso) == ["falso._solve_grade_fused: grade_fused"]
    finally:
        inspect.getsource = original


def test_as_guardas_de_rodada_escrevem_o_tipo_como_texto():
    for funcao, tag in (("_solve_grade", "grade"),
                        ("_solve_grade_fused", "grade_fused"),
                        ("_solve_imagem", "imagem")):
        fonte = inspect.getsource(getattr(solver, funcao))
        assert f"[captcha/{tag}] Rodada {{rnd}}: restam" in fonte, funcao
        assert f"[captcha/{{{tag}}}]" not in fonte, funcao
