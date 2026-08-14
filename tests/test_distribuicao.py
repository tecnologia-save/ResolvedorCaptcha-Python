"""Metadados da distribuicao — o que chega ao runtime que instala este pacote.

Dois defeitos reais motivam este arquivo, ambos silenciosos:

1. **Dependencia perdida.** Com um `[project]` presente, o setuptools ignora
   `install_requires` do setup.py. Ao introduzir o pyproject, o pacote passou a
   ser publicado SEM `Requires-Dist` — uma instalacao limpa ficaria sem
   google-genai e sem Pillow, e o solver cairia em `_GENAI=False`/`_PIL=False`
   sem erro visivel.

2. **Versao estatica.** O AutoHub Edge roda `pip install -r requirements.txt`
   sem `--upgrade`. Se o codigo muda e a versao nao, o pip considera o pacote
   ja satisfeito e mantem o antigo — provado em ambiente isolado. Mudanca de
   codigo desta biblioteca EXIGE bump de versao, ou a correcao nao chega ao
   runtime.
"""
import pathlib
import re
import tomllib

RAIZ = pathlib.Path(__file__).resolve().parent.parent
PYPROJECT = tomllib.loads((RAIZ / "pyproject.toml").read_text(encoding="utf-8"))


def _requisitos_do_txt():
    linhas = (RAIZ / "requirements.txt").read_text(encoding="utf-8").splitlines()
    return sorted(
        re.split(r"\s+#", linha.strip())[0].strip()
        for linha in linhas
        if linha.strip() and not linha.strip().startswith("#")
    )


def _nome(requisito):
    return re.split(r"[<>=!~\[ ]", requisito, maxsplit=1)[0].strip().lower()


def test_distribuicao_declara_dependencias_de_runtime():
    """Sem isto o pacote instala 'limpo' e o solver perde genai/Pillow."""
    deps = PYPROJECT["project"]["dependencies"]
    assert {_nome(d) for d in deps} == {"google-genai", "pillow"}


def test_dependencias_batem_com_requirements_txt():
    """Duas listas para a mesma coisa so servem se nao divergirem."""
    do_projeto = sorted(PYPROJECT["project"]["dependencies"])
    assert do_projeto == _requisitos_do_txt()


def test_setup_py_nao_repete_metadados():
    """Duplicata e o mecanismo pelo qual o `Requires-Dist` sumiu."""
    fonte = (RAIZ / "setup.py").read_text(encoding="utf-8")
    for campo in ("version=", "name=", "install_requires", "packages="):
        assert campo not in fonte


def test_build_backend_declarado():
    assert PYPROJECT["build-system"]["build-backend"] == "setuptools.build_meta"


def test_versao_e_semver_de_tres_partes():
    assert re.fullmatch(r"\d+\.\d+\.\d+", PYPROJECT["project"]["version"])
