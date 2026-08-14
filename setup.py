"""Compatibilidade com instaladores legados.

Os metadados vivem TODOS no `pyproject.toml` — nome, versao, dependencias e
pacotes. Este arquivo nao os repete de proposito: com um `[project]` presente,
o setuptools ignora o que for declarado aqui, e a duplicata so serviria para
divergir em silencio (foi assim que a distribuicao ficou sem `Requires-Dist`).
"""
from setuptools import setup

setup()
