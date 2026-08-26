"""Resolvedor de hCaptcha.

Alem de resolver, o pacote expoe INSPECAO: `captcha_presente` diz se ha desafio,
`detectar_tipo_captcha` diz de qual tipo ele e, e `abrir_desafio` abre o desafio
a partir do widget "Sou humano" SEM resolve-lo — abrir nao e resolver. Quem
integra usa isso para
decidir POLITICA POR TIPO — nem todo fluxo autoriza resolucao automatica de
todos os formatos.
"""
from .solver import (
    TIPO_CARTAO_ANIMAL,
    TIPO_DESCONHECIDO,
    TIPO_GRADE,
    TIPO_GRADE_FUSED,
    TIPO_IMAGEM,
    TIPO_NENHUM,
    TIPOS_CONHECIDOS,
    abrir_desafio,
    captcha_presente,
    cell_to_viewport,
    detectar_tipo_captcha,
    modelos_ativos,
    preparar_modelos,
    solve_captcha,
    solve_hcaptcha,
)

__all__ = [
    "TIPOS_CONHECIDOS",
    "preparar_modelos",
    "modelos_ativos",
    "TIPO_CARTAO_ANIMAL",
    "TIPO_DESCONHECIDO",
    "TIPO_GRADE",
    "TIPO_GRADE_FUSED",
    "TIPO_IMAGEM",
    "TIPO_NENHUM",
    "abrir_desafio",
    "captcha_presente",
    "cell_to_viewport",
    "detectar_tipo_captcha",
    "solve_captcha",
    "solve_hcaptcha",
]
