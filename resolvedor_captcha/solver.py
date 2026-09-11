"""Resolvedor de hCaptcha.

Combina o melhor de duas implementações:
  - Detecção robusta do frame ATIVO (múltiplos iframes pré-carregados pelo hCaptcha)
  - Grade 3x3: screenshot do iframe ativo → Gemini (response_schema) → índices 0-8
               → clique direto em .task[n] no DOM do frame
  - Imagem completa: 3 estratégias de screenshot + grid 20x20 (PIL) → Gemini → col/row → pixels
  - Imagem de referência extraída separadamente para prompt mais específico
  - Submit com 5 estratégias em cascata (JS, frame.locator, frame_locator, coords, XPath)
  - google.genai SDK com response_schema → JSON sempre estruturado e válido
  - Thinking LIGADO por padrão (THINKING_BUDGET=4096) — acurácia é o que resolve
    o captcha rápido (evita o loop de retentativas por "confiança baixa");
    ajustável via env CAPTCHA_THINKING_BUDGET
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import random
import time
from itertools import pairwise
from typing import NamedTuple, Optional

try:
    from PIL import Image, ImageChops, ImageDraw, ImageFont
    _PIL = True
except ImportError:
    _PIL = False

try:
    from google import genai as _genai_lib
    from google.genai import types as _gt
    _GENAI = True
except ImportError:
    _GENAI = False

# ──────────────────────────────────────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────────────────────────────────────

# Modelos tentados em ordem: se o principal estiver sobrecarregado (503/UNAVAILABLE)
# ou indisponível (404), a chamada cai para o próximo. Modelos diferentes têm pools
# de capacidade separados no Google, então o fallback resolve picos momentâneos.
#
# A ordem é definida por MEDIÇÃO na forma de chamada real do solver (imagem de
# grade 3x3 + _PROMPT_GRADE + response_schema + thinking 4096 + timeout 30s).
# Remedição em 17/08/2026, 16 chamadas por modelo (4 desafios sintéticos de
# gabarito conhecido x 4 repetições) — chamadas concluídas e latência média:
#   3.5-flash        16/16  2,7s (pior 3,5s)   3-flash-preview  16/16  3,1s (4,1s)
#   3.5-flash-lite   16/16  2,2s (pior 2,7s)   flash-lite-latest 16/16 2,2s (3,3s)
#   3.1-flash-lite   16/16  4,7s (pior 25,4s)  3.1-pro-preview   4/16  (12 timeouts)
#   pro-latest        3/16  (11 timeouts)      flash-latest       3/16  (9 timeouts, 4x 503)
#   3.6-flash         3/8                      3.7-flash          2/8
# Acurácia: TODAS as chamadas concluídas acertaram o gabarito, em qualquer
# modelo. Ou seja, o que separa os modelos nesta tarefa NÃO é acerto — é
# CONCLUIR a chamada. Por isso a lista é ordenada por disponibilidade medida.
#
# Saíram da lista por instabilidade comprovada, não por pico isolado:
#   - pro-latest e 3.1-pro-preview: o pool "pro" estoura o teto de 30s na maioria
#     das chamadas. O pro era a "carta de acurácia" quando os flash erram, mas
#     esse papel nunca existiu de fato — `_gemini_call` só troca de modelo em
#     erro de DISPONIBILIDADE (503/404/timeout), nunca por resposta errada. Um
#     modelo que só é alcançado quando os anteriores caem precisa ser o mais
#     disponível, e o pro é justamente o menos.
#   - flash-latest, 3.6-flash, 3.7-flash: os flash "de topo" seguem com o pool
#     saturado — 503 e timeout são a regra, não o pico. A run de 18/08/2026
#     reconfirmou em produção: flash-latest com 503/ReadTimeout, 3.6-flash com
#     504, pro-latest com 429. Voltar qualquer um deles ao caminho quente exige
#     medição nova, não impressão de uma execução.
#
# NÃO usar a família 2.x: gemini-2.0-flash e gemini-2.5-flash respondem 404
# ("no longer available") para chaves novas — reconfirmado nesta medição, 0/6.
#
# O caminho quente é só de IDs ESTÁVEIS, e isso custou duas cartas boas:
#   - "-latest" (flash-lite-latest, 16/16) sai porque o alias troca de versão
#     por trás; o que roda em produção deixa de ser o que foi medido, e a
#     medição é o único critério que esta lista tem;
#   - "-preview" (3-flash-preview, 16/16) sai porque pode ser aposentado sem
#     aviso. Se um ID fixo morrer, `_is_overloaded_error` trata o 404 como
#     motivo de troca e a chamada cai para o próximo — o custo é uma tentativa.
# Sobra uma cadeia de três, toda ela 16/16 na medição de 17/08/2026.
# Sobrescrevível por ambiente: GEMINI_MODELS="modelo1,modelo2,...".
# A lista PADRAO fica separada da resolvida, e nao e detalhe: as guardas desta
# lista (nada de `-latest`, nada de `-preview`, nada de reprovado por medicao)
# viravam letra morta quando alguem sobrescrevia por ambiente — os testes liam a
# lista ja resolvida e passavam a validar a escolha do ambiente, nao a nossa.
# Aconteceu em 08/09/2026: com GEMINI_MODELS apontando para um modelo REPROVADO,
# tres testes cairam e so entao o problema apareceu.
GEMINI_MODELS_PADRAO   = [
    "gemini-3.5-flash-lite",   # primário: 16/16, 2,2s — o mais rápido e previsível
    "gemini-3.5-flash",        # fallback: 16/16, 2,7s — flash COMPLETO, pool distinto
    "gemini-3.1-flash-lite",   # último recurso: 16/16, mas com cauda de 25,4s
]

# Sobrescrevivel por ambiente: GEMINI_MODELS="modelo1,modelo2,...". Util para
# medir um candidato em producao sem mexer em codigo — mas note que a
# sobrescrita NAO passa pelas guardas acima. E ferramenta de medicao, nao de
# configuracao permanente.
GEMINI_MODELS          = [m.strip() for m in os.environ.get("GEMINI_MODELS", "").split(",") if m.strip()] or list(GEMINI_MODELS_PADRAO)
GEMINI_MODEL           = GEMINI_MODELS[0]

# "Thinking" (raciocínio interno do Gemini antes de responder). Para captcha,
# ACURÁCIA É VELOCIDADE: com thinking o modelo acerta os tiles em 1-2 tentativas;
# SEM thinking ele responde "confiança baixa" e o solver entra em loop de
# retentativas que nunca resolve — ou seja, fica MAIS lento e ainda falha.
# Por isso o padrão é um orçamento POSITIVO (4096, valor comprovado).
#   >0 (padrão) = envia esse orçamento de thinking (acurado).
#   0           = NÃO envia thinking_config → modelo usa o default (rápido, mas
#                 impreciso em captcha). Nunca enviamos "0" explícito: estes
#                 modelos respondem 400 INVALID_ARGUMENT ao receber budget=0.
# Ajustável sem recompilar via CAPTCHA_THINKING_BUDGET no ambiente (ex.: 2048
# para tentar acelerar um pouco, à custa de possível queda de acurácia).
try:
    THINKING_BUDGET = max(0, int(os.getenv("CAPTCHA_THINKING_BUDGET", "4096") or "4096"))
except (ValueError, TypeError):
    THINKING_BUDGET = 4096

GRID_COLS              = 20
GRID_ROWS              = 20
# Tentativas nos loops de alto nivel (screenshot/semantica). Igual ao numero
# de modelos, e nao por acaso: a chamada usa temperature=0.0, entao a mesma
# imagem no mesmo modelo devolve SEMPRE a mesma resposta. Com o rodizio,
# tres tentativas ja ouviram os tres modelos — a quarta e garantida a
# repetir uma das anteriores. O que ajuda dali em diante e um screenshot
# NOVO, e disso cuida a rodada seguinte.
MAX_GEMINI_TRIES       = 3
GEMINI_TRIES_PER_MODEL = 2    # tentativas por modelo dentro de _gemini_call (troca rápido)

# Quantos MODELOS do Gemini tentar antes de ir ao segundo provedor.
#
# Um. Insistir no provedor que acabou de falhar e a ordem errada, e o custo nao
# e teorico. Medido em 11/09/2026, LEONARDO VIEIRA RESTAURANTE:
#
#     'gemini-3.5-flash'      descansa 1min (ReadTimeout)
#     'gemini-3.1-flash-lite' descansa 1min (ReadTimeout)
#     segundo provedor NAO chamado: restam 9.2s e o minimo viavel e 10s
#
# O astra foi recusado por OITO DECIMOS DE SEGUNDO. Os dois modelos do Gemini
# consumiram o orcamento disputando entre si, e quem podia responder nao foi
# perguntado. A empresa terminou como "exige validacao manual".
#
# A rotacao entre modelos do Gemini continua existindo — ela so deixa de vir
# ANTES da alternativa de verdade. Quando nao ha segundo provedor configurado,
# nada muda: ai a rotacao e tudo o que existe, e encurtar so tiraria tentativa
# sem dar nada em troca.
MODELOS_GEMINI_ANTES_DO_SEGUNDO = 1

# Teto por TENTATIVA de chamada ao modelo, em milissegundos.
#
# 20s, medido em 26/08/2026 contra a API real. Era 30s, e cada falha custava os
# 30 inteiros — num log de producao isso se repetiu dezenas de vezes numa so
# execucao. Mas 12s, que foi a primeira tentativa de corte, era CEDO DEMAIS:
# uma chamada boa levou 19,8s na mesma medicao.
#
# REMEDIDO em 11/09/2026, sobre 9 runs e 38 chamadas que responderam:
#
#     p50 = 9,6s   p90 = 22,4s   p95 = 25,3s   max = 30,1s
#
# A pior resposta boa subiu de 19,8s para 30,1s — o numero acima envelheceu, e
# decisoes estavam sendo tomadas com ele. O que cada teto custaria em respostas
# boas perdidas: 12s corta 37%, 15s corta 18%, 20s corta 11%, 25s corta 5%,
# 27s corta 3%, 40s corta 0%.
#
# Por isso o teto por tipo NAO foi reduzido: ele nao cortava nada. O defeito
# dele era outro — podia valer sozinho e consumir o orcamento inteiro —, e quem
# resolve isso e `timeout_efetivo_ms(preservar_retentativa=True)`.
#
# A primeira leitura dessa medicao foi ERRADA, e vale registrar: eu contei 76
# falhas para 38 sucessos e conclui que a taxa de falha era o problema.
# Classificando as falhas por causa (`ferramentas/medir_falhas.py`):
#
#     57%  nossa (propagada): cadeia sem orcamento
#     20%  nossa: sem orcamento para o 2o provedor
#     10%  provedor: ReadTimeout
#      9%  provedor: HTTP 503
#      4%  provedor: outros
#
# 77% nunca chegaram a sair. Falha de infra de verdade sao 16 em 9 runs — menos
# de 2 por run, contra 38 sucessos.
#
# O provedor nao esta quebrado. O que existe e uma cascata: uma chamada lenta
# come o orcamento, e dali em diante toda tentativa nasce impossivel e e
# contabilizada como falha. Foi isso que fez o numero parecer catastrofico, e e
# exatamente o que `preservar_retentativa` ataca.
#
# O que os numeros mostram e que a latencia varia enormemente no MESMO modelo,
# minuto a minuto: 1,2s numa chamada e 504 DEADLINE_EXCEEDED aos 29,1s na
# seguinte. Nao e modelo ruim — e o lado do Google oscilando. Por isso o teto
# fica logo acima da pior resposta BOA observada, e nao abaixo dela.
#
# Sem timeout explícito o SDK usa o default dele, e em produção uma única
# tentativa chegou a durar ~2 minutos. O custo não é a espera: é que o
# screenshot analisado envelhece, o hCaptcha troca o desafio e a resposta nasce
# obsoleta. O freshness guard impede o clique errado, mas cada análise perdida
# é uma rodada jogada fora.
#
# 30 s cobre com MUITA folga os modelos da lista atual (pior chamada medida:
# 4,1 s no 3-flash-preview — ver o comentário de GEMINI_MODELS). A folga é
# proposital: o teto não está aqui para cortar chamada lenta, e sim para
# denunciar pool saturado. Foi exatamente assim que pro-latest, flash-latest e
# 3.6-flash saíram da lista — estouraram o teto em vez de responder.
# Ajustável por ambiente para diagnóstico, com piso de 1 s.
try:
    GEMINI_TIMEOUT_MS = max(1_000, int(os.getenv("GEMINI_TIMEOUT_MS", "20000") or "20000"))
except (ValueError, TypeError):
    GEMINI_TIMEOUT_MS = 20_000

# Tipos de desafio que `_detect_challenge_type` classifica. Vocabulário FECHADO
# e público: quem integra precisa decidir POLÍTICA por tipo — o portal Serviços
# RF, por exemplo, só autoriza resolução automática de alguns deles ao
# representar um CNPJ.
TIPO_NENHUM = "nenhum"
TIPO_GRADE = "grade"
TIPO_GRADE_FUSED = "grade_fused"
# "Clique no animal que a bola nunca toca". Imagem unica e quadrada como o
# grade_fused, mas a resposta nao esta numa imagem: uma bola se move e pausa
# sobre cada animal, e a resposta e quem ela nunca toca. Sem o desvio por
# palavra-chave em `_detect_challenge_type`, isto caia no fallback geometrico
# de `grade_fused` (imagem quadrada, 0 tiles) e ia parar num resolvedor que
# nunca poderia acertar: ele olha UM quadro.
TIPO_BOLA = "bola_em_movimento"
TIPO_CARTAO_ANIMAL = "cartao_animal"
TIPO_IMAGEM = "imagem"
# Só a API pública de INSPEÇÃO devolve este: `_detect_challenge_type` chuta
# `grade` quando a classificação falha, e para quem decide política esse chute
# é perigoso — ver `detectar_tipo_captcha`.
TIPO_DESCONHECIDO = "desconhecido"

TIPOS_CONHECIDOS = (TIPO_NENHUM, TIPO_GRADE, TIPO_GRADE_FUSED, TIPO_BOLA,
                    TIPO_CARTAO_ANIMAL, TIPO_IMAGEM, TIPO_DESCONHECIDO)

CHECKBOX_SEL   = "iframe[src*='hcaptcha.com'][src*='frame=checkbox']"
CHALLENGE_SEL  = "iframe[src*='hcaptcha.com'][src*='frame=challenge']"
TASK_SEL       = ".task"
SUBMIT_SELS    = [
    ".button-submit",
    '[data-cy="submit-button"]',
    ".challenge-submit",
    'button[type="submit"]',
]
PROXIMO_XPATH  = "xpath=/html/body/div/div[2]/div[3]"

# ──────────────────────────────────────────────────────────────────────────────
# Schemas JSON (response_schema para google.genai SDK)
# ──────────────────────────────────────────────────────────────────────────────

_SCHEMA_GRADE = {
    "type": "object",
    "properties": {
        "task_summary": {
            "type": "string",
            "description": "Criterio identificado: texto do enunciado ou categoria da imagem de referencia.",
        },
        "matching_tiles": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "Indices 0-8 dos tiles que atendem ao criterio.",
        },
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
        },
    },
    "required": ["task_summary", "matching_tiles", "confidence"],
}

_SCHEMA_CARTAO_ANIMAL = {
    "type": "object",
    "properties": {
        "carta_0": {
            "type": "string",
            "description": "Animal identificado na Imagem 1 (carta 0 — superior esquerda).",
        },
        "carta_1": {
            "type": "string",
            "description": "Animal identificado na Imagem 2 (carta 1 — superior direita).",
        },
        "carta_2": {
            "type": "string",
            "description": "Animal identificado na Imagem 3 (carta 2 — inferior esquerda).",
        },
        "carta_3": {
            "type": "string",
            "description": "Animal identificado na Imagem 4 (carta 3 — inferior direita).",
        },
        "indice_diferente": {
            "type": "integer",
            "description": "Indice 0-3 da carta que contem o animal UNICO (sem par).",
        },
        "justificativa": {
            "type": "string",
            "description": "Ex.: 'Porco aparece 3x; gato aparece 1x (carta 2)'.",
        },
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
        },
    },
    "required": ["indice_diferente", "confidence"],
}

_SCHEMA_GRID = {
    "type": "object",
    "properties": {
        "instruction": {
            "type": "string",
            "description": "Texto exato da instrucao do captcha.",
        },
        "action": {
            "type": "string",
            "enum": ["click", "type"],
        },
        "click_positions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "col": {
                        "type": "integer",
                        "description": f"Coluna do grid, 0-based (0=esquerda, {GRID_COLS - 1}=direita).",
                    },
                    "row": {
                        "type": "integer",
                        "description": f"Linha do grid, 0-based (0=topo, {GRID_ROWS - 1}=baixo).",
                    },
                    "description": {"type": "string"},
                },
                "required": ["col", "row"],
            },
        },
        "text_answer": {"type": "string"},
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
        },
    },
    "required": ["instruction", "action", "confidence"],
}

# ──────────────────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────────────────

_PROMPT_GRADE = """\
Voce esta resolvendo um desafio hCaptcha. Analise o screenshot com maxima atencao.

=== ESTRUTURA DA IMAGEM ===
- CABECALHO (topo colorido): contem o ENUNCIADO em texto e, as vezes, uma IMAGEM DE REFERENCIA pequena no canto direito.
- GRADE 3x3: 9 tiles logo abaixo, numerados assim:
    ┌───┬───┬───┐
    │ 0 │ 1 │ 2 │  linha superior
    ├───┼───┼───┤
    │ 3 │ 4 │ 5 │  linha do meio
    ├───┼───┼───┤
    │ 6 │ 7 │ 8 │  linha inferior
    └───┴───┴───┘

=== PASSO 1 — LEIA O ENUNCIADO ===
Leia o texto do cabecalho com atencao total. Ha dois tipos de enunciado:

TIPO A — Enunciado direto (ex.: "Selecione todos os onibus", "Click on cars"):
  → Procure exatamente o objeto mencionado no texto.

TIPO B — Enunciado por categoria com imagem de referencia (ex.: "Selecione a imagem da mesma categoria que a imagem de referencia"):
  → Identifique o objeto mostrado na imagem de referencia no canto do cabecalho.
  → Determine a CATEGORIA AMPLA desse objeto conforme a tabela abaixo.
  → NUNCA limite ao objeto exato — inclua todos da mesma categoria.

TIPO C — Enunciado RELACIONAL com imagem de referencia (ex.: "Selecione os itens
que cabem dentro dele", "que sao maiores que ele", "que combinam com ele"):
  → O enunciado aponta para a referencia com um pronome ("dele", "dela", "nele")
    ou com "mostrado acima". Identifique o objeto da referencia primeiro.
  → O criterio NAO e categoria: e a RELACAO que o proprio enunciado descreve.
    Leia a relacao literalmente e aplique-a a cada tile.
  → Categoria aqui atrapalha: numa mala cabem sapato e livro, que nao sao da
    mesma categoria entre si nem da mala. Julgue a relacao, nao o parentesco.
  → NAO amplie o criterio. Ampliar e o certo no TIPO B e o ERRADO aqui: incluir
    "de perto" um item que nao cumpre a relacao e simplesmente errar o tile.

=== TABELA DE CATEGORIAS (use para TIPO B) ===
  aviao, helicoptero, foguete, drone     → "veiculos aereos / transportes"
  carro, trem, onibus, caminhao, barco   → "veiculos terrestres ou aquaticos / transportes"
  aviao + trem + carro (qualquer mistura) → "transportes / veiculos"
  cachorro, gato, coelho, passaro        → "animais"
  rosa, girassol, tulipa, arvore         → "flores / plantas / natureza"
  hamburguer, pizza, fruta, comida       → "alimentos / comida"
  celular, laptop, televisao             → "eletronicos / tecnologia"
  casa, predio, ponte                    → "construcoes / arquitetura"

=== PASSO 2 — ANALISE CADA TILE INDIVIDUALMENTE ===
Examine os tiles 0 a 8 um por um. Para cada tile:
  - Identifique o objeto principal
  - Verifique se pertence ao criterio do enunciado
  - Em caso de duvida razoavel: INCLUA

=== REGRAS CRITICAS — NUNCA IGNORE ===
  !! Retornar lista VAZIA [] e QUASE SEMPRE ERRADO — o hCaptcha sempre tem pelo menos 2 tiles corretos.
  !! Tipicamente de 2 a 5 tiles correspondem ao criterio em cada rodada.
  !! Retornar todos os 9 tiles tambem esta errado.

  As duas regras abaixo valem SO para o TIPO B. Elas mandam ampliar, e ampliar
  e correto quando o criterio e pertencer a uma categoria — mas destrutivo
  quando o criterio e uma relacao (TIPO C): "cabe dentro dele" nao fica mais
  verdadeiro por generosidade, so mais errado.
  !! (TIPO B) Se voce retornou [] nas tentativas anteriores, AMPLIE a categoria e seja mais generoso.
  !! (TIPO B) Se a referencia e um aviao e a grade tem trens e onibus → INCLUA (todos sao transportes).

=== RETORNE ===
  task_summary: criterio identificado de forma clara (TIPO B: a categoria ampla, ex.: "transportes / veiculos"; TIPO C: a relacao e a referencia, ex.: "cabe dentro da mala")
  matching_tiles: lista de indices 0-8 (NUNCA retorne lista vazia sem antes ampliar a categoria)
  confidence: "high" | "medium" | "low"
"""

_PROMPT_GRADE_COM_REF = """\
Voce esta resolvendo um hCaptcha do tipo "mesma categoria que a imagem de referencia".

=== IMAGENS RECEBIDAS ===
  IMAGEM 1 — Screenshot completo do desafio (cabecalho + grade 3x3 com 9 tiles).
  IMAGEM 2 — A IMAGEM DE REFERENCIA isolada e ampliada, extraida do cabecalho.

=== PASSO 1 — ANALISE A IMAGEM DE REFERENCIA (IMAGEM 2) ===
Identifique o objeto mostrado na IMAGEM 2 e determine sua CATEGORIA AMPLA.

TABELA OBRIGATORIA DE CATEGORIAS:
  aviao / helicoptero / foguete / drone           → categoria: "veiculos aereos / transportes"
  carro / trem / onibus / caminhao / barco / moto → categoria: "veiculos / transportes"
  QUALQUER veiculo (aereo, terrestre, aquatico)   → categoria: "transportes / veiculos"
  cachorro / gato / coelho / passaro / peixe      → categoria: "animais"
  rosa / girassol / tulipa / planta / arvore      → categoria: "flores / plantas / natureza"
  hamburguer / pizza / fruta / prato de comida    → categoria: "alimentos / comida"
  celular / laptop / tablet / televisao           → categoria: "eletronicos / tecnologia"
  casa / predio / ponte / monumento               → categoria: "construcoes / arquitetura"

REGRA FUNDAMENTAL: a categoria e SEMPRE mais ampla que o objeto especifico.
  → Se a IMAGEM 2 mostra um AVIAO, a categoria e "transportes / veiculos" — NAO "avioes".
  → Se a IMAGEM 2 mostra um CACHORRO, a categoria e "animais" — NAO "cachorros".

=== PASSO 2 — ANALISE CADA TILE DA GRADE (IMAGEM 1) ===
Grade numerada:
    ┌───┬───┬───┐
    │ 0 │ 1 │ 2 │  linha superior
    ├───┼───┼───┤
    │ 3 │ 4 │ 5 │  linha do meio
    ├───┼───┼───┤
    │ 6 │ 7 │ 8 │  linha inferior
    └───┴───┴───┘

Para cada tile (0 a 8), identifique o objeto principal e verifique se pertence a mesma categoria ampla.
  → Se referencia = aviao: inclua TRENS, ONIBUS, CARROS, BARCOS, MOTOS, OUTROS AVIOES — todos sao transportes.
  → Qualquer angulo, cor, estilo artistico, foto parcial — inclua se o objeto for da categoria.

=== REGRAS CRITICAS — NUNCA IGNORE ===
  !! Lista vazia [] e QUASE SEMPRE ERRADA. O hCaptcha garante pelo menos 2 tiles corretos por rodada.
  !! Se voce esta em duvida entre incluir ou nao um tile: INCLUA.
  !! Tipicamente 2 a 5 tiles sao corretos em cada rodada.
  !! Se sua analise retornar 0 tiles, releia a categoria e amplie — voce esta sendo restrito demais.

=== RETORNE ===
  task_summary: categoria ampla identificada (ex.: "transportes / veiculos")
  matching_tiles: indices 0-8 dos tiles corretos (lista com pelo menos 1 elemento)
  confidence: "high" | "medium" | "low"
"""

_PROMPT_GRID_TMPL = """\
Voce esta resolvendo um captcha de imagem com um GRID {cols}x{rows} desenhado sobre ela.

=== INSTRUCAO DO CAPTCHA ===
{instruction}

=== COMO LER O GRID ===
Cada celula esta rotulada com "col,row" no seu canto superior esquerdo.
  col: 0 = coluna mais a ESQUERDA, {max_col} = coluna mais a DIREITA
  row: 0 = linha mais ao TOPO, {max_row} = linha mais ABAIXO
As linhas do grid sao vermelhas. Os rotulos sao amarelos com sombra preta.

=== TAREFA ===
1. Leia a instrucao do captcha.
2. Identifique TODOS os objetos que atendem ao criterio.
3. Para cada objeto, leia o rotulo "col,row" da celula que cobre o CENTRO do objeto.
4. Se o centro estiver entre duas celulas, escolha a que cobre mais do objeto.

=== REGRAS ===
  - Nao omita nenhum objeto valido.
  - col deve estar entre 0 e {max_col}.
  - row deve estar entre 0 e {max_row}.
  - confidence = "low" apenas se a imagem estiver ilegivel ou os objetos nao visiveis.

=== RETORNE ===
  instruction: a instrucao do captcha
  action: "click" (ou "type" se for captcha de texto)
  click_positions: lista de celulas {{col, row, description}} para cada objeto
  confidence: "high" | "medium" | "low"
"""

_PROMPT_CARTAO_ANIMAL = """\
Voce esta resolvendo um captcha de CARTAS COM ANIMAIS (grid 2x2 animado).

=== IMAGENS RECEBIDAS ===
Voce recebeu 4 screenshots — um por carta, capturados individualmente durante a animacao de revelacao:
  IMAGEM 1 → Carta 0  (posicao: superior esquerda do grid 2x2)
  IMAGEM 2 → Carta 1  (posicao: superior direita)
  IMAGEM 3 → Carta 2  (posicao: inferior esquerda)
  IMAGEM 4 → Carta 3  (posicao: inferior direita)

=== TAREFA ===
A instrucao do captcha e: "Selecione o cartao com um animal diferente"
  → 3 cartas mostram o MESMO animal (maioria)
  → 1 carta mostra um animal DIFERENTE (minoria/unico)
  → Voce deve identificar qual carta (indice 0-3) tem o animal diferente.

=== PASSO A PASSO ===
1. Identifique o animal em cada carta (0, 1, 2 e 3). Se uma imagem mostrar a carta fechada
   (cor solida sem animal), registre como "vazio".
2. Conte quantas vezes cada especie aparece entre as 4 cartas.
3. O animal que aparece APENAS UMA VEZ e o diferente.
4. Retorne o indice (0, 1, 2 ou 3) dessa carta.

=== REGRAS CRITICAS ===
  !! Se uma imagem mostrar a carta fechada (sem animal visivel), ignore essa carta na contagem.
  !! Considere variacoes de especie: elefante/elefante-bebe sao a mesma especie.
  !! confidence = "low" apenas se voce nao conseguiu identificar os animais.
  !! indice_diferente DEVE ser exatamente 0, 1, 2 ou 3.

=== RETORNE ===
  carta_0, carta_1, carta_2, carta_3: nome do animal (ou "vazio" se nao visivel)
  indice_diferente: 0, 1, 2 ou 3 (a carta com o animal unico)
  justificativa: breve explicacao (ex.: "Corvo aparece 1x na carta 2; elefante aparece 3x nas demais")
  confidence: "high" | "medium" | "low"
"""

# ──────────────────────────────────────────────────────────────────────────────
# Debug screenshots
# ──────────────────────────────────────────────────────────────────────────────

_DEBUG_DIR = os.path.join(os.path.dirname(__file__), "debug_screenshots")
_debug_counter: int = 0


_AMOSTRAS_GUARDADAS: set = set()


def _guardar_amostra(page, tipo: str, instrucao: str = "") -> None:
    """Guarda UMA amostra do desafio que o solver nao conseguiu resolver.

    Existe porque melhorar um resolvedor exige iterar contra o desafio REAL, e
    sem amostra so resta gastar run atras de run — foi assim que o desafio da
    bola foi de 0 para 3/3, testando offline contra quadros arquivados.

    Uma coleta anterior fazia isso sem freio: 40 screenshots a cada
    classificacao, dentro de `detectar_tipo_captcha`, que a lib de login chama
    REPETIDAMENTE enquanto aguarda o desfecho. Na pratica a run ficava
    capturando sem parar, e os 20s de captura saiam ANTES de o relogio do
    orcamento comecar. Por isso aqui:

      - atras de `CAPTCHA_DEBUG_AMOSTRAS_DIR`: sem a variavel, nada acontece e
        o comportamento e identico ao de antes;
      - UMA por tipo por processo, nao uma por rodada;
      - so no caminho em que o solver JA desistiu, onde nao ha mais orcamento
        a proteger;
      - um screenshot, nao quarenta.

    Salva o ENUNCIADO junto. E ele que indexa o catalogo: as imagens mudam a
    cada desafio, o texto da instrucao se repete.
    """
    destino = os.environ.get("CAPTCHA_DEBUG_AMOSTRAS_DIR", "").strip()
    if not destino or tipo in _AMOSTRAS_GUARDADAS:
        return
    _AMOSTRAS_GUARDADAS.add(tipo)
    try:
        os.makedirs(destino, exist_ok=True)
        marca = f"{time.strftime('%Y%m%d-%H%M%S')}-{tipo}"

        # DESAFIO ANIMADO PRECISA DE SEQUENCIA, nao de um retrato.
        #
        # Um screenshot so nao permite testar nada de um formato cuja resposta
        # existe no MOVIMENTO — e formato animado novo e exatamente o que a
        # coleta deveria destravar. Em 08/09/2026 apareceu "clique na flor em
        # que a abelha nunca pousa", falhou, e a amostra guardada era uma foto:
        # inutil para reproduzir o problema fora da run.
        #
        # Os quadros saem com o mesmo mecanismo da resolucao — clip e
        # `animations="allow"` —, entao o que fica no disco e o que o resolvedor
        # teria visto, e nao uma aproximacao.
        if tipo == TIPO_BOLA:
            quadros, _caixa = _capturar_frames_bola(page)
            for i, q in enumerate(quadros):
                with open(os.path.join(destino, f"{marca}_f{i:02d}.png"), "wb") as f:
                    f.write(q)
            print(f"    [captcha] Sequência guardada: {len(quadros)} quadros.")
        png, _caixa = _capturar_desafio(page)
        if png:
            with open(os.path.join(destino, f"{marca}.png"), "wb") as f:
                f.write(png)
        texto = instrucao or _extrair_instrucao(page) or ""
        with open(os.path.join(destino, f"{marca}.txt"), "w", encoding="utf-8") as f:
            f.write(texto)
        print(f"    [captcha] Amostra guardada em {destino} ({marca}).")
    except Exception:  # noqa: BLE001, S110 — coleta nunca derruba a resolucao
        pass


_SCHEMA_TRIAGEM = {
    "type": "object",
    "properties": {
        "instrucao_lida": {"type": "string"},
        "mecanica": {"type": "string"},
        "alvos": {"type": "string"},
        "acao_necessaria": {"type": "string"},
        "e_animado": {"type": "boolean"},
        "quantos_cliques": {"type": "integer"},
        "por_que_falhou": {"type": "string"},
        "familia_conhecida": {"type": "boolean"},
    },
    "required": ["mecanica", "acao_necessaria", "e_animado", "quantos_cliques"],
}

_PROMPT_TRIAGEM = """Você está olhando um captcha hCaptcha que uma automação NÃO conseguiu resolver.
Não resolva o desafio. Descreva a MECÂNICA dele, para que um humano decida se
vale escrever um resolvedor novo.

A automação já sabe lidar com esta família: "leia a instrução e clique em UM
alvo" — seja num quadro parado (ex.: o ícone diferente dos demais) ou numa
sequência animada (ex.: o alvo que o elemento móvel nunca alcança).

Responda:
  instrucao_lida    o enunciado, exatamente como está escrito na imagem
  mecanica          em uma frase: o que o desafio pede que se faça
  alvos             o que são os elementos clicáveis (ícones, fotos, animais...)
  acao_necessaria   "clicar" | "arrastar" | "ordenar" | "digitar" | outro
  e_animado         true se algo se move; false se a cena é estática
  quantos_cliques   quantos elementos precisam ser clicados para responder
  familia_conhecida true se cai na família descrita acima; false se é outra coisa
  por_que_falhou    seu palpite do que impediu a automação de resolver
"""


def _diagnosticar_desafio(page, api_key: str, tipo: str, instrucao: str = "") -> None:
    """Descreve a MECANICA de um desafio que o solver nao resolveu.

    Uma amostra sozinha diz "nao consegui". Isto diz "nao consegui, e o que vi
    foi isto" — que e a diferenca entre abrir a imagem e adivinhar, e ler uma
    triagem pronta.

    Custa UMA chamada ao modelo, so quando a resolucao ja terminou em fracasso e
    so com `CAPTCHA_DEBUG_AMOSTRAS_DIR` definida. Nao roda dentro do laco de
    rodadas: ali ainda ha orcamento de tempo a proteger, e um desafio que ainda
    pode ser resolvido nao precisa de autopsia.
    """
    destino = os.environ.get("CAPTCHA_DEBUG_AMOSTRAS_DIR", "").strip()
    if not destino or not api_key:
        return
    try:
        png, _caixa = _capturar_desafio(page)
        if not png:
            return
        conteudo = [_PROMPT_TRIAGEM, _parte_imagem(png)]
        # Politica propria e curta: isto e diagnostico, nao resolucao. Se
        # demorar, o valor dele ja passou.
        d = _gemini_call(conteudo, _SCHEMA_TRIAGEM, api_key, "triagem",
                         PoliticaLatencia(timeout_ms=15_000,
                                          fim=time.monotonic() + 20.0))
        marca = f"{time.strftime('%Y%m%d-%H%M%S')}-{tipo}"
        os.makedirs(destino, exist_ok=True)
        linhas = [
            f"# Triagem — desafio nao resolvido ({tipo})",
            "",
            f"- instrucao na tela : {instrucao or d.get('instrucao_lida', '?')}",
            f"- mecanica          : {d.get('mecanica', '?')}",
            f"- alvos             : {d.get('alvos', '?')}",
            f"- acao necessaria   : {d.get('acao_necessaria', '?')}",
            f"- animado           : {d.get('e_animado')}",
            f"- cliques           : {d.get('quantos_cliques')}",
            f"- familia conhecida : {d.get('familia_conhecida')}",
            f"- por que falhou    : {d.get('por_que_falhou', '?')}",
            "",
            "Familia conhecida = a automacao ja tem resolvedor para a mecanica.",
            "Se for false, e formato novo e precisa de trabalho.",
        ]
        with open(os.path.join(destino, f"{marca}-triagem.md"), "w",
                  encoding="utf-8") as f:
            f.write(chr(10).join(linhas) + chr(10))
        print(f"    [captcha] Triagem gravada | familia_conhecida="
              f"{d.get('familia_conhecida')} acao={d.get('acao_necessaria')!r} "
              f"cliques={d.get('quantos_cliques')} animado={d.get('e_animado')}")
    except Exception:  # noqa: BLE001, S110 — autopsia nunca derruba nada
        pass


def _salvar_debug(png: bytes, sufixo: str = "") -> None:
    """Salva o PNG em debug_screenshots/ para inspeção visual do que foi enviado ao Gemini."""
    global _debug_counter
    _debug_counter += 1
    try:
        os.makedirs(_DEBUG_DIR, exist_ok=True)
        nome = f"{_debug_counter:03d}_{sufixo}.png"
        path = os.path.join(_DEBUG_DIR, nome)
        with open(path, "wb") as f:
            f.write(png)
        print(f"    [captcha/debug] Screenshot salvo: {path}")
    except Exception as e:
        print(f"    [captcha/debug] Erro ao salvar: {type(e).__name__}")


# ──────────────────────────────────────────────────────────────────────────────
# Cliente Gemini
# ──────────────────────────────────────────────────────────────────────────────

_client_cache: Optional[object] = None


def _get_client(api_key: str):
    global _client_cache
    if _client_cache is None:
        if not _GENAI:
            raise RuntimeError(
                "google.genai nao disponivel. Instale: pip install google-genai"
            )
        _client_cache = _genai_lib.Client(api_key=api_key)
    return _client_cache


class PoliticaLatencia(NamedTuple):
    """Orçamento de tempo de UMA chamada a `solve_hcaptcha`. Sem estado global.

    Existe porque consumidores diferentes têm paciências diferentes. Na
    representação de CNPJ o portal impõe seu próprio ritmo, e uma resolução que
    se estende por minutos chega tarde demais para servir; no captcha de login
    o padrão de 30 s continua sendo o certo.

    `fim` é um instante MONOTÔNICO. O timeout que vai para a request é sempre
    `min(timeout_ms, tempo restante)` — com 8 s de orçamento sobrando, nenhuma
    chamada individual pode reservar 30 s.
    """

    timeout_ms: int = GEMINI_TIMEOUT_MS
    fim: float | None = None

    @property
    def restante_ms(self) -> int:
        """Milissegundos até o fim do orçamento. `-1` = sem orçamento total."""
        if self.fim is None:
            return -1
        return int(max(0.0, self.fim - time.monotonic()) * 1000)

    @property
    def esgotado(self) -> bool:
        return self.fim is not None and self.restante_ms <= 0

    def timeout_efetivo_ms(self, preservar_retentativa: bool = False,
                           piso_ms: int = 0, reserva_ms: int = 0) -> int:
        """O teto real desta request: nunca além do que sobra do orçamento.

        Com `preservar_retentativa`, também nunca além da METADE do que sobra —
        nenhuma tentativa pode consumir a chance da próxima.

        Essa segunda regra nasceu da RUN-ef3f4b9d, 11/09/2026. A PREMIUM TEXTIL
        caiu num captcha trivial ("clique em todos os objetos feitos
        principalmente de metal", dois baldes óbvios) e terminou marcada como
        "exige validação manual". A causa, em três linhas de log:

            Teto por chamada ajustado ao tipo: 10s -> 40s (grade)
            falha na chamada | gemini-3.5-flash-lite | ReadTimeout
            parando a cadeia do Gemini com 12s

        Orçamento de 55s, teto de 40s para grade. Uma requisição que ficou
        pendurada levou os 40 — e os 12 que sobraram não cabem em ninguém: o
        mínimo viável são 10s para o Gemini e 10s para o segundo provedor. O
        rodízio de modelos funcionou, o alternativo foi acionado, e herdou um
        orçamento impossível.

        Nada disso tinha a ver com dificuldade: nenhum modelo chegou a ver a
        imagem.

        Metade, e não um valor fixo: com 55s de orçamento o primeiro tiro vale
        27,5s, que ainda é folgado sobre os 19,8s da pior chamada BOA já
        medida (ver o comentário de `GEMINI_TIMEOUT_MS`). Um teto fixo menor
        cortaria chamadas boas — foi o erro de 12s, corrigido em 26/08 — e um
        fixo maior repetiria este caso. A fração se ajusta ao orçamento de cada
        consumidor sem precisar de tabela.

        Na ÚLTIMA tentativa possível não se reserva nada: aí gastar tudo é o
        certo, porque não há próxima para proteger.
        """
        if self.fim is None:
            return self.timeout_ms
        # `reserva_ms` e o que fica guardado para OUTRO provedor. Sai do teto
        # desta chamada, senao ela gasta para dentro da reserva e a guarda que
        # decide continuar a cadeia chega tarde.
        disponivel = self.restante_ms - reserva_ms if reserva_ms else self.restante_ms
        teto = min(self.timeout_ms, max(piso_ms, disponivel)
                   if reserva_ms else self.restante_ms)
        if preservar_retentativa:
            # A fração NUNCA empurra abaixo do piso de quem vai ser chamado.
            #
            # Sem isto a reserva fabrica exatamente o que veio eliminar. Um
            # teste que já existia — a run real reproduzida, orçamento 30s e
            # teto de 10s — mostrou a terceira chamada caindo para 5s, abaixo
            # dos 10s que a API do Gemini exige ("Manually set deadline 5s is
            # too short"). Guardar para a próxima tentativa não pode custar a
            # viabilidade DESTA.
            #
            # Quando não cabem os dois, a preferência é a tentativa de agora:
            # ela tem uma imagem fresca na mão, e a próxima ainda pode nem
            # acontecer.
            teto = max(min(teto, self.restante_ms // 2), min(teto, piso_ms))
        return max(1, teto)


POLITICA_PADRAO = PoliticaLatencia()


def _politica(politica) -> PoliticaLatencia:
    """`None` significa o comportamento de sempre — nada de default implícito."""
    return politica if isinstance(politica, PoliticaLatencia) else POLITICA_PADRAO


def _make_config(schema: dict, model: str = GEMINI_MODEL, timeout_ms: int | None = None,
                 sem_opcionais: bool = False):
    """GenerateContentConfig com response_schema e thinking conforme THINKING_BUDGET.

    THINKING_BUDGET == 0 (padrão): NÃO enviamos thinking_config — o modelo usa seu
    default, que para os flash-lite é um raciocínio mínimo (rápido). Isso é o que
    dá agilidade SEM quebrar a chamada. IMPORTANTE: mandar thinking_budget=0
    explicitamente faz estes modelos (flash-lite-latest / 3.x) responderem
    400 INVALID_ARGUMENT — eles aceitam um budget POSITIVO (ex.: 4096) ou nenhum,
    mas não o valor 0. Por isso o 0 vira "omitir", nunca "enviar 0".

    THINKING_BUDGET > 0: envia thinking_config com esse orçamento (troca
    velocidade por acurácia). Os legados 2.0-flash não aceitam thinking_config.
    """
    kwargs: dict = {
        "temperature": 0.0,
        "response_mime_type": "application/json",
        "response_schema": schema,
    }
    # `sem_opcionais` e a segunda tentativa depois de um 400: manda o
    # minimo que a API exige. Ver o tratamento de `requisicao_invalida`
    # em `_gemini_call` — a lista estatica de modelos sem thinking
    # envelhece a cada release, entao a verdade vem da resposta.
    _sem_thinking = ("2.0-flash",)
    if (not sem_opcionais and THINKING_BUDGET > 0
            and not any(m in model for m in _sem_thinking)):
        try:
            kwargs["thinking_config"] = _gt.ThinkingConfig(thinking_budget=THINKING_BUDGET)
        except Exception:
            pass
    # Teto por tentativa. Fica na config, e não no cliente, porque o cliente é
    # cacheado no módulo: aqui o limite acompanha cada chamada e é inspecionável.
    try:
        kwargs["http_options"] = _gt.HttpOptions(
            timeout=GEMINI_TIMEOUT_MS if timeout_ms is None else timeout_ms)
    except Exception:  # noqa: BLE001, S110 — SDK antigo sem HttpOptions
        pass
    try:
        return _gt.GenerateContentConfig(**kwargs)
    except Exception:
        # Fallback sem os opcionais que versões antigas não aceitam
        kwargs_safe = {k: v for k, v in kwargs.items()
                       if k not in ("thinking_config", "http_options")}
        return _gt.GenerateContentConfig(**kwargs_safe)


def _is_overloaded_error(e) -> bool:
    """True quando vale a pena TROCAR DE MODELO.

    Dois casos distintos, ambos resolvidos pelo fallback:
      - sobrecarga transitória: 503/UNAVAILABLE/overloaded/429/RESOURCE_EXHAUSTED
      - modelo morto: 404 "no longer available" — o Google aposenta IDs fixos
        (gemini-2.0-flash e gemini-2.5-flash já respondem 404 para chaves novas).
        Sem o 404 aqui, um modelo aposentado derruba a resolução inteira em vez
        de cair para o próximo da lista.
      - estouro do teto de tempo: `GEMINI_TIMEOUT_MS`. A lista de modelos está
        ordenada por latência MEDIDA, então o próximo é literalmente o mais
        rápido dos que restam — insistir no que acabou de estourar é a pior
        escolha disponível.

    O que NÃO entra aqui, de propósito: erro de autenticação, chave inválida e
    argumento inválido. Esses não melhoram trocando de modelo, e deixá-los
    circular pela lista transformaria um erro de configuração em quatro chamadas
    inúteis e um diagnóstico pior.

    O parágrafo acima sempre foi a INTENÇÃO; a implementação não a cumpria. Ela
    procurava substrings no texto inteiro do erro, então um 400 cujo corpo
    mencionasse "timeout" ou "deadline" — ou que trouxesse "400" em qualquer
    campo — entrava como sobrecarga e circulava pela lista exatamente como o
    docstring dizia que não deveria. Agora quem decide é a CATEGORIA, que vai
    pelo status HTTP quando ele existe.
    """
    return _categoria_do_erro(e) in (
        "indisponivel", "limite_de_uso", "modelo_ausente", "tempo_esgotado")


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostico de erro externo — sem corpo, sem mensagem do provedor
# ──────────────────────────────────────────────────────────────────────────────
#
# `str(e)` de um erro do google.genai carrega o JSON cru da resposta. Este
# solver roda dentro de automacoes cujo stdout vira log de execucao na
# plataforma, entao o corpo do provedor estava chegando ao registro da run.
# Erros do Playwright, por sua vez, embutem seletor, URL do frame e trechos do
# DOM.
#
# A regra passa a ser: nada que venha de fora entra no log em texto. Sai apenas
# o que e NOSSO — categoria de vocabulario fechado, nome de modelo da nossa
# lista, status numerico e o nome da classe. Ler `str(e)` para CLASSIFICAR
# continua permitido; o que nao pode e imprimi-lo.

_CATEGORIAS_ERRO = (
    ("indisponivel",        ("unavailable", "overloaded", "high demand")),
    ("limite_de_uso",       ("resource_exhausted", "rate limit")),
    ("modelo_ausente",      ("not_found", "not found", "no longer available",
                             "not available")),
    ("tempo_esgotado",      ("timeout", "timed out", "deadline")),
    ("credencial",          ("api key", "unauthorized", "permission_denied")),
    ("requisicao_invalida", ("invalid_argument",)),
)

# O STATUS manda. A busca por substring so entra quando nao ha status nenhum.
#
# Antes a classificacao era so por substring, e a lista e ordenada: o par
# ("timeout", "timed out", "deadline") vinha ANTES de "400". Um 400 cujo CORPO
# mencionasse qualquer uma dessas tres palavras saia classificado como
# `tempo_esgotado`. O historico do orquestrador mostra o sintoma direto: a mesma
# sequencia de erro, tentativa 1 registrada como `tempo_esgotado` e tentativa 2
# como `requisicao_invalida` — o mesmo 400, duas categorias, porque o texto do
# corpo variava entre elas.
#
# Nao era so log errado. `_is_overloaded_error` usa as MESMAS substrings, entao
# o 400 tambem passava por indisponibilidade: a cadeia percorria todos os
# modelos com a requisicao invalida intacta, e desde `_penalizar` cada volta
# ainda mandava um modelo saudavel para o banco de reservas.
_CATEGORIA_POR_STATUS = {
    400: "requisicao_invalida",
    401: "credencial",
    403: "credencial",
    404: "modelo_ausente",
    408: "tempo_esgotado",
    429: "limite_de_uso",
    503: "indisponivel",
    504: "tempo_esgotado",
}


def _categoria_do_erro(e) -> str:
    """Categoria de vocabulario FECHADO. Nunca devolve texto do provedor."""
    status = _status_do_erro(e)
    if status is not None and status in _CATEGORIA_POR_STATUS:
        return _CATEGORIA_POR_STATUS[status]
    # Sem status: o NOME DA CLASSE entra junto do texto. Um estouro de tempo do
    # httpx chega como `ReadTimeout` com `str(e)` vazio — so o texto nao
    # classificava isso, e ele caia em `desconhecido`, que nao troca de modelo.
    s = f"{type(e).__name__} {e or ''}".lower()
    for categoria, marcas in _CATEGORIAS_ERRO:
        if any(m in s for m in marcas):
            return categoria
    return "desconhecido"


def _status_do_erro(e) -> int | None:
    """Status HTTP, se houver. Inteiro entre 100 e 599 — e nada alem disso."""
    for atributo in ("code", "status_code", "status"):
        valor = getattr(e, atributo, None)
        if isinstance(valor, int) and not isinstance(valor, bool) and 100 <= valor <= 599:
            return valor
    achado = re.search(r"\b([1-5]\d{2})\b", str(e or ""))
    if achado:
        return int(achado.group(1))
    return None


def _despejar_erro_para_diagnostico(e, tag: str, modelo: str | None = None) -> None:
    """Grava o erro CRU em arquivo local — nunca no stdout.

    A regra do bloco acima continua valendo integralmente: o log da run nao
    recebe texto de fora. Mas ha diagnostico que so o corpo responde, e hoje ha
    um em aberto — os 400 aparecem nos tres resolvedores (grade, grade_fused e
    bola), com modelos diferentes recusando a MESMA requisicao em 2 segundos.
    Isso descarta disponibilidade e aponta para um campo invalido comum as tres
    chamadas; qual campo, so o corpo diz, e ele nunca foi gravado em lugar
    nenhum.

    Fica atras de `CAPTCHA_DEBUG_ERRO_DIR`. Sem a variavel nada e escrito e o
    comportamento e identico ao de antes — e por isso que isto pode existir num
    pacote de producao. Aponte para pasta LOCAL da maquina que investiga, nunca
    para dentro de algo que suba para a plataforma.
    """
    destino = os.environ.get("CAPTCHA_DEBUG_ERRO_DIR", "").strip()
    if not destino:
        return
    try:
        os.makedirs(destino, exist_ok=True)
        caminho = os.path.join(
            destino, f"{time.strftime('%Y%m%d-%H%M%S')}-{tag}-{os.getpid()}.txt")
        corpo = ""
        resposta = getattr(e, "response", None)
        for atributo in ("text", "content"):
            bruto = getattr(resposta, atributo, None)
            if bruto:
                corpo = bruto if isinstance(bruto, str) else repr(bruto)
                break
        with open(caminho, "w", encoding="utf-8") as f:
            f.write(f"tag={tag}\nmodelo={modelo}\ntipo={type(e).__name__}\n"
                    f"status={_status_do_erro(e)}\n"
                    f"categoria={_categoria_do_erro(e)}\n\n"
                    f"--- str(e) ---\n{e}\n\n--- response ---\n{corpo}\n")
    except Exception:  # noqa: BLE001, S110 — diagnostico nunca derruba a resolucao
        pass


def _diagnostico_erro(e, modelo: str | None = None) -> str:
    """Linha de log de um erro externo. So campos sob nosso controle.

    `modelo` so aparece se estiver na nossa lista — um nome vindo de outro
    lugar nao e nosso e nao entra.
    """
    partes = []
    if modelo and modelo in GEMINI_MODELS:
        partes.append(f"modelo={modelo}")
    partes.append(f"categoria={_categoria_do_erro(e)}")
    partes.append(f"tipo={type(e).__name__}")
    status = _status_do_erro(e)
    if status is not None:
        partes.append(f"status={status}")
    return " | ".join(partes)


# ──────────────────────────────────────────────────────────────────────────────
# Banco de reservas: quem falhou descansa, mas volta
# ──────────────────────────────────────────────────────────────────────────────
#
# `GEMINI_MODELS` é a ordem de preferência, medida e fixa. O que muda dentro de
# uma execução é quem está RESPONDENDO agora.
#
# Duas versões deste código já erraram, em direções opostas:
#
#   1. Sem memória nenhuma: `_gemini_call` recomeçava do primeiro modelo a cada
#      chamada. Um pool saturado custava o timeout inteiro, caía para o
#      alternativo, resolvia — e no captcha seguinte pagava tudo de novo,
#      redescobrindo o que já sabia.
#
#   2. Memória definitiva: o modelo que falhasse UMA vez saía para sempre. Num
#      log de produção isso esvaziou a bancada em dois minutos — um 503 no
#      flash-lite, um ReadTimeout no flash, e sobrou só o 3.1-flash-lite, que é
#      justamente o mais lento (cauda de 25s). Sem alternativa, cada chamada
#      seguinte pagava o timeout cheio. A correção deixou pior do que achou.
#
# O erro da segunda foi de LEITURA: 503 e ReadTimeout dizem "ocupado agora", não
# "morto". Tratar pico como sentença descarta justamente o modelo mais rápido.
#
# Daí o banco de reservas: quem falha descansa um tempo que CRESCE a cada falha
# seguida, e volta. Um acerto zera a conta. Se todos estiverem descansando, joga
# o que descansou mais — uma tentativa lenta é melhor do que nenhuma.
# A API RECUSA deadline abaixo de 10s: "Manually set deadline 8s is too
# short. Minimum allowed deadline is 10s." — um 400 que a sondagem leria
# como "modelo indisponivel", condenando todos por um erro nosso.
TIMEOUT_MINIMO_API_MS = 10_000

_DESCANSOS = (60.0, 300.0, 900.0)      # 1min, 5min, 15min

# modelo -> [falhas seguidas, instante em que pode voltar]
_BANCO: dict[str, list] = {}


def _pode_jogar(model: str) -> bool:
    estado = _BANCO.get(model)
    return not estado or time.monotonic() >= estado[1]


def _voltar_em(model: str) -> float:
    estado = _BANCO.get(model)
    return estado[1] if estado else 0.0


def modelos_ativos() -> list[str]:
    """Os modelos prontos para jogar AGORA, na ordem de preferência.

    Vazio nunca: se todos estão descansando, devolve o que volta primeiro. Uma
    chamada lenta ainda resolve captcha; nenhuma chamada não resolve nada.
    """
    prontos = [m for m in GEMINI_MODELS if _pode_jogar(m)]
    if prontos:
        return prontos
    return [min(GEMINI_MODELS, key=_voltar_em)]


def _penalizar(model: str, motivo: str) -> None:
    """Manda o modelo para o banco por um tempo crescente."""
    falhas = _BANCO.get(model, [0, 0.0])[0] + 1
    descanso = _DESCANSOS[min(falhas, len(_DESCANSOS)) - 1]
    _BANCO[model] = [falhas, time.monotonic() + descanso]
    restantes = [m for m in GEMINI_MODELS if m != model and _pode_jogar(m)]
    print(f"    [captcha] '{model}' descansa {descanso / 60:.0f}min "
          f"({falhas}ª falha seguida: {motivo}). "
          + (f"Em campo: {', '.join(restantes)}" if restantes
             else "TODOS descansando — o próximo a voltar joga assim mesmo."))


def _premiar(model: str) -> None:
    """Um acerto zera a ficha: o modelo estava só ocupado, não quebrado."""
    if model in _BANCO:
        del _BANCO[model]


def preparar_modelos(api_key: str | None = None,
                     timeout_ms: int = 12_000) -> list[str]:
    """Sonda os modelos UMA vez e bane os que nem respondem. Devolve os que ficam.

    Feita para rodar no início da automação. O que ela pega bem é o modelo
    MORTO — um ID aposentado que responde 404 — e esse não vale nem uma
    tentativa depois.

    O que ela NÃO prevê é saturação: num log de produção os três passaram na
    sondagem e dois falharam no primeiro captcha, minutos depois. Por isso a
    sondagem não reordena nada e não promete nada — quem cuida do pico é o
    banco de reservas, durante o uso.

    Nunca levanta e nunca esvazia a lista.
    """
    api_key = api_key or os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        print("    [captcha] sem GEMINI_API_KEY; sondagem de modelos pulada.")
        return list(GEMINI_MODELS)

    try:
        client = _get_client(api_key)
    except Exception as e:
        print(f"    [captcha] sondagem de modelos pulada ({type(e).__name__}).")
        return list(GEMINI_MODELS)

    print("    [captcha] sondando os modelos do Gemini...")
    vivos = []
    for model in GEMINI_MODELS:
        inicio = time.monotonic()
        try:
            client.models.generate_content(
                model=model,
                contents=["responda apenas: ok"],
                config=_gt.GenerateContentConfig(
                    http_options=_gt.HttpOptions(
                        timeout=max(TIMEOUT_MINIMO_API_MS, timeout_ms)),
                    max_output_tokens=8,
                ),
            )
        except Exception as e:
            diag = _diagnostico_erro(e, model)
            # Só o modelo MORTO fica de fora de saída. Um pico na sondagem não
            # pode custar o modelo mais rápido pela execução inteira — foi
            # exatamente esse excesso que esvaziou a bancada antes.
            if "404" in diag or "not_found" in diag.lower():
                print(f"    [captcha]   {model}: NÃO EXISTE mais ({diag})")
                _BANCO[model] = [len(_DESCANSOS), float("inf")]
                continue
            print(f"    [captcha]   {model}: instável agora ({diag}) — "
                  "fica de reserva, sem ser banido")
            _penalizar(model, "não respondeu à sondagem")
            vivos.append(model)
            continue
        gasto = time.monotonic() - inicio
        _premiar(model)
        vivos.append(model)
        print(f"    [captcha]   {model}: ok ({gasto:.1f}s)")

    print(f"    [captcha] modelos desta execução: {', '.join(vivos) or '(nenhum)'}")
    return vivos


# ──────────────────────────────────────────────────────────────────────────────
# Carta de ACURACIA: um segundo provedor, para quando o Gemini responde e erra
# ──────────────────────────────────────────────────────────────────────────────
#
# O comentario de GEMINI_MODELS registra o buraco: "`_gemini_call` so troca de
# modelo em erro de DISPONIBILIDADE (503/404/timeout), nunca por resposta
# errada. O pro era a 'carta de acuracia' quando os flash erram, mas esse papel
# nunca existiu de fato". Tentar preenche-lo com outro Gemini falhou porque um
# modelo alcancado so quando os outros caem precisa ser o MAIS disponivel, e o
# pro era o menos.
#
# Um provedor DIFERENTE resolve o que outro Gemini nao resolvia:
#   - cota e pool independentes — um 429 do Google nao o afeta;
#   - nao deterministico (recusa temperature=0), entao repetir tem valor, ao
#     contrario do Gemini com temperature=0, onde a mesma imagem no mesmo modelo
#     devolve byte a byte a mesma resposta.
#
# QUANDO ele entra: so depois que o rodizio ja ouviu TODOS os modelos do Gemini.
# Com 3 modelos, as rodadas 1-3 ouvem os tres e a 4a em diante repetiria um deles
# — rodada que hoje e desperdicio garantido. O astra ocupa esse espaco, nao o de
# ninguem.
#
# Isso distribui sozinho da forma medida em 08/09/2026:
#     _solve_bola    (2 rodadas) nunca o alcanca — astra fez 0/3 na animacao
#     _solve_imagem  (5 rodadas) alcanca na 4a   — astra fez 3/3 em imagem unica
#
# Sem OPENAI_API_KEY, nada disso existe e o comportamento e o de antes.
OPENAI_MODEL_PADRAO = "gpt-6-astra"

# Teto de tempo do segundo provedor, em segundos.
#
# Ele NAO pode herdar o do Gemini. O astra so e chamado depois que o Gemini
# nao fechou, entao herdar o orcamento significa receber o que sobrou de quem
# acabou de falhar — e quanto PIOR o Gemini esta, MENOS tempo o substituto
# ganha. Exatamente ao contrario do que um substituto precisa.
#
# Medido em 09/09/2026: os tres modelos do Gemini devolveram 504
# DEADLINE_EXCEEDED gastando a rodada inteira, e o astra foi chamado quatro
# vezes com ~10s cada. Nenhuma das quatro chegou a receber resposta. Uma
# chamada de visao com imagem de ~450 KB nao cabe em 10s.
ASTRA_TIMEOUT_MIN_S = 30.0

# Abaixo disto a chamada ao segundo provedor nao se faz.
#
# Medido em 09/09/2026: com teto de 30s ele respondeu em 6,9s. Nas onze
# chamadas seguintes o orcamento ja estava no fim e o teto caiu para 1,0s —
# porque eu escrevi `max(1.0, min(teto, restante))`, que FABRICA uma chamada
# impossivel em vez de recusar. Onze idas ao servidor com um segundo de prazo,
# todas condenadas antes de sair.
#
# E o mesmo principio do `GEMINI_DEADLINE_MIN_MS`: se o prazo nao permite a
# resposta, a chamada nao e uma tentativa, e desperdicio com aparencia de
# tentativa — e ainda polui o diagnostico com timeouts que nao dizem nada sobre
# o provedor.
class SegundoProvedorSemOrcamento(RuntimeError):
    """O segundo provedor NAO foi chamado: nao havia tempo viavel.

    Nao e falha do provedor, e a distincao importa. No log as duas saiam
    identicas — `segundo provedor tambem falhou | categoria=desconhecido |
    tipo=RuntimeError` —, porque `_diagnostico_erro` nunca imprime o texto do
    erro (regra de privacidade, e ela esta certa). A frase "nao chamado" ficava
    invisivel.

    Medido em 11/09/2026, RUN-ef3f4b9d, PREMIUM TEXTIL: doze "falhas" seguidas
    do segundo provedor, todas assim. O captcha era "Clique em todos os objetos
    feitos principalmente de metal", com dois baldes obvios — nada tinha a ver
    com dificuldade. A run concluiu "exige validacao manual" e mandou uma
    pessoa resolver um desafio trivial.

    As duas pedem acoes opostas: provedor quebrado se investiga na chave e na
    API; sem orcamento se resolve dando mais tempo ou parando antes.
    """


ASTRA_DEADLINE_MIN_S = 10.0

# O Gemini RECUSA prazo abaixo disto, com 400 INVALID_ARGUMENT:
#     "Manually set deadline 2s is too short. Minimum allowed deadline is 10s."
#
# A gente passava o restante do orcamento como deadline da chamada. Com o
# orcamento no fim, isso virava 2s — e o 400 que apareceu quatro vezes em dois
# dias era NOSSO pedido impossivel, nao um campo invalido. Cheguei a codificar
# um contorno para `thinking_config` por causa disso, tratando o sintoma errado.
#
# Abaixo deste piso a chamada nao pode dar certo, entao nao se faz: gasta-se
# uma ida ao servidor para receber a recusa e o desafio envelhece de graca.
GEMINI_DEADLINE_MIN_MS = 10_000

# Falhas que dizem "o PROVEDOR esta fora", e nao "este MODELO nao serviu".
#
# Os tres modelos do Gemini nao sao alternativas independentes: mesma
# infraestrutura, mesma cota, mesma fila. Medido em 09/09/2026 as 14:59 —
# 3.5-flash devolveu 503 "high demand" e 3.1-flash-lite devolveu 503 igual, um
# atras do outro. Percorrer a lista inteira e descobrir tres vezes a mesma
# indisponibilidade, pagando o orcamento por cada descoberta.
#
# E o preco nao para ai: com o orcamento gasto, a chamada seguinte ao segundo
# provedor herda o troco. Na mesma run, o astra respondeu em 6,9s quando teve
# 30s, e deu timeout quando sobrou o resto.
#
# So o que e comprovadamente da CONTA, e nao do modelo.
#
# Eu tinha posto 503 e timeout aqui tambem, e a suite derrubou os dois com
# evidencia melhor que a minha:
#
#   timeout — um teste reproduz uma run real em que DOIS modelos deram
#             ReadTimeout e o TERCEIRO respondeu. E lentidao momentanea de um
#             modelo, nao queda do provedor.
#   503     — a mensagem do proprio Google e "This MODEL is currently
#             experiencing high demand". Ele afirma o escopo, e o escopo e o
#             modelo. Dois 503 seguidos em 09/09 me pareceram prova de queda
#             geral; eram duas ocorrencias independentes.
#
# 429 e diferente: cota e da chave, nao do modelo. Trocar de modelo com a cota
# estourada e pagar outra ida para ouvir o mesmo nao.
CATEGORIAS_DE_PROVEDOR_FORA = frozenset({
    "limite_de_uso",    # 429 — cota da conta, comum aos tres
})

# A partir de QUAL rodada o segundo provedor responde. SEGUNDA — pedido do Jean,
# e a insistencia dele estava certa.
#
# Historia curta desta constante, porque ela mudou duas vezes hoje e as duas
# tinham defesa:
#
#   len(GEMINI_MODELS)  "so depois de ouvir todos os Gemini"
#   2                   "a rodada 2 e o flash COMPLETO, um degrau real"
#   1                   agora
#
# O que derrubou as duas: em producao, o que NAO fecha e o formato ESTATICO
# ("clique na figura diferente"), e nele o segundo provedor mediu 3/3 contra as
# amostras arquivadas — enquanto o Gemini nao fechava. Segurar ele ate a 3a
# rodada e adiar o unico que acertou, e a rodada 3 as vezes nem chega: medido em
# 08/09/2026, a rodada 2 levou 39 s por ReadTimeout e banco de modelos, e o
# orcamento acabou antes.
#
# Meu erro foi dividir isto em dois limites com base numa medicao de TRES
# amostras da bola — um formato que nao e o que falha. Uma amostra pequena de um
# caso irrelevante nao pode vetar o que a producao pede.
# ZERO: o segundo provedor pergunta PRIMEIRO.
#
# Ele entrou como reserva, para a rodada 2 em diante, quando a suposicao era
# que o Gemini resolveria a maioria e o outro cobriria a excecao. O dia
# 09/09/2026 mediu o contrario:
#
#     Gemini   30 falhas    (14 + 10 + 6 nos tres modelos)
#     astra     2 chamadas com prazo adequado  ->  2 respostas (6,9s e 17,0s)
#
# Nao e caso isolado: foram 504 DEADLINE_EXCEEDED e 503 "high demand" o dia
# todo, nos tres modelos, e o astra so nao respondia quando chegava nele com o
# troco do orcamento. Manter como reserva quem responde, e como principal quem
# nao responde, custa o orcamento inteiro para descobrir todo dia a mesma coisa.
#
# O Gemini continua na cadeia, como alternativa: quando o astra falhar, ele e
# tentado logo em seguida. Inverteu-se a ordem, nao se removeu ninguem.
RODIZIO_DO_SEGUNDO_PROVEDOR = 0


_AVISOU_SEM_SEGUNDO_PROVEDOR = False


# Quem pergunta PRIMEIRO, por tipo de desafio.
#
# Inverter tudo de uma vez foi erro meu, e a run das 16:36 de 09/09/2026 cobrou:
# quinze rodadas de `grade`, quinze respostas do astra em 3,5-6,1s, e nenhuma
# fechou o captcha. O orcamento acabou e o desafio foi para triagem.
#
# A medicao que ja existia neste arquivo dizia isso, e eu passei por cima:
#
#     grade 3x3    Gemini 16/16 medido       astra NUNCA medido
#     imagem/bola  Gemini nao fechava        astra 3/3 nas amostras arquivadas
#
# Sao formatos diferentes. Na grade o modelo escolhe entre 9 tiles com um
# rotulo textual — trabalho que o Gemini faz bem e barato. No estatico e na
# animacao ele precisa localizar um alvo num fundo desenhado para atrapalhar
# visao computacional, e ali quem acerta e o outro.
#
# Quem responde primeiro deve ser quem ACERTA naquele formato, e nao quem
# respondeu mais rapido no formato ao lado. Velocidade sem acerto so gasta o
# orcamento mais depressa — foi exatamente o que aconteceu.
ORDEM_DO_SEGUNDO_PROVEDOR = {
    TIPO_GRADE:         2,   # Gemini primeiro: 16/16 medido aqui
    TIPO_GRADE_FUSED:   2,
    TIPO_IMAGEM:        0,   # astra primeiro: 3/3 no estatico
    TIPO_BOLA:          0,
    TIPO_CARTAO_ANIMAL: 0,
    # O resolvedor de imagem chama `_gemini_grid`, cuja tag e "grid" e nao
    # "imagem". Ele ja caia no astra, mas por DEFAULT — dava o resultado certo
    # pelo motivo errado, e sumiria no dia em que o default mudasse. Entra
    # explicito.
    "grid":             0,
    # A triagem descreve um desafio que a automacao NAO resolveu. E o caso mais
    # dificil por definicao, e o unico cujo produto e uma explicacao em vez de
    # um clique.
    "triagem":          0,
}


def _rodizio_do_segundo_provedor(tag: str) -> int:
    """A partir de que rodada o segundo provedor entra, para este desafio.

    `tag` e o nome do resolvedor que esta chamando (`grade`, `imagem`, `bola`,
    `grid`...). Desconhecido cai no padrao, que continua sendo perguntar cedo:
    formato novo e justamente onde o Gemini menos costuma fechar.
    """
    for tipo, ordem in ORDEM_DO_SEGUNDO_PROVEDOR.items():
        if tag.startswith(tipo) or tipo.startswith(tag):
            return ordem
    return RODIZIO_DO_SEGUNDO_PROVEDOR


def _astra_configurado() -> bool:
    """Ha segundo provedor? E se NAO ha, isso aparece no log.

    O silencio aqui custou uma tarde inteira em 08/09/2026. A chave da OpenAI
    foi gravada em nivel de usuario, mas o agente ja estava rodando — no Windows
    o ambiente e copiado no nascimento do processo, entao ele nunca a viu.
    `_astra_configurado()` devolvia False e o segundo provedor era pulado sem
    dizer nada.
    
    Passamos horas ajustando EM QUE RODADA ele deveria entrar sem perceber que
    ele nao podia entrar em nenhuma. Um aviso de uma linha teria encerrado isso
    na primeira run.
    
    Avisa UMA vez por processo: e configuracao, nao evento — repetir a cada
    rodada so encheria o log.
    """
    global _AVISOU_SEM_SEGUNDO_PROVEDOR
    if os.environ.get("OPENAI_API_KEY", "").strip():
        return True
    if not _AVISOU_SEM_SEGUNDO_PROVEDOR:
        _AVISOU_SEM_SEGUNDO_PROVEDOR = True
        print("    [captcha] SEM segundo provedor: OPENAI_API_KEY ausente neste "
              "processo. Quem a injeta é o runner, a partir do Cofre da "
              "plataforma — então o que falta é o segredo cadastrado com o "
              "alias exato e VINCULADO a esta automação. Definir a variável na "
              "máquina não resolve: o ambiente do agente é copiado quando ele "
              "nasce, e a run herda o dele.")
    return False


def _contents_para_openai(contents: list, schema: dict) -> list:
    """Traduz o `contents` do Gemini para o formato de mensagem da OpenAI.

    Os solvers montam `contents` com strings e `Part.from_bytes` do SDK do
    Google. Traduzir aqui — e nao neles — e o que mantem os cinco resolvedores
    intocados.

    O schema vai no TEXTO porque o modo `json_object` da OpenAI exige a palavra
    "json" na mensagem e nao aceita `response_schema`. Medido: sem isso a API
    recusa com 400.
    """
    import base64
    import json as _json
    blocos = []
    for parte in contents:
        if isinstance(parte, str):
            blocos.append({"type": "text", "text": parte})
            continue
        dados = mime = None
        for atrib in ("inline_data", "_inline_data"):
            inline = getattr(parte, atrib, None)
            if inline is not None:
                dados = getattr(inline, "data", None)
                mime = getattr(inline, "mime_type", None) or "image/jpeg"
                break
        if dados:
            b64 = base64.b64encode(dados).decode("ascii")
            blocos.append({
                "type": "image_url",
                # `detail: high` desliga a reducao automatica da OpenAI. Sem
                # isso o modelo nao le a grade sobreposta, que e de onde sai a
                # posicao do clique.
                "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"},
            })
    blocos.append({"type": "text", "text":
                   "Responda em JSON, um unico objeto, seguindo exatamente este "
                   "schema:" + chr(10) + _json.dumps(schema, ensure_ascii=False) +
                   chr(10) + "Nada de texto fora do JSON."})
    return blocos


def _tracar_astra(linha: str) -> None:
    """Uma linha por chamada ao segundo provedor, em arquivo local.

    O `print` acima ja vai para o log da run, e o agente nao corta linha
    nenhuma — mas "cade o astra?" foi perguntado seis vezes em dois dias, e
    todas as respostas dependiam de alguem abrir a timeline da plataforma e
    ler. Sucesso nao deixava rastro nenhum em disco: so falha gravava arquivo,
    entao "funcionou" e "nem foi chamado" eram indistinguiveis daqui.

    Vai para o MESMO diretorio de diagnostico dos erros, que ja e local, ja e
    opcional e ja fica fora do que sobe para a plataforma.
    """
    destino = os.environ.get("CAPTCHA_DEBUG_ERRO_DIR", "").strip()
    if not destino:
        return
    try:
        os.makedirs(destino, exist_ok=True)
        with open(os.path.join(destino, "astra-chamadas.log"), "a",
                  encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"pid={os.getpid()} {linha}" + chr(10))
    except Exception:  # noqa: BLE001 — diagnostico nunca derruba a run
        pass


def _astra_call(contents: list, schema: dict, tag: str,
                politica: PoliticaLatencia | None = None) -> dict:
    """Uma chamada ao segundo provedor. Levanta como qualquer outra falha."""
    import json as _json

    from openai import OpenAI
    chave = os.environ.get("OPENAI_API_KEY", "").strip()
    modelo = os.environ.get("OPENAI_MODEL", "").strip() or OPENAI_MODEL_PADRAO
    base = os.environ.get("OPENAI_BASE_URL", "").strip() or None
    politica = _politica(politica)
    # Piso proprio, e nunca alem do que resta do orcamento total.
    #
    # `max_retries=0` porque o padrao do SDK e 2: sob teto apertado, a
    # repeticao interna divide o mesmo tempo em tentativas ainda menores e
    # transforma uma chance ruim em tres chances piores — tudo isso invisivel,
    # porque o SDK so levanta o erro no fim. Quem retenta aqui e o laco de
    # fora, que recaptura a tela antes de perguntar de novo.
    restante_s = politica.restante_ms / 1000 if politica.fim is not None else None
    teto_s = max(politica.timeout_efetivo_ms() / 1000, ASTRA_TIMEOUT_MIN_S)
    if restante_s is not None:
        if restante_s < ASTRA_DEADLINE_MIN_S:
            raise SegundoProvedorSemOrcamento(
                f"segundo provedor não chamado: restam {restante_s:.1f}s e o "
                f"mínimo viável é {ASTRA_DEADLINE_MIN_S:.0f}s "
                f"(medido: responde em ~7s quando tem prazo)")
        teto_s = min(teto_s, restante_s)
    cliente = OpenAI(api_key=chave, base_url=base,
                     timeout=teto_s, max_retries=0)
    print(f"    [captcha/{tag}] Gemini não fechou — perguntando ao segundo "
          f"provedor.")
    _tracar_astra(f"CHAMANDO tag={tag} modelo={modelo} teto={teto_s:.1f}s")
    # Sem `temperature`: este modelo recusa 0 ("Only the default (1) value is
    # supported") e responder com o padrao e o que o torna util aqui — duas
    # perguntas iguais podem dar respostas diferentes.
    _t0 = time.monotonic()
    resp = cliente.chat.completions.create(
        model=modelo,
        messages=[{"role": "user",
                   "content": _contents_para_openai(contents, schema)}],
        response_format={"type": "json_object"},
    )
    _tracar_astra(f"RESPONDEU tag={tag} em {time.monotonic() - _t0:.1f}s")
    return _json.loads(resp.choices[0].message.content)


def _gemini_call(contents: list, schema: dict, api_key: str, tag: str,
                 politica: PoliticaLatencia | None = None,
                 rodizio: int = 0,
                 rodizio_segundo_provedor: int | None = None,
                 direto_ao_segundo: bool = False) -> dict:
    """Chama o Gemini com FALLBACK de modelos quando o principal está sobrecarregado.

    Para cada modelo em GEMINI_MODELS, tenta GEMINI_TRIES_PER_MODEL vezes com backoff
    curto. Se o modelo estiver indisponível (503/sobrecarga), passa para o próximo da
    lista. Retorna o JSON já parseado; levanta RuntimeError se todos falharem.
    """
    client = _get_client(api_key)
    politica = _politica(politica)
    last_exc = None
    ativos = modelos_ativos()
    # `rodizio` gira a ordem: e o que faz a RETENTATIVA perguntar a OUTRO
    # modelo. Com temperature=0.0 a mesma imagem no mesmo modelo devolve a
    # mesma resposta — repetir "confianca baixa" cinco vezes no mesmo modelo
    # era garantido nao mudar nada, so gastar chamada.
    # Esgotado o rodizio, a proxima rodada repetiria um modelo ja ouvido com a
    # mesma imagem — resposta identica garantida. E o espaco onde o segundo
    # provedor cabe sem tirar o lugar de ninguem.
    limite_segundo = (_rodizio_do_segundo_provedor(tag)
                      if rodizio_segundo_provedor is None
                      else rodizio_segundo_provedor)
    # `not politica.esgotado` aqui tambem, e nao so na cadeia esgotada la
    # embaixo. Enquanto este ponto de entrada valia da 2a rodada em diante, o
    # relogio ja tinha sido conferido no caminho; com o segundo provedor em
    # PRIMEIRO, ele e a primeira coisa que roda, e sem esta guarda uma politica
    # ja vencida ainda dispararia uma chamada. Orcamento estourado nao melhora
    # trocando de provedor — piora, porque a tela envelhece mais.
    if rodizio >= limite_segundo and ativos and not politica.esgotado:
        # Por que ele NAO foi chamado, quando nao foi.
        #
        # Em 08/09/2026 os arquivos de diagnostico mostravam so modelos Gemini,
        # e nao havia como distinguir tres casos muito diferentes: o astra nao
        # esta configurado, o astra foi chamado e falhou, ou o rodizio nem
        # chegou nele. Os tres deixavam o mesmo rastro — nenhum. A pergunta
        # "cade o astra?" foi feita cinco vezes sem que os arquivos pudessem
        # responder.
        if not _astra_configurado():
            _despejar_erro_para_diagnostico(
                RuntimeError("OPENAI_API_KEY ausente no processo da run"),
                f"{tag}-astra-ausente", "astra")
        else:
            try:
                return _astra_call(contents, schema, tag, politica)
            except Exception as e:  # noqa: BLE001
                print(f"    [captcha/{tag}] segundo provedor falhou | "
                      f"{_diagnostico_erro(e)} — voltando ao Gemini.")
                _despejar_erro_para_diagnostico(e, f"{tag}-astra", "astra")
    # `direto_ao_segundo`: pula o Gemini e vai ao segundo goleiro.
    #
    # Existe para o caso em que o Gemini RESPONDEU e a resposta nao serve —
    # confianca baixa, tiles vazios. Ate 11/09/2026 isso mandava a pergunta de
    # volta para o Gemini com outro modelo, e o segundo provedor so era
    # consultado quando havia ERRO de chamada. Erro e imprecisao sao a mesma
    # coisa do ponto de vista de quem espera a resposta: o Gemini nao fechou.
    #
    # Com `temperature=0.0` o Gemini tende a repetir a propria resposta; a
    # rotacao de modelo atenua, mas quem muda de verdade e trocar de PROVEDOR.
    if direto_ao_segundo and _astra_configurado():
        # A recusa por orcamento e tratada AQUI, como no caminho normal.
        #
        # Sem este `except` ela escapava como excecao qualquer e o chamador a
        # registrava como "Gemini erro (tentativa 2)" — culpando o provedor
        # errado e ainda disparando "voltando ao Gemini". Medido em
        # 11/09/2026, RUN-910bd939:
        #
        #     tiles vazios — indo ao segundo provedor (tentativa 1)...
        #     Gemini erro (tentativa 2) | tipo=SegundoProvedorSemOrcamento
        #
        # Nao ha para onde cair: o Gemini nesse ponto ja falhou, e a decisao de
        # vir para ca foi justamente por isso. Entao a recusa sobe com o tipo
        # dela, e quem espera a resposta sabe que foi falta de tempo.
        try:
            return _astra_call(contents, schema, tag, _politica(politica))
        except SegundoProvedorSemOrcamento:
            _p = _politica(politica)
            print(f"    [captcha/{tag}] segundo provedor NAO chamado: "
                  f"restam {_p.restante_ms / 1000:.1f}s e o minimo viavel e "
                  f"{ASTRA_DEADLINE_MIN_S:.0f}s. Nao e falha dele.")
            raise

    if rodizio and len(ativos) > 1:
        giro = rodizio % len(ativos)
        ativos = ativos[giro:] + ativos[:giro]
    # Reserva de orcamento para o segundo provedor.
    #
    # Medido em 09/09/2026: os tres modelos do Gemini devolveram 504
    # DEADLINE_EXCEEDED, consumiram o orcamento inteiro, e o astra foi recusado
    # com "restam 0.0s". Ele e o unico que respondeu hoje — 6,9s no `grade` e
    # 17,0s no `grid` — e nao chegou a ser perguntado.
    #
    # Gastar ate a ultima gota no provedor que esta falhando, e so entao
    # procurar alternativa, e a ordem errada: quanto pior o primeiro esta, mais
    # caro fica descobrir isso, e menos sobra para quem poderia responder.
    #
    # A reserva vale APENAS quando ha segundo provedor configurado. Sem ele nao
    # ha para quem guardar, e encurtar a cadeia do Gemini so tiraria tentativas
    # sem dar nada em troca.
    reserva_ms = (int(ASTRA_DEADLINE_MIN_S * 1000)
                  if (_astra_configurado() and politica.fim is not None)
                  else 0)
    provedor_fora = False
    for mi, model in enumerate(ativos):
        # Depois do primeiro modelo falhar, a alternativa de verdade e o OUTRO
        # PROVEDOR — nao o proximo modelo do mesmo. Ver
        # `MODELOS_GEMINI_ANTES_DO_SEGUNDO`.
        # Condicionado a HAVER segundo provedor — nao a `reserva_ms`.
        #
        # `reserva_ms` so e diferente de zero quando ha astra E existe orcamento
        # total (`politica.fim`). Amarrar a troca de provedor a isso deixava de
        # fora quem chama sem prazo total: ali o Gemini rodaria os tres modelos
        # de novo, que e exatamente o comportamento que esta mudanca remove.
        # A pergunta certa e "existe alternativa?", e nao "existe orcamento?".
        if _astra_configurado() and mi >= MODELOS_GEMINI_ANTES_DO_SEGUNDO:
            print(f"    [captcha/{tag}] {mi} modelo(s) do Gemini falharam — "
                  f"indo ao segundo provedor em vez de tentar o proximo "
                  f"(restam {politica.restante_ms / 1000:.0f}s).")
            break
        if reserva_ms and politica.restante_ms < reserva_ms + GEMINI_DEADLINE_MIN_MS:
            print(f"    [captcha/{tag}] parando a cadeia do Gemini com "
                  f"{politica.restante_ms / 1000:.0f}s: o resto e reserva do "
                  f"segundo provedor, que precisa de "
                  f"{ASTRA_DEADLINE_MIN_S:.0f}s.")
            break
        if provedor_fora:
            print(f"    [captcha/{tag}] '{model}' e os demais ficam de fora: "
                  f"a cota é da CHAVE, não do modelo. Indo direto ao segundo "
                  f"provedor.")
            break
        if politica.timeout_efetivo_ms() < GEMINI_DEADLINE_MIN_MS:
            print(f"    [captcha/{tag}] restam "
                  f"{politica.timeout_efetivo_ms() / 1000:.0f}s e o Gemini "
                  f"recusa prazo abaixo de "
                  f"{GEMINI_DEADLINE_MIN_MS / 1000:.0f}s — não vale a ida.")
            break
        if politica.esgotado:
            # Orçamento acabou: tentar o próximo modelo só adiaria o mesmo
            # desfecho, agora com o screenshot ainda mais velho.
            print(f"    [captcha/{tag}] orçamento de tempo esgotado — "
                  "encerrando a cadeia de modelos.")
            break
        sem_opcionais = False
        for attempt in range(1, GEMINI_TRIES_PER_MODEL + 1):
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=_make_config(
                        schema, model,
                        # Preserva metade do que sobra, EXCETO na última
                        # chance real — último modelo, última tentativa, e sem
                        # segundo provedor para quem guardar. Aí gastar tudo é
                        # o certo: não há próxima para proteger.
                        # A reserva do segundo provedor sai do teto DESTA
                        # chamada, e nao so da decisao de continuar a cadeia.
                        #
                        # O guard de reserva roda uma vez por MODELO; dentro de
                        # cada um cabem duas tentativas, e elas gastavam para
                        # dentro da reserva. Foi assim que o astra chegou a ser
                        # recusado com 9,2s quando precisa de 10 — a decisao de
                        # parar estava certa e tarde.
                        timeout_ms=politica.timeout_efetivo_ms(
                            preservar_retentativa=not (
                                mi == len(ativos) - 1
                                and attempt == GEMINI_TRIES_PER_MODEL
                                and not _astra_configurado()),
                            piso_ms=GEMINI_DEADLINE_MIN_MS,
                            reserva_ms=reserva_ms),
                        sem_opcionais=sem_opcionais),
                )
                if sem_opcionais:
                    print(f"    [captcha/{tag}] Respondeu SEM os campos "
                          f"opcionais — o 400 vinha de thinking_config em "
                          f"'{model}'.")
                _premiar(model)
                if mi > 0:
                    print(f"    [captcha/{tag}] Resolvido com modelo alternativo '{model}'.")
                return json.loads(resp.text)
            except Exception as e:
                last_exc = e
                print(f"    [captcha/{tag}] falha na chamada ao modelo | "
                      f"{_diagnostico_erro(e, model)} | "
                      f"tentativa={attempt}/{GEMINI_TRIES_PER_MODEL}")
                _despejar_erro_para_diagnostico(e, tag, model)
                if _categoria_do_erro(e) in CATEGORIAS_DE_PROVEDOR_FORA:
                    provedor_fora = True

                # 400 = a requisição é NOSSA, e há um suspeito nomeado no
                # docstring de `_make_config`: estes modelos respondem
                # INVALID_ARGUMENT a valores de `thinking_config` que não
                # aceitam, e a lista de exclusão só cobre `2.0-flash`.
                #
                # Em vez de adivinhar quais modelos suportam o quê — lista que
                # envelhece a cada release do Google —, tenta UMA vez sem os
                # campos opcionais. Se passar, a causa era essa e fica dito no
                # log; se não passar, o 400 é de outra coisa e a cadeia segue.
                #
                # Isto conserta E diagnostica: era o único jeito de saber, já
                # que o corpo do erro nunca chega ao log por regra de higiene.
                if (_categoria_do_erro(e) == "requisicao_invalida"
                        and not sem_opcionais):
                    print(f"    [captcha/{tag}] requisição inválida — repetindo "
                          "sem os campos opcionais (thinking_config).")
                    sem_opcionais = True
                    continue
                # Sobrecarga/timeout é do POOL daquele modelo, não da chamada:
                # a segunda tentativa longa no mesmo modelo só gasta a validade
                # do screenshot. Os logs de produção mostram exatamente isso —
                # `flash-latest 1/2: 503` e sucesso imediato no modelo seguinte.
                if _is_overloaded_error(e):
                    break
                if attempt < GEMINI_TRIES_PER_MODEL and not politica.esgotado:
                    time.sleep(min(2 ** attempt, 8))
        # Esgotou as tentativas neste modelo.
        if not _is_overloaded_error(last_exc):
            break  # erro não é de sobrecarga — trocar de modelo não ajuda
        # Caiu por DISPONIBILIDADE: e do POOL, nao da chamada. Vai para o
        # banco por um tempo — insistir nele no proximo captcha custaria o
        # timeout inteiro de novo, e bani-lo de vez esvazia a bancada.
        _penalizar(model, _diagnostico_erro(last_exc, model))
        if mi < len(ativos) - 1:
            print(f"    [captcha/{tag}] '{model}' indisponível — tentando modelo alternativo...")
    # CADEIA ESGOTADA: o segundo provedor e a ultima chance antes de desistir.
    #
    # O outro ponto de entrada dele (rodizio esgotado) so serve quando o Gemini
    # RESPONDE e nao fecha. Nao cobre o caso que mais dói: a chave do Gemini com
    # a cota estourada. Um 429 na PRIMEIRA rodada derruba a cadeia inteira antes
    # de qualquer rodizio, e o desafio morre sem nenhum modelo ter olhado a
    # imagem — indistinguivel, no log e no desfecho, de "o captcha era dificil".
    #
    # Vale para qualquer motivo de esgotamento, e nao so cota: pool
    # indisponivel, chave invalida, requisicao recusada. O que todos tem em
    # comum e que sao problemas DO GOOGLE ou da nossa integracao com ele — e o
    # segundo provedor nao compartilha nenhum dos dois.
    #
    # NAO entra se o orcamento de tempo acabou: ai o problema e o relogio, que
    # ele tambem nao resolve, e a chamada so chegaria com o screenshot mais
    # velho ainda.
    if _astra_configurado() and not politica.esgotado:
        try:
            return _astra_call(contents, schema, tag, politica)
        except SegundoProvedorSemOrcamento:
            # Nao e falha: e a guarda de viabilidade recusando uma chamada que
            # ja nasceria condenada.
            #
            # A mensagem e montada AQUI, dos numeros que a gente calcula, e nao
            # interpolando a excecao. O gate de higiene de logs proibe o
            # segundo, e com razao: texto de excecao pode carregar conteudo do
            # provedor, e abrir excecao para "esta aqui e nossa" e como a regra
            # morre. Quem precisa saber e o restante em segundos, que temos.
            print(f"    [captcha/{tag}] segundo provedor NAO chamado: "
                  f"restam {politica.restante_ms / 1000:.1f}s e o minimo "
                  f"viavel e {ASTRA_DEADLINE_MIN_S:.0f}s. Nao e falha dele.")
        except Exception as e:  # noqa: BLE001
            print(f"    [captcha/{tag}] segundo provedor também falhou | "
                  f"{_diagnostico_erro(e)}")

    # A mensagem desta excecao tambem e log: quem a captura acima imprime.
    raise RuntimeError(
        f"Gemini {tag}: falhou em todos os modelos ({_diagnostico_erro(last_exc)})")


# ──────────────────────────────────────────────────────────────────────────────
# Detecção de frame e tipo de desafio
# ──────────────────────────────────────────────────────────────────────────────

def _frame_tem_captcha_ativo(frame) -> bool:
    """Verifica via JS se o frame tem desafio visível e habilitado.

    O hCaptcha pré-carrega múltiplos iframes frame=challenge no DOM; apenas um
    está ativo. Esta função distingue o ativo dos inativos verificando:
      1. .challenge-container com dimensões reais (>= 100x100 px)
      2. .prompt-text com texto de instrução preenchido
      3. .button-submit com aria-disabled != "true"
    """
    try:
        return bool(frame.evaluate("""() => {
            const c = document.querySelector('.challenge-container');
            if (!c) return false;
            const r = c.getBoundingClientRect();
            if (r.width < 100 || r.height < 100) return false;
            const p = document.querySelector('.prompt-text');
            if (!p || !p.textContent.trim()) return false;
            const b = document.querySelector('.button-submit');
            if (!b || b.getAttribute('aria-disabled') === 'true') return false;
            return true;
        }"""))
    except Exception:
        return False


def _get_challenge_frame(page):
    """Retorna o frame ATIVO do desafio hCaptcha ou None.

    Itera page.frames verificando conteúdo interno — evita capturar o iframe
    inativo quando múltiplos frame=challenge estão pré-carregados no DOM.
    """
    for f in page.frames:
        url = f.url or ""
        if "hcaptcha.com" not in url or "frame=challenge" not in url:
            continue
        if _frame_tem_captcha_ativo(f):
            return f

    # Fallback: element_handle → content_frame
    try:
        el = page.locator(CHALLENGE_SEL).first.element_handle(timeout=1_000)
        if el:
            f = el.content_frame()
            if f and _frame_tem_captcha_ativo(f):
                return f
    except Exception:
        pass
    return None


def _get_active_iframe_index(page) -> int:
    """Retorna o índice (0-based) do iframe de desafio ATIVO, ou -1."""
    try:
        count = page.locator(CHALLENGE_SEL).count()
    except Exception:
        return -1
    for idx in range(count):
        try:
            el = page.locator(CHALLENGE_SEL).nth(idx).element_handle(timeout=500)
            if el:
                f = el.content_frame()
                if f and _frame_tem_captcha_ativo(f):
                    return idx
        except Exception:
            continue
    return -1


def _get_challenge_frame_locator(page):
    """FrameLocator apontando para o iframe de desafio ATIVO."""
    idx = _get_active_iframe_index(page)
    if idx >= 0:
        return page.frame_locator(CHALLENGE_SEL).nth(idx)
    return page.frame_locator(CHALLENGE_SEL).first


def _get_challenge_element_locator(page):
    """Locator do elemento <iframe> ativo (para screenshot e bounding_box)."""
    idx = _get_active_iframe_index(page)
    if idx >= 0:
        return page.locator(CHALLENGE_SEL).nth(idx)
    return page.locator(CHALLENGE_SEL).first


def _challenge_visible(page) -> bool:
    return _get_challenge_frame(page) is not None


def detectar_tipo_captcha(page) -> str:
    """Classifica o desafio ATUAL — INSPEÇÃO, nunca resolução.

    Existe para que o integrador decida POLÍTICA POR TIPO. O portal Serviços RF
    apresenta, ao representar um CNPJ, desafios de formatos diferentes: alguns a
    automação resolve, outros não. Sem saber o tipo ANTES, a única escolha seria
    tentar tudo — e uma tentativa de ~40 s num formato que não vai sair é tempo
    perdido e chamada ao modelo desperdiçada.

    Devolve um de `TIPOS_CONHECIDOS`. Ao contrário do uso interno, **não chuta
    `grade` quando a classificação falha**: devolve `TIPO_DESCONHECIDO`, porque
    quem decide política precisa distinguir "é grade" de "não consegui saber".

    Nunca levanta.
    """
    try:
        if _get_challenge_frame(page) is None:
            return TIPO_NENHUM
        tipo = _detect_challenge_type(page, ao_falhar=TIPO_DESCONHECIDO)
    except Exception:  # noqa: BLE001 — indeterminado é "não sei classificar"
        return TIPO_DESCONHECIDO
    return tipo if tipo in TIPOS_CONHECIDOS else TIPO_DESCONHECIDO


def captcha_presente(page) -> bool:
    """Há hCaptcha aguardando interação nesta página? — DETECÇÃO, sem resolver.

    Existe para quem precisa SABER que o captcha apareceu sem pedir que ele
    seja resolvido: o portal Serviços RF apresenta um segundo desafio ao
    representar um CNPJ, e esse não é automatizado — vai para intervenção
    humana. Chamar `solve_hcaptcha` só para descobrir se há algo ali gastaria
    chamadas ao modelo e clicaria em tiles que ninguém pediu.

    Cobre os dois estados: o desafio ABERTO (`frame=challenge` ativo, com
    enunciado e submit habilitado) e o widget "Sou humano" ainda fechado
    (`frame=checkbox`) — dos dois lados o fluxo está parado esperando alguém.

    O checkbox precisa estar VISÍVEL. Existir no DOM não basta: o hCaptcha
    deixa seus iframes para trás, e num fluxo onde um captcha antecede outro —
    login e depois representação, no portal Serviços RF — o widget da etapa
    anterior continua no documento. `count() > 0` chamava isso de "captcha
    aguardando interação" e mandava o integrador para um ramo de captcha que
    não existia mais.

    Nunca levanta: o chamador está justamente perguntando sobre um estado
    incerto, e uma exceção aqui viraria ruído no diagnóstico dele.
    """
    try:
        if _challenge_visible(page):
            return True
        return _indice_checkbox_visivel(page) is not None
    except Exception:  # noqa: BLE001 — indeterminado é "não detectei"
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Freshness guard — a resposta so vale para o desafio que a originou
# ──────────────────────────────────────────────────────────────────────────────
#
# O defeito que isto fecha foi observado em producao: uma chamada ao modelo
# chegou a levar ~2 minutos (503 no primeiro modelo, fallback no segundo) e,
# nesse intervalo, o hCaptcha trocou o desafio. A resposta do desafio A foi
# aplicada ao desafio B.
#
# O que tornava o bug SILENCIOSO: os cliques recriam o locator, e locators do
# Playwright resolvem na hora do clique. O locator "fresco" aplica os indices
# velhos a grade nova sem levantar nada. Ou seja, "o locator resolveu" e
# "o desafio ainda existe" NAO sao prova de identidade — e por isso o guard nao
# se apoia em nenhum dos dois.

MSG_DESCARTE = "resposta descartada: desafio mudou"


def _prompt_do_desafio(page) -> str | None:
    """Enunciado normalizado do desafio ativo, ou None se ilegivel."""
    frame = _get_challenge_frame(page)
    if frame is None:
        return None
    try:
        bruto = frame.evaluate(
            "() => { const p = document.querySelector('.prompt-text'); "
            "return p ? p.textContent : ''; }"
        )
    except Exception:  # noqa: BLE001 — qualquer falha aqui e' 'desafio mudou'
        return None
    texto = " ".join(str(bruto or "").split()).lower()
    return texto or None


def _capturar_desafio(page) -> tuple[bytes | None, dict | None]:
    """Captura a regiao do desafio SEMPRE do mesmo jeito. (png, caixa).

    Ter um unico ponto de captura e o que garante que a imagem comparada e a
    imagem analisada. Duas capturas por mecanismos diferentes nunca bateriam, e
    o guard rejeitaria tudo.
    """
    loc = _get_challenge_element_locator(page)
    try:
        caixa = loc.bounding_box()
        png = loc.screenshot(timeout=8_000)
    except Exception:  # noqa: BLE001 — captura falhou = identidade indeterminada
        return None, None
    return png, caixa


def _fingerprint_desafio(page, png: bytes | None) -> str | None:
    """Identidade do desafio: enunciado normalizado + hash da captura.

    Os dois componentes se cobrem: o hCaptcha reusa o mesmo enunciado entre
    rodadas (so o texto nao distingue), e uma troca de enunciado com imagens
    parecidas passaria batida so pelos pixels.

    Hash EXATO, de proposito. Nada de perceptual hash ou limiar de similaridade
    nesta versao: um falso "mudou" custa uma recaptura; um falso "e o mesmo"
    custa um clique no desafio errado. Ver o trade-off no README.

    None significa "nao foi possivel determinar" — e indeterminado e tratado
    como mudou.
    """
    if not png:
        return None
    prompt = _prompt_do_desafio(page)
    if prompt is None:
        return None
    h = hashlib.sha256()
    h.update(prompt.encode("utf-8", "replace"))
    h.update(b"\x1f")
    h.update(png)
    return h.hexdigest()


# Fracao de pixels que precisa mudar para o desafio contar como OUTRO.
#
# O hash byte a byte era estrito demais, e isso custava caro: ESTES desafios tem
# FUNDO ANIMADO. Medido em 08/09/2026 nos quadros reais, com o desafio parado e
# nada acontecendo, 0,27% a 0,42% dos pixels mudam sozinhos entre duas capturas.
# Um hash exato nunca bate — entao o modelo respondia certo e a resposta era
# descartada como "desafio mudou", quando o que mudou foi o fundo.
#
# Registrado em producao, tres respostas boas jogadas fora em sequencia:
#
#     'flor em que a abelha pousa'                | high   | tiles=[1]
#     'flores em que a abelha nao esta pousando'  | medium | tiles=[1, 6, 8]
#     'flores onde a abelha nao esta pousada'     | high   | tiles=[1, 3, 4, 8]
#     -> resposta descartada: desafio mudou   (nas tres)
#
# O `_solve_bola` ja tinha aprendido isso e usa enunciado + geometria; os
# resolvedores estaticos ficaram para tras.
#
# 5% separa com folga os dois mundos: fundo animado da decimos de porcento, e
# troca real de desafio muda a cena inteira — medido, 42.924 px numa area de
# 651x714, que sao ~9%.
DESAFIO_MUDOU_MIN_FRACAO = 0.05


def _desafio_ainda_e_o_mesmo(page, fingerprint_origem: str | None,
                             png_origem: bytes | None = None) -> bool:
    """True SO se o desafio atual for COMPROVADAMENTE o que gerou a analise.

    Devolve False em qualquer indeterminacao: desafio ausente, enunciado
    ilegivel, recaptura falhada ou fingerprint de origem inexistente.

    A comparacao de IMAGEM tolera fundo animado (ver DESAFIO_MUDOU_MIN_FRACAO);
    o ENUNCIADO continua exigido igual, e e ele que pega a troca de rodada — o
    hCaptcha muda o texto entre uma e outra.
    """
    if not fingerprint_origem:
        return False
    if not _challenge_visible(page):
        return False
    png, _caixa = _capturar_desafio(page)
    atual = _fingerprint_desafio(page, png)
    if not atual:
        return False
    if atual == fingerprint_origem:
        return True
    # Difere no hash. Foi o fundo, ou e outro desafio?
    if not (_PIL and png and png_origem):
        return False
    try:
        a = Image.open(io.BytesIO(png_origem)).convert("RGB")
        b = Image.open(io.BytesIO(png)).convert("RGB")
        if a.size != b.size:
            return False
        dif = ImageChops.difference(a, b).convert("L")
        mudou = sum(dif.point(lambda p: 255 if p > 40 else 0).point(bool).getdata())
        fracao = mudou / (a.size[0] * a.size[1])
        if fracao < DESAFIO_MUDOU_MIN_FRACAO:
            print(f"    [captcha] Desafio é o mesmo ({fracao * 100:.2f}% de "
                  "diferença — fundo animado, não troca de desafio).")
            return True
        print(f"    [captcha] Desafio realmente mudou ({fracao * 100:.1f}% "
              "de diferença).")
        return False
    except Exception:  # noqa: BLE001
        return False


def _mesma_caixa(a: dict | None, b: dict | None,
                 tolerancia_px: float = 1.0) -> bool:
    """Geometria equivalente.

    A tolerancia cobre arredondamento de float do `bounding_box`, NAO
    similaridade: 1 px nao desloca um tile de ~130 px para o vizinho.
    """
    if not a or not b:
        return False
    try:
        return all(abs(float(a[k]) - float(b[k])) <= tolerancia_px
                   for k in ("x", "y", "width", "height"))
    except (KeyError, TypeError, ValueError):
        return False


def _geometria_estavel(page, caixa_origem: dict | None) -> bool:
    """O iframe continua onde estava quando a captura foi feita.

    Necessario so onde o clique e por PIXEL: ali um deslocamento da pagina faz
    a coordenada antiga acertar um ponto arbitrario — pior do que nao clicar.
    """
    if not caixa_origem:
        return False
    try:
        atual = _get_challenge_element_locator(page).bounding_box()
    except Exception:  # noqa: BLE001 — geometria indeterminada = nao clicar
        return False
    return _mesma_caixa(atual, caixa_origem)


# Fração de pixels que precisa mudar entre dois quadros para a área contar como
# ANIMADA. Medido: área parada dá ~51 px de diferença numa região de 651x714
# (0,011% — ruído de compressão JPEG do próprio screenshot), e a bola em
# movimento dá ~5.000 px na mesma região (1,1%).
#
# 0,3% cobria a bola e REPROVAVA a abelha. Em 08/09/2026, na run das 16:51, a
# sonda mediu 0,28% no desafio da abelha e imprimiu "estático" — abaixo do
# limiar por dois centésimos. O desafio foi para o resolvedor de quadro único,
# que não tem como responder "em qual flor ela nunca pousa": essa informação
# não existe num quadro. A mesma run mediu 0,00% numa tela genuinamente parada.
#
# Ou seja, os dois casos continuam separados por ordens de grandeza — mas o
# sinal fraco é a abelha (0,28%–0,42%), não a bola (0,90%+). O limiar foi para
# 0,12%: 10x acima do ruído medido e menos da metade do sinal mais fraco.
#
# FRAÇÃO, e não contagem absoluta: a área do desafio varia de tamanho (651x714 e
# 520x402 já foram vistos), e um limiar em pixels viraria sensibilidade
# diferente para cada tamanho.
BOLA_MOVIMENTO_MIN_FRACAO = 0.0012
BOLA_SONDA_INTERVALO_S = 0.45
# Quantas amostras a sonda tira. NAO e detalhe de performance — e o que decide
# se ela enxerga o movimento.
#
# O elemento movel PAUSA sobre cada alvo: e a mesma propriedade que obrigou o
# `_amostrar_frames_distintos` a existir na captura. Com apenas DUAS amostras a
# 0,45 s, uma janela que caia dentro de uma pausa nao ve movimento nenhum e
# conclui "estatico".
#
# Medido em producao em 08/09/2026, com o desafio "clique na flor em que a
# abelha nunca pousa": o MESMO desafio foi classificado `bola_em_movimento` uma
# vez e `grade_fused` duas — e a diferenca era so em que instante os dois
# screenshots caíram.
#
# 7 amostras x 0,45 s cobrem ~2,7 s, contra um ciclo de animacao de ~9,9 s. Sai
# CEDO na primeira deteccao, entao o caso animado quase nunca paga o custo
# inteiro; quem paga e o estatico, e so no caminho ambiguo (sem tiles).
BOLA_SONDA_AMOSTRAS = 7
BOLA_SONDA_DIF_MIN = 40         # por canal, para ignorar recompressão


# Memoria da sonda, por DESAFIO. Sem isto ela repete a janela inteira a cada
# iteracao do laco de `solve_hcaptcha` — sao ate 6 — e 7 amostras viram 42
# screenshots. Foi o que o Jean viu como "trocentos prints", e a culpa e da
# mudanca que ampliou a janela: com 2 amostras o desperdicio passava
# despercebido, com 7 nao passa.
#
# A chave e o ENUNCIADO: se o desafio mudou, o texto muda junto e a memoria se
# invalida sozinha. Se o texto nao mudou, a resposta de "isto se mexe?" tambem
# nao mudou — a natureza do desafio nao oscila entre iteracoes.
_SONDA_MEMORIA: dict = {}


def _area_do_desafio_se_move(page) -> bool:
    """Dois screenshots com intervalo curto: a área do desafio muda sozinha?

    É o sinal FÍSICO que separa a bola de qualquer outro desafio de imagem
    única. Não depende de ler o enunciado, de acertar palavra-chave nem de
    idioma — a bola é o único formato em que a cena se mexe sozinha.

    Captura por CLIP, nunca por elemento: `locator.screenshot()` espera o
    elemento ficar ESTÁVEL antes de capturar, e numa área animada isso devolve
    quase sempre o mesmo quadro — foi medido, 51 px de diferença média com a
    bola visivelmente andando na tela. `page.screenshot(clip=...,
    animations="allow")` captura pela geometria e não congela nada.

    Custa ~0,5 s e dois screenshots, e só roda no caminho ambíguo (0 tiles com
    proporção de grade). Falha de qualquer natureza devolve False: na dúvida,
    mantém a classificação que já existia.
    """
    if not _PIL:
        return False
    chave = ""
    try:
        chave = (_prompt_do_desafio(page) or "")[:120]
    except Exception:  # noqa: BLE001
        chave = ""
    if chave and chave in _SONDA_MEMORIA:
        lembrado = _SONDA_MEMORIA[chave]
        print(f"    [captcha] Sonda de movimento: "
              f"{'animado' if lembrado else 'estático'} (lembrado deste desafio).")
        return lembrado
    try:
        loc = _get_challenge_element_locator(page)
        caixa = loc.bounding_box()
        if not caixa:
            return False
        clip = {"x": caixa["x"], "y": caixa["y"],
                "width": caixa["width"], "height": caixa["height"]}
        primeira = Image.open(io.BytesIO(
            page.screenshot(clip=clip, animations="allow", timeout=4_000))
        ).convert("RGB")
        total = primeira.size[0] * primeira.size[1]
        if not total:
            return False
        maior = 0.0
        # Cada amostra e comparada com a PRIMEIRA, nao com a anterior: um
        # elemento que sai e volta ao mesmo ponto daria diferenca zero entre
        # quadros vizinhos, e movimento nenhum seria visto.
        for i in range(1, BOLA_SONDA_AMOSTRAS):
            time.sleep(BOLA_SONDA_INTERVALO_S)
            atual = Image.open(io.BytesIO(
                page.screenshot(clip=clip, animations="allow", timeout=4_000))
            ).convert("RGB")
            if atual.size != primeira.size:
                return False
            dif = ImageChops.difference(primeira, atual).convert("L")
            mudou = sum(dif.point(lambda p: 255 if p > BOLA_SONDA_DIF_MIN else 0)
                        .point(bool).getdata())
            fracao = mudou / total
            maior = max(maior, fracao)
            if fracao >= BOLA_MOVIMENTO_MIN_FRACAO:
                print(f"    [captcha] Sonda de movimento: {fracao * 100:.2f}% dos "
                      f"pixels mudaram em {i * BOLA_SONDA_INTERVALO_S:.2f}s "
                      f"(limiar {BOLA_MOVIMENTO_MIN_FRACAO * 100:.1f}%) — animado.")
                if chave:
                    _SONDA_MEMORIA[chave] = True
                return True
        print(f"    [captcha] Sonda de movimento: máximo {maior * 100:.2f}% em "
              f"{(BOLA_SONDA_AMOSTRAS - 1) * BOLA_SONDA_INTERVALO_S:.1f}s "
              f"(limiar {BOLA_MOVIMENTO_MIN_FRACAO * 100:.1f}%) — estático.")
        if chave:
            _SONDA_MEMORIA[chave] = False
        return False
    except Exception:  # noqa: BLE001 — sonda nunca derruba a classificação
        return False


# Marcas de quantidade no enunciado. Elas dizem qual MECANICA o desafio pede, e
# isso a geometria nao sabe.
_MARCAS_VARIOS = ("todas", "todos", "cada ", "quantas", "quantos",
                  "todas as imagens", "all images", "each ")
_MARCAS_UM_SO = ("clique no ", "clique na ", "clique em um", "clique em uma",
                 "selecione o ", "selecione a ", "click the ", "click on the ")


def _pede_um_clique_so(instrucao: str) -> bool:
    """O enunciado pede UM clique, ou varios?

    Esta pergunta decidia por GEOMETRIA, e decidia errado. Medido em 08/09/2026,
    o MESMO desafio "Por favor, clique na figura diferente" foi roteado de duas
    formas conforme o tamanho em que a Receita o renderizou:

        605x410  ratio 1,48  ->  imagem       ->  grade 20x20  ->  RESOLVEU
        651x714  ratio 0,91  ->  grade_fused  ->  9 tiles      ->  falhou

    `grade_fused` fatia a area em 3x3 e pergunta QUAIS tiles marcar — mecanica
    de "selecione todas as imagens com onibus". Um desafio com 5 ou 6 formas
    espalhadas em posicoes arbitrarias nao tem tiles: pedir tiles ali e a
    ferramenta errada, e ela nao acerta por sorte.

    Todos os enunciados vistos ate hoje sao de clique unico:

        Clique no animal que a bola nunca toca
        Clique na flor em que a abelha nunca pousa
        Por favor, clique no icone que quebra o padrao
        Por favor, clique na figura diferente

    A marca de VARIOS vence a de UM: "clique em todas as figuras diferentes"
    seria plural, apesar do "clique".
    """
    t = (instrucao or "").lower()
    if not t:
        return False
    if any(m in t for m in _MARCAS_VARIOS):
        return False
    return any(m in t for m in _MARCAS_UM_SO)


def _detect_challenge_type(page, timeout_ms: int = 12_000,
                           ao_falhar: str = TIPO_GRADE) -> str:
    """Detecta tipo do desafio: 'grade', 'grade_fused', 'imagem', ou 'nenhum'.

    grade       — 9+ .task separados (grade 3x3 normal)
    grade_fused — imagem única que forma grade 3x3; tiles < 9 no DOM mas
                  seleção é feita pelos 9 tiles (clique por posição pixel)
    imagem      — imagem livre para clique por coordenadas (20x20 grid)
    """
    deadline = time.time() + timeout_ms / 1000
    frame = None
    while time.time() < deadline:
        frame = _get_challenge_frame(page)
        if frame:
            break
        time.sleep(0.3)

    if frame is None:
        print("    [captcha] Nenhum desafio detectado.")
        return "nenhum"

    try:
        page.wait_for_timeout(500)  # aguarda conteúdo do iframe começar a carregar

        # Polling até 2s extra para tiles carregarem (resolve timing em grades lentas)
        count = 0
        poll_deadline = time.time() + 2.0
        while time.time() < poll_deadline:
            count = frame.locator(TASK_SEL).count()
            if count >= 9:
                break
            time.sleep(0.2)

        if count >= 9:
            print(f"    [captcha] Tipo: grade 3x3 ({count} tiles).")
            return "grade"

        # Verifica selector alternativo .task-image
        alt_count = 0
        try:
            alt_count = frame.locator(".task-image").count()
        except Exception:
            pass
        if alt_count >= 9:
            print(f"    [captcha] Tipo: grade 3x3 (task-image, {alt_count} tiles).")
            return "grade"

        # Verifica instrução ANTES de checar tiles — cartao_animal tem 0 tiles no início
        # (cartas face-down) mas a instrução já está visível no DOM
        #
        # Inicializada FORA do try: o caminho geométrico mais abaixo também a
        # imprime, e sem isto um `evaluate` que falhe deixaria a variável sem
        # ligação — NameError numa linha de log, derrubando a classificação
        # inteira por causa de um print.
        instrucao_lower = ""
        try:
            instrucao_lower = frame.evaluate("""() => {
                for (const s of ['.prompt-text', 'h2', '.header-text', '[class*="prompt"]']) {
                    const el = document.querySelector(s);
                    if (el && el.textContent.trim()) return el.textContent.trim().toLowerCase();
                }
                return '';
            }""")
            _kw_animal = ("animal" in instrucao_lower and "diferente" in instrucao_lower)
            _kw_cartao = (("cartão" in instrucao_lower or "cartao" in instrucao_lower)
                           and "diferente" in instrucao_lower)
            if _kw_animal or _kw_cartao:
                print(
                    f"    [captcha] Tipo: cartao_animal "
                    f"(instrucao: '{instrucao_lower[:60]}')."
                )
                return "cartao_animal"

            # "nunca toca"/"nunca alcanca" e especifico o bastante para nao
            # colidir com outros enunciados; "bola" sozinho NAO serve de ancora
            # — ja apareceu em variantes com bola de volei e de futebol que sao
            # outro desafio. E a mecanica ("nunca toca") que identifica este.
            #
            # Precisa vir antes do fallback geometrico logo abaixo: a area e uma
            # imagem unica e quadrada, entao ela seria classificada `grade_fused`
            # e mandada a um resolvedor que olha UM quadro — incapaz de acertar,
            # por construcao, um desafio cuja resposta so existe na sequencia.
            _kw_bola = ("bola" in instrucao_lower
                        and any(m in instrucao_lower for m in (
                            "nunca toca", "nunca alcança", "nunca alcanca",
                            "não toca", "nao toca", "nunca encosta",
                            "não encosta", "nao encosta", "jamais toca",
                            "nunca passa")))
            if _kw_bola:
                print(
                    f"    [captcha] Tipo: {TIPO_BOLA} "
                    f"(instrucao: '{instrucao_lower[:60]}')."
                )
                return TIPO_BOLA
        except Exception:
            pass

        # 1-8 elementos .task ou .task-image → grade fused
        detected = max(count, alt_count)
        if detected > 0:
            print(f"    [captcha] Tipo: grade fused ({detected} tile(s) — imagem 3x3 única).")
            return "grade_fused"

        # count == 0: verifica se área do desafio tem aspecto de grade 3x3
        try:
            bounds = frame.evaluate("""() => {
                const promptSels = ['.prompt-text', '.challenge-header', 'h2',
                                    '.header-text', '[class*="prompt"]'];
                let imgTop = 0;
                for (const sel of promptSels) {
                    const el = document.querySelector(sel);
                    if (el) { const b = el.getBoundingClientRect().bottom; if (b > imgTop) imgTop = b; }
                }
                const btnSels = ['.button-submit', '.button-verify', '[class*="submit"]'];
                let imgBottom = document.documentElement.clientHeight;
                for (const sel of btnSels) {
                    const el = document.querySelector(sel);
                    if (el) { const t = el.getBoundingClientRect().top; if (t < imgBottom) imgBottom = t; }
                }
                const w = document.documentElement.clientWidth;
                const h = imgBottom - imgTop;
                if (h < 50 || w < 50) return null;
                return {width: w, height: h};
            }""")
            if bounds:
                ratio = bounds["width"] / bounds["height"]
                # A ÁREA SE MEXE? Esta pergunta vem antes de qualquer geometria,
                # e vale para QUALQUER proporção.
                #
                # Movimento é a única propriedade que separa "a resposta está
                # neste quadro" de "a resposta está na sequência", e ela não
                # depende de ler o enunciado, de acertar palavra-chave nem de
                # idioma. Enquanto a sonda vivia só dentro da faixa 0,75–1,40,
                # ela pegava a bola (quadrada) e perdia formatos mais largos: o
                # "clique na flor em que a abelha nunca pousa" apareceu em
                # 08/09/2026 numa área de razão ~1,48 e teria ido para um
                # resolvedor que olha UM quadro.
                #
                # Custa ~0,5 s e dois screenshots, só quando não há tiles — ou
                # seja, só no caminho ambíguo, onde a geometria decidiria
                # sozinha e podia decidir errado.
                if _area_do_desafio_se_move(page):
                    print(f"    [captcha] Tipo: {TIPO_BOLA} (área animada, "
                          f"{bounds['width']:.0f}x{bounds['height']:.0f}px "
                          f"ratio={ratio:.2f}).")
                    return TIPO_BOLA
                # Grade 3x3 é aproximadamente quadrada (0.75–1.4); imagem livre é mais retangular
                # O ENUNCIADO decide a MECANICA; a proporcao so desempata.
                #
                # Sem isto, um desafio de clique unico renderizado quase
                # quadrado ia para o resolvedor de tiles, que pergunta "quais
                # marcar" — e nao ha tiles para marcar.
                if _pede_um_clique_so(instrucao_lower):
                    print(f"    [captcha] Tipo: imagem (enunciado pede UM "
                          f"clique, {bounds['width']:.0f}x{bounds['height']:.0f}px "
                          f"ratio={ratio:.2f}, instrucao: "
                          f"'{(instrucao_lower or '')[:60]}').")
                    return "imagem"
                if 0.75 <= ratio <= 1.4:
                    # O ENUNCIADO entra no log AQUI, e não só nos tipos
                    # reconhecidos por palavra-chave.
                    #
                    # Levantamento do histórico de dev: todo tipo classificado
                    # por texto registra a instrução; os dois classificados por
                    # GEOMETRIA (grade e grade_fused) não registram nada. Ou
                    # seja, justamente quando a classificação é incerta, o dado
                    # que resolveria a dúvida é o único que falta — e
                    # `grade_fused` tem 116 ocorrências. Sem isto, "era bola
                    # classificada errado ou outro formato?" é indecidível
                    # depois do fato, para sempre.
                    print(
                        f"    [captcha] Tipo: grade fused (0 tiles, "
                        f"{bounds['width']:.0f}x{bounds['height']:.0f}px "
                        f"ratio={ratio:.2f}, instrucao: "
                        f"'{(instrucao_lower or '')[:60]}')."
                    )
                    return "grade_fused"
        except Exception:
            pass

        print(f"    [captcha] Tipo: imagem completa ({count} tile(s)).")
        return "imagem"
    except Exception as e:
        # `ao_falhar` existe para separar dois usos com necessidades opostas:
        # `solve_hcaptcha` prefere chutar `grade` e tentar; quem decide POLÍTICA
        # precisa saber que não deu para classificar.
        print(f"    [captcha] Erro ao detectar tipo: {type(e).__name__}. "
              f"Assumindo {ao_falhar}.")
        return ao_falhar


# ──────────────────────────────────────────────────────────────────────────────
# Screenshot helpers
# ──────────────────────────────────────────────────────────────────────────────

def _wait_for_tiles(page) -> bool:
    """Aguarda tiles visíveis E imagens carregadas dentro dos tiles."""
    if not _challenge_visible(page):
        return False   # challenge já sumiu — não há tiles para esperar
    cf = _get_challenge_frame_locator(page)
    try:
        cf.locator(TASK_SEL).nth(8).wait_for(state="visible", timeout=6_000)
    except Exception:
        print("    [captcha] Timeout aguardando tiles.")
        return False

    # Verifica via JS se as imagens dos tiles carregaram (CSS bg ou img.complete)
    deadline = time.time() + 2.0
    while time.time() < deadline:
        try:
            frame = _get_challenge_frame(page)
            if not frame:
                break
            ready = frame.evaluate("""() => {
                const tiles = document.querySelectorAll('.task-image');
                if (tiles.length < 9) return false;
                for (const t of tiles) {
                    const bg = window.getComputedStyle(t).backgroundImage;
                    const img = t.querySelector('img');
                    if (!(bg && bg !== 'none' && bg.includes('url(')) &&
                        !(img && img.complete && img.naturalWidth > 0)) return false;
                }
                return true;
            }""")
            if ready:
                return True
        except Exception:
            pass
        time.sleep(0.08)
    return True


def _get_reference_image_bytes(page) -> Optional[bytes]:
    """Extrai a imagem de referência do prompt do captcha, quando presente.

    Usa frame.locator (Frame real) em vez de FrameLocator, e is_visible() em
    vez de count() > 0 — garante que a imagem está realmente renderizada antes
    de capturar o screenshot (bug crítico na versão anterior com FrameLocator).
    """
    try:
        frame = _get_challenge_frame(page)
        if not frame:
            return None
        idx = frame.evaluate("""() => {
            const taskSrcs = new Set(
                [...document.querySelectorAll('.task img, .task-image img')].map(i => i.src)
            );
            const all = [...document.querySelectorAll('img')];
            for (let i = 0; i < all.length; i++) {
                const img = all[i];
                if (!taskSrcs.has(img.src) && img.complete && img.naturalWidth > 40) return i;
            }
            return -1;
        }""")
        if idx < 0:
            return None
        # Frame real (não FrameLocator) — mesmo índice que o JS acima, garantido
        ref_loc = frame.locator("img").nth(idx)
        if ref_loc.is_visible(timeout=1_000):
            png = ref_loc.screenshot()
            print("    [captcha] Imagem de referência extraída separadamente.")
            return png
    except Exception:
        pass
    return None


def _get_task_image_screenshot_and_bbox(
    page,
) -> tuple[Optional[bytes], Optional[dict]]:
    """Screenshot da área de imagem do desafio (sem cabeçalho/rodapé).

    3 estratégias em cascata:
      1. Seletores CSS conhecidos para elemento <img>
      2. JS — maior <img> com área >= 150x150
      3. DOM bounds (borda inferior do prompt → borda superior do submit)
         capturado via page.screenshot(clip=...) — funciona com CSS background.

    Returns (png_bytes, page_bbox) ou (None, None).
    """
    frame = _get_challenge_frame(page)
    if not frame:
        return None, None

    iframe_loc = _get_challenge_element_locator(page)
    iframe_box = iframe_loc.bounding_box()
    if not iframe_box:
        return None, None

    cf_fl = _get_challenge_frame_locator(page)

    def build_page_bbox(box: dict) -> dict:
        return {
            "x":      iframe_box["x"] + box["x"],
            "y":      iframe_box["y"] + box["y"],
            "width":  box["width"],
            "height": box["height"],
        }

    # Estratégia 1: seletores CSS conhecidos
    for sel in [
        ".task-image img", "img.task-image",
        ".challenge-image img", "img.challenge-image",
        ".task img", ".challenge-container img",
        "img[src*='hcaptcha']", "img[src*='hmt-']",
    ]:
        try:
            img_box = frame.locator(sel).first.bounding_box(timeout=800)
            if not img_box or img_box["width"] < 80 or img_box["height"] < 80:
                continue
            png = cf_fl.locator(sel).first.screenshot()
            if png:
                print(f"    [captcha/imagem] Via CSS '{sel}': {img_box['width']:.0f}x{img_box['height']:.0f}px")
                return png, build_page_bbox(img_box)
        except Exception:
            continue

    # Estratégia 2: JS — maior <img> com área >= 150x150
    try:
        js_img = frame.evaluate("""() => {
            const imgs = [...document.querySelectorAll('img')];
            let best = null, bestArea = 0;
            for (let i = 0; i < imgs.length; i++) {
                const r = imgs[i].getBoundingClientRect();
                const a = r.width * r.height;
                if (a > bestArea && r.width >= 150 && r.height >= 150) {
                    bestArea = a;
                    best = {index: i, x: r.x, y: r.y, width: r.width, height: r.height};
                }
            }
            return best;
        }""")
        if js_img:
            img_box = {k: js_img[k] for k in ("x", "y", "width", "height")}
            png = cf_fl.locator("img").nth(js_img["index"]).screenshot()
            if png:
                print(f"    [captcha/imagem] Via JS img[{js_img['index']}]: {img_box['width']:.0f}x{img_box['height']:.0f}px")
                return png, build_page_bbox(img_box)
    except Exception as e:
        print(f"    [captcha/imagem] JS img fallback: {type(e).__name__}")

    # Estratégia 3: DOM bounds (prompt.bottom → submit.top) + page.screenshot(clip)
    try:
        bounds = frame.evaluate("""() => {
            const promptSels = ['.prompt-text', '.challenge-header', 'h2', '.header-text',
                                '.task-label', '[class*="prompt"]', '[class*="label"]'];
            let imgTop = 0;
            for (const sel of promptSels) {
                const el = document.querySelector(sel);
                if (el) { const b = el.getBoundingClientRect().bottom; if (b > imgTop) imgTop = b; }
            }
            const btnSels = ['.button-submit', '.button-verify', '[class*="submit"]', '[class*="verify"]'];
            let imgBottom = document.documentElement.clientHeight;
            for (const sel of btnSels) {
                const el = document.querySelector(sel);
                if (el) { const t = el.getBoundingClientRect().top; if (t < imgBottom) imgBottom = t; }
            }
            const w = document.documentElement.clientWidth;
            const h = imgBottom - imgTop;
            if (imgTop < 10 || h < 50 || w < 50) return null;
            return {x: 0, y: imgTop, width: w, height: h};
        }""")
        if bounds:
            page_bbox = build_page_bbox(bounds)
            png = page.screenshot(clip=page_bbox)
            if png:
                print(f"    [captcha/imagem] Via DOM bounds: {bounds['width']:.0f}x{bounds['height']:.0f}px")
                return png, page_bbox
    except Exception as e:
        print(f"    [captcha/imagem] DOM bounds falhou: {type(e).__name__}")

    return None, None


def _extrair_instrucao(page) -> str:
    """Lê o texto de instrução do iframe do desafio."""
    try:
        frame = _get_challenge_frame(page)
        if not frame:
            return ""
        for sel in [
            ".prompt-text", ".challenge-prompt", ".task-instructions",
            "h2", ".header-text", "[class*='prompt']", "[class*='instruction']",
        ]:
            try:
                txt = frame.locator(sel).first.inner_text(timeout=500).strip()
                if txt:
                    return txt
            except Exception:
                continue
    except Exception:
        pass
    return ""


# ──────────────────────────────────────────────────────────────────────────────
# Grid 20x20 overlay (PIL)
# ──────────────────────────────────────────────────────────────────────────────

def _overlay_grid(png: bytes, cols: int = GRID_COLS, rows: int = GRID_ROWS) -> bytes:
    """Desenha grid col x row com rótulos 'col,row' + sombra sobre o PNG."""
    if not _PIL:
        return png
    try:
        img = Image.open(io.BytesIO(png)).convert("RGB")
        w, h = img.size
        cw, ch = w / cols, h / rows

        # Linhas via alpha_composite (semitransparente)
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        for c in range(1, cols):
            x = int(c * cw)
            od.line([(x, 0), (x, h)], fill=(220, 30, 30, 180), width=1)
        for r in range(1, rows):
            y = int(r * ch)
            od.line([(0, y), (w, y)], fill=(220, 30, 30, 180), width=1)
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(img)

        # Fonte — tenta Arial, fallback bitmap
        sz = max(8, int(min(cw, ch) * 0.35))
        try:
            font = ImageFont.truetype("arial.ttf", size=sz)
        except Exception:
            font = ImageFont.load_default()

        # Rótulos com sombra preta para legibilidade
        for r in range(rows):
            for c in range(cols):
                lx = int(c * cw) + 2
                ly = int(r * ch) + 1
                label = f"{c},{r}"
                draw.text((lx + 1, ly + 1), label, fill=(0, 0, 0), font=font)
                draw.text((lx, ly), label, fill=(255, 255, 0), font=font)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        print(f"    [captcha] Erro no grid overlay: {type(e).__name__}")
        return png


def _overlay_3x3_grid(png: bytes) -> bytes:
    """Desenha grid 3×3 com índices 0-8 no centro de cada tile sobre o PNG."""
    if not _PIL:
        return png
    try:
        img = Image.open(io.BytesIO(png)).convert("RGB")
        w, h = img.size
        tw, th = w / 3, h / 3

        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        for c in range(1, 3):
            x = int(c * tw)
            od.line([(x, 0), (x, h)], fill=(220, 30, 30, 230), width=3)
        for r in range(1, 3):
            y = int(r * th)
            od.line([(0, y), (w, y)], fill=(220, 30, 30, 230), width=3)
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(img)

        sz = max(14, int(min(tw, th) * 0.28))
        try:
            font = ImageFont.truetype("arial.ttf", size=sz)
        except Exception:
            font = ImageFont.load_default()

        for r in range(3):
            for c in range(3):
                idx = r * 3 + c
                cx = int((c + 0.5) * tw)
                cy = int((r + 0.5) * th)
                label = str(idx)
                for dx, dy in [(-2,-2),(2,-2),(-2,2),(2,2),(0,-2),(0,2),(-2,0),(2,0)]:
                    try:
                        draw.text((cx + dx, cy + dy), label, fill=(0, 0, 0), font=font, anchor="mm")
                    except Exception:
                        draw.text((cx + dx, cy + dy), label, fill=(0, 0, 0), font=font)
                try:
                    draw.text((cx, cy), label, fill=(255, 255, 0), font=font, anchor="mm")
                except Exception:
                    draw.text((cx, cy), label, fill=(255, 255, 0), font=font)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        print(f"    [captcha] Erro no 3x3 overlay: {type(e).__name__}")
        return png


# ──────────────────────────────────────────────────────────────────────────────
# Redução de payload enviado ao Gemini
# ──────────────────────────────────────────────────────────────────────────────

def _png_dims(png: bytes) -> tuple[int, int]:
    """Retorna (largura, altura) do PNG sem depender do PIL (lê o header IHDR)."""
    try:
        if png and len(png) >= 24 and png[12:16] == b"IHDR":
            w = int.from_bytes(png[16:20], "big")
            h = int.from_bytes(png[20:24], "big")
            return w, h
    except Exception:
        pass
    return 0, 0


def _shrink_png(png: bytes, max_dim: int = 900) -> bytes:
    """Reduz o PNG para no máximo `max_dim` px no maior lado antes de enviar ao Gemini.

    O Chrome abre maximizado (no_viewport) na resolução do monitor com o scaling do
    Windows, então os screenshots do iframe do hCaptcha saem bem maiores do que o
    necessário — e, se a detecção do iframe ativo cair no fallback, pode capturar um
    iframe do tamanho do viewport quase todo em branco. Enviar essa imagem cheia
    deixa a chamada ao Gemini lenta sem ganho de acurácia (o desafio é pequeno).
    Mantém a proporção; se já couber, ou se o PIL não estiver disponível, devolve o
    PNG original inalterado.

    Seguro para os cliques: a matemática de clique usa a bounding box da PÁGINA
    (não os pixels da imagem enviada), então reduzir a imagem não afeta as posições.
    """
    if not _PIL or not png:
        return png
    try:
        img = Image.open(io.BytesIO(png))
        w, h = img.size
        if max(w, h) <= max_dim:
            return png
        scale = max_dim / max(w, h)
        img = img.convert("RGB").resize(
            (max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS
        )
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()
    except Exception:
        return png


QUALIDADE_JPEG = 85


def _para_envio(img: bytes, max_dim: int = 900) -> tuple[bytes, str]:
    """Bytes prontos para o Gemini + o mime_type correspondente.

    O screenshot nasce PNG, e PNG e o formato errado para o que ele carrega: os
    tiles do hCaptcha sao FOTOS. Um recorte de 459x689 saia com 291 KB, e cada
    chamada subia isso antes de o modelo comecar a pensar. Na execucao de
    27/08/2026 foram 22 `ReadTimeout` de 20s — cerca de sete minutos, um quarto
    da execucao, esperando requisicoes que nao voltaram.

    Em JPEG a mesma imagem fica na casa das dezenas de KB. A perda e irrelevante
    para a tarefa (dizer se o tile tem um coelho), e o upload deixa de ser o
    gargalo.

    O mime_type volta junto porque a resposta e condicional: sem PIL nao ha
    conversao, e ai o que sobe e o PNG original — declarar "image/jpeg" para
    bytes de PNG quebraria a chamada.
    """
    img = _shrink_png(img, max_dim=max_dim)
    if not _PIL or not img:
        return img, "image/png"
    try:
        foto = Image.open(io.BytesIO(img)).convert("RGB")
        buf = io.BytesIO()
        foto.save(buf, format="JPEG", quality=QUALIDADE_JPEG, optimize=True)
        return buf.getvalue(), "image/jpeg"
    except Exception:
        return img, "image/png"


def _parte_imagem(img: bytes, max_dim: int = 900):
    """A imagem como Part do Gemini, ja no formato e no tamanho de envio."""
    dados, mime = _para_envio(img, max_dim=max_dim)
    return _gt.Part.from_bytes(data=dados, mime_type=mime)


def _limpar_texto(valor, max_len: int = 200) -> str:
    """Colapsa qualquer sequência de espaços/quebras de linha em um único espaço e
    trunca o resultado.

    Campos de texto livre devolvidos pelo Gemini (task_summary, instruction, etc.)
    às vezes vêm degenerados — centenas de '\\n' — quando o modelo recebe uma imagem
    ruim/enorme. Imprimir esse valor cru inunda o console com linhas em branco. Esta
    função garante que qualquer print de texto do modelo caiba em uma única linha.
    """
    s = " ".join(str(valor or "").split())
    return s if len(s) <= max_len else s[:max_len] + "…"


# ──────────────────────────────────────────────────────────────────────────────
# Chamadas ao Gemini
# ──────────────────────────────────────────────────────────────────────────────

def _gemini_grade(png: bytes, ref_img: Optional[bytes], api_key: str,
                  politica: PoliticaLatencia | None = None,
                  rodizio: int = 0, direto_ao_segundo: bool = False) -> dict:
    """Grade 3x3 → Gemini → {task_summary, matching_tiles, confidence}."""
    if ref_img:
        contents = [
            _parte_imagem(png),
            _parte_imagem(ref_img, max_dim=512),
            _PROMPT_GRADE_COM_REF,
        ]
    else:
        contents = [
            _parte_imagem(png),
            _PROMPT_GRADE,
        ]
    return _gemini_call(contents, _SCHEMA_GRADE, api_key, "grade", politica,
                        rodizio=rodizio,
                        direto_ao_segundo=direto_ao_segundo)


_PROMPT_GRADE_FUSED = """\
Voce esta resolvendo um hCaptcha especial: os 9 tiles estao fundidos em UMA UNICA imagem.

=== IMAGENS RECEBIDAS ===
  IMAGEM 1 — Screenshot completo do iframe do desafio (cabecalho colorido com o enunciado + area dos tiles).
  IMAGEM 2 — SOMENTE a area dos tiles recortada, com grid VERMELHO 3x3 desenhado sobre ela.
             Cada celula esta numerada de 0 a 8 no CENTRO:
             ┌───┬───┬───┐
             │ 0 │ 1 │ 2 │  linha superior
             ├───┼───┼───┤
             │ 3 │ 4 │ 5 │  linha do meio
             ├───┼───┼───┤
             │ 6 │ 7 │ 8 │  linha inferior
             └───┴───┴───┘

=== PASSO 1 — LEIA O ENUNCIADO ===
Leia o texto do cabecalho da IMAGEM 1 com atencao total. Ha dois tipos:

TIPO A — Enunciado direto (ex.: "Selecione todos os onibus", "Click on cars"):
  → Procure exatamente o objeto mencionado.

TIPO B — Categoria com imagem de referencia (ex.: "Selecione a mesma categoria que a imagem de referencia"):
  → Identifique o objeto na imagem de referencia do cabecalho.
  → Determine a CATEGORIA AMPLA e inclua TODOS os objetos dela.

TABELA DE CATEGORIAS (Tipo B):
  aviao, helicoptero, foguete, drone      → "veiculos aereos / transportes"
  carro, trem, onibus, caminhao, barco    → "veiculos / transportes"
  qualquer veiculo (aereo/terrestre/agua) → "transportes / veiculos"
  cachorro, gato, coelho, passaro, peixe  → "animais"
  rosa, girassol, tulipa, planta, arvore  → "flores / plantas / natureza"
  hamburguer, pizza, fruta, comida        → "alimentos / comida"
  celular, laptop, tablet, televisao      → "eletronicos / tecnologia"
  casa, predio, ponte, monumento          → "construcoes / arquitetura"

=== PASSO 2 — ANALISE CADA TILE NA IMAGEM 2 ===
Para cada celula (0-8), identifique o objeto principal e verifique se atende ao criterio.
Use a IMAGEM 2 (com grid numerico) para determinar com precisao qual tile contem o que.
Em caso de duvida razoavel: INCLUA o tile.

=== REGRAS CRITICAS ===
  !! Lista vazia [] e QUASE SEMPRE ERRADA — o hCaptcha garante pelo menos 2 tiles corretos.
  !! Tipicamente 2 a 5 tiles correspondem ao criterio por rodada.
  !! Se voce retornou [] antes, AMPLIE a categoria e seja mais generoso.
  !! Se referencia = aviao: inclua TRENS, ONIBUS, CARROS, BARCOS — todos sao transportes.

=== RETORNE ===
  task_summary: criterio identificado (categoria ampla para Tipo B)
  matching_tiles: lista de indices 0-8 (NUNCA vazia)
  confidence: "high" | "medium" | "low"
"""


def _gemini_grade_fused(iframe_png: bytes, tiles_png: bytes, api_key: str,
                        politica: PoliticaLatencia | None = None) -> dict:
    """Grade fused: envia iframe completo (contexto) + tiles recortados com overlay 3x3 → Gemini."""
    contents = [
        _parte_imagem(iframe_png),
        _parte_imagem(tiles_png),
        _PROMPT_GRADE_FUSED,
    ]
    return _gemini_call(contents, _SCHEMA_GRADE, api_key, "grade_fused", politica)


def _gemini_grid(png: bytes, instrucao: str, api_key: str,
                 politica: PoliticaLatencia | None = None,
                 rodizio: int = 0) -> dict:
    """Imagem+grid → Gemini → {instruction, action, click_positions, confidence}."""
    prompt = _PROMPT_GRID_TMPL.format(
        cols=GRID_COLS,
        rows=GRID_ROWS,
        max_col=GRID_COLS - 1,
        max_row=GRID_ROWS - 1,
        instruction=instrucao or "Leia a instrucao que aparece na imagem.",
    )
    contents = [
        _parte_imagem(png),
        prompt,
    ]
    return _gemini_call(contents, _SCHEMA_GRID, api_key, "grid", politica,
                        rodizio=rodizio)


# ──────────────────────────────────────────────────────────────────────────────
# Execução de cliques
# ──────────────────────────────────────────────────────────────────────────────

def _click_grade_tiles(page, indices: list[int]) -> None:
    """Clica nos tiles por índice 0-8 diretamente no DOM do frame ativo."""
    if not indices:
        return
    cf = _get_challenge_frame_locator(page)
    tasks = cf.locator(TASK_SEL)
    # Ordem embaralhada, intervalo variavel, `delay` variavel.
    #
    # `sorted()` clicava sempre em ordem crescente de indice, com 30ms fixos e
    # 50ms de pausa: tres constantes numa sequencia que um humano nunca
    # produz. A ordem em que alguem marca os quadrados nao e a ordem do DOM.
    alvos = list(set(indices))
    random.shuffle(alvos)
    for idx in alvos:
        try:
            tasks.nth(idx).click(delay=random.randint(40, 110))
            time.sleep(random.uniform(0.18, 0.55))
            print(f"    [captcha] Tile {idx} clicado.")
        except Exception as e:
            print(f"    [captcha] Erro ao clicar tile {idx}: {type(e).__name__}")


def _click_fused_grade_tiles(page, indices: list[int],
                              bbox: Optional[dict] = None) -> None:
    """Clica nos tiles de grade 3x3 fundida por posição pixel.

    Se `bbox` (coordenadas de página da área dos tiles) for fornecido, usa
    diretamente. Caso contrário, calcula via DOM bounds (prompt.bottom → submit.top).
    """
    if not indices:
        return

    grid_bbox = bbox

    if grid_bbox is None:
        frame = _get_challenge_frame(page)
        iframe_loc = _get_challenge_element_locator(page)
        iframe_box = iframe_loc.bounding_box()
        if not iframe_box or not frame:
            return

        try:
            bounds = frame.evaluate("""() => {
                const promptSels = ['.prompt-text', '.challenge-header', 'h2',
                                    '.header-text', '.task-label',
                                    '[class*="prompt"]', '[class*="label"]'];
                let imgTop = 0;
                for (const sel of promptSels) {
                    const el = document.querySelector(sel);
                    if (el) { const b = el.getBoundingClientRect().bottom; if (b > imgTop) imgTop = b; }
                }
                const btnSels = ['.button-submit', '.button-verify',
                                 '[class*="submit"]', '[class*="verify"]'];
                let imgBottom = document.documentElement.clientHeight;
                for (const sel of btnSels) {
                    const el = document.querySelector(sel);
                    if (el) { const t = el.getBoundingClientRect().top; if (t < imgBottom) imgBottom = t; }
                }
                const w = document.documentElement.clientWidth;
                const h = imgBottom - imgTop;
                if (imgTop < 10 || h < 50 || w < 50) return null;
                return {x: 0, y: imgTop, width: w, height: h};
            }""")
            if bounds:
                grid_bbox = {
                    "x":      iframe_box["x"] + bounds["x"],
                    "y":      iframe_box["y"] + bounds["y"],
                    "width":  bounds["width"],
                    "height": bounds["height"],
                }
        except Exception:
            pass

        if grid_bbox is None:
            grid_bbox = {
                "x": iframe_box["x"], "y": iframe_box["y"],
                "width": iframe_box["width"], "height": iframe_box["height"],
            }

    tile_w = grid_bbox["width"]  / 3
    tile_h = grid_bbox["height"] / 3

    _mover_cursor_suave(1)
    for idx in sorted(set(indices)):
        row = idx // 3
        col = idx % 3
        x = grid_bbox["x"] + (col + 0.5) * tile_w
        y = grid_bbox["y"] + (row + 0.5) * tile_h
        try:
            page.mouse.click(x, y)
            time.sleep(0.08)
            print(f"    [captcha] Tile fused {idx} (row={row},col={col}) -> ({x:.0f},{y:.0f})")
        except Exception as e:
            print(f"    [captcha] Erro ao clicar tile fused {idx}: {type(e).__name__}")


ESQUEMA_PIXEL = {
    "type": "object",
    "properties": {
        "x": {"type": "integer", "description": "Coluna do pixel, 0 = borda esquerda."},
        "y": {"type": "integer", "description": "Linha do pixel, 0 = borda de cima."},
        "description": {"type": "string", "description": "Que figura e essa, e por que ela destoa."},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["x", "y", "confidence"],
}


def _gemini_pixel(png: bytes, instrucao: str, api_key: str,
                  politica: PoliticaLatencia | None = None, rodizio: int = 0) -> dict:
    """Pergunta o CENTRO da figura em pixels, sem malha desenhada por cima.

    Medido em 09/09/2026 contra as amostras arquivadas de "figura diferente",
    com as respostas marcadas na imagem para conferencia visual:

        com malha 20x20   caiu na agua vazia entre duas pipas; 10-26s
        pixel direto      caiu em cima da pipa de painel verde, a certa; 2-5s

    Nas duas amostras com gabarito verificado a malha errou e o pixel acertou.
    Faz sentido: a malha existia para dar ao modelo um vocabulario de posicao,
    mas ela DESENHA linhas e numeros sobre uma imagem que ja e camuflagem
    deliberada — soma ruido ao problema que o modelo tem de resolver, e o
    formato "figura diferente" e justamente o que depende de ver a forma.

    A grade 3x3 continua com `_gemini_grid`: la os tiles ja sao celulas de
    verdade, entao a malha nao inventa nada.
    """
    largura, altura = _dimensoes_png(png)
    prompt = (
        f'Instrução do captcha: "{_limpar_texto(instrucao)}". '
        f"A imagem tem {largura}x{altura} pixels. Compare as figuras ENTRE SI "
        f"e escolha a que destoa das demais. "
        f"Responda o CENTRO dela em pixels da imagem: x de 0 a {largura - 1}, "
        f"y de 0 a {altura - 1}, com 0,0 no canto superior esquerdo. "
        f"Um ponto só — o enunciado pede um clique."
    )
    return _gemini_call([_parte_imagem(png), prompt], ESQUEMA_PIXEL, api_key,
                        "imagem", politica, rodizio=rodizio)


def _dimensoes_png(png: bytes) -> tuple[int, int]:
    """(largura, altura) da imagem, para o prompt e para a conversao do clique."""
    from PIL import Image
    with Image.open(io.BytesIO(png)) as im:
        return im.size


def _aproximar_do_alvo(page, x: float, y: float) -> None:
    """Leva o ponteiro ate (x, y) por uma TRAJETORIA, e nao por um salto.

    `page.mouse.click(x, y)` emite um unico `mousemove` ja no destino: o
    ponteiro nunca esteve em outro lugar. Nenhum humano produz isso, e a
    sequencia de eventos e exatamente o que os antibot amostram.

    `_mover_cursor_suave`, que ja existia e continua sendo chamado, mexe no
    cursor do SISTEMA por `user32`. A pagina nao ve o cursor do sistema — ela
    ve os eventos que o Playwright injeta. Sao coisas diferentes, e so esta
    aqui chega ate ela.

    Duas etapas de proposito: um ponto intermediario deslocado do alvo e
    depois a aproximacao final. Movimento humano tem correcao de rota; reta
    perfeita ate o pixel exato tambem e assinatura.

    Nao levanta: se a movimentacao falhar, o clique de quem chama continua
    valendo — isto e disfarce, nao funcionalidade.
    """
    try:
        desvio_x = random.uniform(-70, 70)
        desvio_y = random.uniform(-55, 55)
        page.mouse.move(x + desvio_x, y + desvio_y,
                        steps=random.randint(12, 22))
        page.wait_for_timeout(random.randint(40, 130))
        page.mouse.move(x, y, steps=random.randint(5, 11))
        page.wait_for_timeout(random.randint(30, 90))
    except Exception:  # noqa: BLE001
        pass


def _click_pixel(page, ponto: dict, bbox: dict, tamanho: tuple[int, int]) -> None:
    """Converte pixel-da-imagem em pixel-do-viewport e clica.

    Passa por FRACAO de propósito: o screenshot pode sair em escala diferente
    do bbox (devicePixelRatio), e converter direto somaria um erro silencioso
    de posicao — exatamente o tipo de defeito que a malha escondia, porque ela
    ja trabalhava em proporcao.
    """
    larg, alt = tamanho
    fx = max(0.0, min(1.0, ponto["x"] / max(1, larg)))
    fy = max(0.0, min(1.0, ponto["y"] / max(1, alt)))
    x = bbox["x"] + fx * bbox["width"]
    y = bbox["y"] + fy * bbox["height"]
    _mover_cursor_suave(1)
    _aproximar_do_alvo(page, x, y)
    try:
        # `delay` entre pressionar e soltar: clique humano nao e instantaneo.
        page.mouse.click(x, y, delay=random.randint(45, 120))
        print(f"    [captcha] Pixel ({ponto['x']},{ponto['y']}) -> ({x:.0f},{y:.0f}) "
              f"| {ponto.get('description', '')}")
    except Exception as e:  # noqa: BLE001
        print(f"    [captcha] Erro ao clicar pixel: {type(e).__name__}")


def _click_grid_positions(page, positions: list[dict], bbox: dict) -> None:
    """Converte col/row → pixels viewport e clica."""
    if not positions:
        return
    _mover_cursor_suave(1)
    cw = bbox["width"]  / GRID_COLS
    ch = bbox["height"] / GRID_ROWS
    for pos in positions:
        col = max(0, min(GRID_COLS - 1, int(pos.get("col", 10))))
        row = max(0, min(GRID_ROWS - 1, int(pos.get("row", 10))))
        x = bbox["x"] + (col + 0.5) * cw
        y = bbox["y"] + (row + 0.5) * ch
        try:
            page.mouse.click(x, y)
            time.sleep(0.12)
            print(f"    [captcha] Grid ({col},{row}) -> ({x:.0f},{y:.0f}) | {pos.get('description', '')}")
        except Exception as e:
            print(f"    [captcha] Erro ao clicar grid ({col},{row}): {type(e).__name__}")


# ──────────────────────────────────────────────────────────────────────────────
# Submit com 5 estratégias em cascata
# ──────────────────────────────────────────────────────────────────────────────

# Quantas vezes NOS submetemos um desafio neste processo.
#
# Existe para responder uma pergunta que o log nao respondia: quando o desafio
# some, fomos nos ou foi outra pessoa? Os resolvedores checam
# `_challenge_visible` no inicio de cada rodada e, se o desafio sumiu, devolvem
# True — "sumiu" e "resolvi" viravam a MESMA linha. Numa run acompanhada por
# alguem, isso torna todo sucesso ambiguo: em 08/09/2026 o Jean resolveu um
# captcha a mao e o log registrou "Captcha resolvido na iteracao 1!", e eu li
# como resolucao automatica.
_SUBMISSOES = 0


def _sumiu(tag: str, marca: int, detalhe: str = "") -> bool:
    """Loga o desaparecimento do desafio dizendo SE foi obra nossa.

    `marca` e o valor de `_SUBMISSOES` na entrada do resolvedor. Se ele nao
    mudou, nao houve submissao nossa nesta chamada — entao o desafio sumiu por
    outro motivo: alguem resolveu na tela, ou ele expirou.

    Devolve True porque o desfecho FUNCIONAL e o mesmo dos dois lados (nao ha
    mais desafio); o que muda e o que o log afirma.
    """
    onde = f" {detalhe}" if detalhe else ""
    if _SUBMISSOES == marca:
        print(f"    [captcha/{tag}] Desafio sumiu{onde} SEM submissão nossa — "
              "resolvido fora da automação (pessoa na tela, ou expirou).")
    else:
        print(f"    [captcha/{tag}] Desafio sumiu{onde} — resolvido!")
    return True


def _submit_captcha(page) -> bool:
    """Clica no botão Verificar — 5 estratégias em cascata."""
    global _SUBMISSOES
    _SUBMISSOES += 1
    page.wait_for_timeout(300)
    print("    [captcha] Submetendo desafio...")

    # 1. CLIQUE REAL no frame — era a estrategia 2, e nunca era alcancada.
    #
    # O JavaScript vinha primeiro e sempre vencia, entao todo submit da
    # producao saia como `btn.click()` disparado por `evaluate`. Um clique
    # sintetico de JS chega na pagina sem `isTrusted`, sem mousedown/mouseup,
    # sem coordenada e sem o mousemove que o antecede — e o hCaptcha amostra
    # exatamente esses eventos. O log dizia isso em toda submissao:
    #
    #     [captcha] Submit via JS/frame real.
    #
    # A estrategia de clique real ja existia e ja funcionava; so estava atras
    # na fila. Inverter nao adiciona risco: o JS continua logo abaixo, como
    # rede, para o caso de o botao nao ser clicavel pelo locator.
    try:
        frame = _get_challenge_frame(page)
        if frame:
            for sel in SUBMIT_SELS + ['[role="button"][title*="ximo"]']:
                try:
                    alvo = frame.locator(sel).first
                    try:
                        caixa = alvo.bounding_box()
                        if caixa:
                            _aproximar_do_alvo(
                                page,
                                caixa["x"] + caixa["width"] / 2,
                                caixa["y"] + caixa["height"] / 2)
                    except Exception:  # noqa: BLE001 — disfarce nao derruba
                        pass
                    alvo.click(timeout=2_000, delay=random.randint(45, 120))
                    print(f"    [captcha] Submit via clique real ({sel}).")
                    return True
                except Exception:
                    continue
    except Exception:
        pass

    # 2. JavaScript no frame real — rede de seguranca, nao o caminho normal.
    try:
        frame = _get_challenge_frame(page)
        if frame:
            ok = frame.evaluate("""() => {
                const btn =
                    document.querySelector('.button-submit') ||
                    [...document.querySelectorAll('[role="button"]')]
                        .find(b => (b.title || b.getAttribute('aria-label') || b.textContent || '')
                                   .toLowerCase().includes('ximo'));
                if (btn) { btn.click(); return true; }
                return false;
            }""")
            if ok:
                print("    [captcha] Submit via JS/frame real (fallback).")
                return True
    except Exception:
        pass

    # 3. FrameLocator + locator Playwright
    try:
        cf = _get_challenge_frame_locator(page)
        for sel in SUBMIT_SELS:
            try:
                cf.locator(sel).first.click(timeout=2_000)
                print(f"    [captcha] Submit via frame_locator({sel}).")
                return True
            except Exception:
                continue
    except Exception:
        pass

    # 4. Coordenadas físicas — botão fica no canto inferior direito do iframe
    try:
        box = _get_challenge_element_locator(page).bounding_box()
        if box and box["width"] > 100:
            x = box["x"] + box["width"] - 40
            y = box["y"] + box["height"] - 17
            page.mouse.click(x, y)
            print(f"    [captcha] Submit via coordenadas ({x:.0f},{y:.0f}).")
            return True
    except Exception:
        pass

    # 5. XPath fallback na página principal
    try:
        page.locator(PROXIMO_XPATH).first.click(timeout=2_000)
        print("    [captcha] Submit via XPath body fallback.")
        return True
    except Exception:
        pass

    print("    [captcha] AVISO: nenhuma estratégia de submit funcionou.")
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Checkbox widget
# ──────────────────────────────────────────────────────────────────────────────

# Estados possíveis no INÍCIO de uma resolução. Vocabulário fechado.
INICIO_DESAFIO = "desafio"
INICIO_CHECKBOX = "checkbox"
INICIO_NENHUM = "nenhum"


def _indice_checkbox_visivel(page) -> int | None:
    """Índice do iframe de checkbox VISÍVEL, ou None. Resolução ÚNICA.

    O hCaptcha mantém mais de um iframe de widget: o da etapa anterior fica
    para trás, oculto, e o novo vem depois na ordem do documento. `.first`
    respondia pelo obsoleto e concluía "não há captcha" com um widget real
    esperando interação na tela.

    Política determinística: o PRIMEIRO visível em ordem de documento. Índice, e
    não booleano, porque quem clica precisa clicar EXATAMENTE o que foi
    detectado — detectar um iframe e clicar outro seria trocar um erro por
    outro.
    """
    try:
        loc = page.locator(CHECKBOX_SEL)
        total = loc.count()
    except Exception:  # noqa: BLE001 — não observar é não estar lá
        return None
    for i in range(total):
        try:
            if loc.nth(i).is_visible():
                return i
        except Exception:  # noqa: BLE001, S112 — um iframe ilegível não
            continue       # invalida os outros
    return None


def _checkbox_visivel(page) -> bool:
    """Inspeção BARATA: há widget 'Sou humano' na tela? Sem esperar."""
    return _indice_checkbox_visivel(page) is not None


def abrir_desafio(page, timeout_ms: int = 10_000) -> bool:
    """Abre o desafio a partir do widget 'Sou humano'. NÃO resolve nada.

    Existe para quem precisa CLASSIFICAR o desafio antes de decidir se pode
    resolvê-lo: o portal Serviços RF aplica política por tipo na representação
    de CNPJ, e com o widget ainda fechado não há tipo nenhum a classificar —
    `detectar_tipo_captcha` só enxerga desafio ABERTO.

    Chamar `solve_hcaptcha` só para abrir violaria essa política, porque ele
    resolveria qualquer tipo. Aqui a fronteira é explícita: abrir não é
    resolver, e nenhuma chamada ao modelo acontece.

    True quando há desafio ativo ao final — inclusive se já havia antes.
    """
    if _get_challenge_frame(page) is not None:
        return True
    if not _click_checkbox_widget(page, timeout_ms=timeout_ms):
        return False
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        if _get_challenge_frame(page) is not None:
            return True
        time.sleep(0.2)
    return False


def _aguardar_desafio_ou_checkbox(page, timeout_ms: int = 10_000) -> str:
    """Observa os DOIS estados dentro do MESMO prazo, e devolve o que vier.

    Antes o início era serial: esperava-se o checkbox por até 10 s e só depois
    se procurava o desafio. Numa grade que já vem aberta — sem checkbox nenhum —
    isso custava os 10 s inteiros antes da primeira classificação, e numa
    execução real o desafio só terminou perto de um minuto e meio depois de
    aberto.

    Desafio ativo tem prioridade sobre checkbox: se o desafio já está na tela,
    clicar num widget antigo não adianta nada.
    """
    deadline = time.time() + timeout_ms / 1000
    while True:
        if _get_challenge_frame(page) is not None:
            return INICIO_DESAFIO
        if _checkbox_visivel(page):
            return INICIO_CHECKBOX
        if time.time() >= deadline:
            return INICIO_NENHUM
        time.sleep(0.2)


def _click_checkbox_widget(page, timeout_ms: int = 10_000) -> bool:
    """Aguarda o checkbox 'Sou humano' aparecer e clica NELE.

    Clica o iframe cujo índice foi comprovadamente visível — não `.first`.
    Detectar um e clicar outro deixaria o fluxo parado num widget obsoleto.
    """
    indice = _indice_checkbox_visivel(page)
    if indice is None:
        deadline = time.time() + timeout_ms / 1000
        while indice is None and time.time() < deadline:
            time.sleep(0.2)
            indice = _indice_checkbox_visivel(page)
    if indice is None:
        print("    [captcha] Checkbox não detectado — pode ser desafio direto.")
        return False

    for tentativa in range(1, 4):
        try:
            cf = page.frame_locator(CHECKBOX_SEL).nth(indice)
            cf.locator("#checkbox").first.click(timeout=3_000)
            print(f"    [captcha] Checkbox clicado (tentativa {tentativa}/3).")
            page.wait_for_timeout(1_000)
            return True
        except Exception as e:
            print(f"    [captcha] Checkbox tentativa {tentativa}/3: {type(e).__name__}")
            page.wait_for_timeout(600)

    print("    [captcha] Não foi possível clicar no checkbox.")
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Polling pós-submit
# ──────────────────────────────────────────────────────────────────────────────

# Onde o acervo rotulado e gravado. Vazio desliga, como o despejo de erro.
#
# Fora do diretorio de artefatos da run de proposito: e material de treino da
# maquina que investiga, nao entrega da automacao.
LICOES_DIR_ENV = "CAPTCHA_LICOES_DIR"

VEREDITO_ACEITO = "aceito"          # o desafio sumiu: resposta certa, confirmada
VEREDITO_AVANCOU = "avancou"        # mudou de desafio: quase certamente certa
VEREDITO_RECUSADO = "recusado"      # mesmo desafio na tela: errada


def _registrar_licao(png: bytes, instrucao: str, tipo: str,
                     resposta, veredito: str) -> None:
    """Guarda (imagem, enunciado, resposta, veredito do PORTAL).

    O gabarito sempre existiu e era jogado fora. Quem decide se a resposta
    estava certa e o proprio portal — `_wait_for_resolve` ja sabia disso, e o
    valor era usado so para decidir se o laco continuava.

    O que se guardava ate 10/09/2026 era o oposto do util: `_guardar_amostra`
    roda quando o solver DESISTE, entao o acervo tinha 81 imagens de fracasso e
    nenhum acerto. Aprender so com erro ensina o que nao fazer, sem ensinar o
    que fazer.

    Com isto o acervo passa a crescer sozinho, rotulado, a cada run — e vira
    material para exemplos no prompt (few-shot) da mesma familia.
    """
    destino = os.environ.get(LICOES_DIR_ENV, "").strip()
    if not destino or not png:
        return
    try:
        pasta = os.path.join(destino, veredito)
        os.makedirs(pasta, exist_ok=True)
        base = f"{time.strftime('%Y%m%d-%H%M%S')}-{tipo}-{os.getpid()}"
        with open(os.path.join(pasta, base + ".png"), "wb") as f:
            f.write(png)
        with open(os.path.join(pasta, base + ".json"), "w", encoding="utf-8") as f:
            json.dump({
                "tipo": tipo,
                "instrucao": _limpar_texto(instrucao),
                "resposta": resposta,
                "veredito": veredito,
                "quando": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, f, ensure_ascii=False, indent=2)
    except Exception:  # noqa: BLE001 — acervo nunca derruba a run
        pass


def _veredito_do_portal(page, fingerprint_antes: str | None,
                        png_antes: bytes | None, timeout_ms: int) -> tuple[bool, str]:
    """(resolveu, veredito) — e o veredito NAO e binario, de proposito.

    O hCaptcha da DUAS rodadas por desafio. Uma rodada 1 respondida CERTO faz
    aparecer a rodada 2, e o desafio continua visivel: `_wait_for_resolve`
    devolve False. Rotular isso como erro ensinaria o acervo ao contrario —
    marcaria como errada exatamente a resposta que funcionou.

    O que separa os dois casos e se o desafio MUDOU, e `_desafio_ainda_e_o_mesmo`
    ja existe para isso (a guarda de frescor). Dai os tres estados:

        sumiu                -> aceito     (certeza)
        continua, mas outro  -> avancou    (quase certamente certa)
        continua, o mesmo    -> recusado   (errada)

    `avancou` fica separado de `aceito` porque nao e a mesma evidencia, e quem
    for usar o acervo precisa poder escolher o quao exigente quer ser.
    """
    if _wait_for_resolve(page, timeout_ms=timeout_ms):
        return True, VEREDITO_ACEITO
    try:
        mesmo = _desafio_ainda_e_o_mesmo(page, fingerprint_antes, png_antes)
    except Exception:  # noqa: BLE001 — sem prova de mudanca, assume o mesmo
        mesmo = True
    return False, VEREDITO_RECUSADO if mesmo else VEREDITO_AVANCOU


def _wait_for_resolve(page, timeout_ms: int = 3_000) -> bool:
    """Polling até o challenge desaparecer ou timeout.

    Verifica a cada 100ms se o desafio sumiu (captcha resolvido).
    Mais preciso do que sleep fixo — detecta resolução rápida imediatamente.
    """
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        if not _challenge_visible(page):
            return True
        time.sleep(0.1)
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Resolvers por tipo
# ──────────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────────
# Tipo: cartao_animal — cartas 2x2 com revelacao animada individual
# ──────────────────────────────────────────────────────────────────────────────

# Centros percentuais de cada carta dentro do viewport do frame (col%, row%)
# Calibrado para hCaptcha cartao_animal 520×402 px:
#   Header ~22% do topo, grid 2×2 ocupa ~53% de altura (22%–75%), footer restante
_CARD_PCT = {
    0: (0.32, 0.40),  # superior-esquerda
    1: (0.70, 0.40),  # superior-direita
    2: (0.32, 0.74),  # inferior-esquerda
    3: (0.70, 0.74),  # inferior-direita
}


def _capturar_sequencia_animacao(page, n_frames: int = 6, interval_s: float = 1.0) -> list:
    """Captura screenshots do iframe a cada interval_s por n_frames vezes.

    6 frames × 1s = cobre ~1,5 ciclos completos (4 cartas × ~1s cada).
    Cada frame é salvo em debug_screenshots/ para inspeção.
    Returns list de PNG bytes.
    """
    iframe_loc = _get_challenge_element_locator(page)
    frames = []

    print(f"    [captcha/cartao] Capturando {n_frames} frames (intervalo {interval_s}s)...")

    for i in range(n_frames):
        try:
            png = iframe_loc.screenshot(timeout=3_000)
            if png and len(png) > 500:
                frames.append(png)
            else:
                print(f"    [captcha/cartao] Frame {i + 1}: vazio ({len(png) if png else 0} bytes).")
        except Exception as e:
            print(f"    [captcha/cartao] Frame {i + 1} erro: {type(e).__name__}")
        if i < n_frames - 1:
            time.sleep(interval_s)

    print(f"    [captcha/cartao] {len(frames)}/{n_frames} frames capturados.")
    return frames


def _frame_carta_esta_faceup(frame, idx_alvo: int) -> bool:
    """Verifica via JS se a carta idx_alvo tem uma imagem carregada (face-up).

    Usa elementFromPoint na posição percentual da carta para encontrar o elemento
    real sem depender de seletores específicos.
    """
    px, py = _CARD_PCT[idx_alvo]
    try:
        return bool(frame.evaluate("""([px, py]) => {
            const x = Math.round(document.documentElement.clientWidth  * px);
            const y = Math.round(document.documentElement.clientHeight * py);
            const el = document.elementFromPoint(x, y);
            if (!el || el === document.documentElement || el === document.body) return false;
            // Verifica <img> carregada no elemento ou descendentes
            const imgs = [el, ...el.querySelectorAll('*')].filter(e => e.tagName === 'IMG');
            for (const img of imgs) {
                if (img.complete && img.naturalWidth > 10) return true;
            }
            // Verifica background-image com área real
            for (const e of [el, ...el.querySelectorAll('*')]) {
                const bg = window.getComputedStyle(e).backgroundImage;
                if (!bg || !bg.startsWith('url(') || bg === 'url()') continue;
                const r = e.getBoundingClientRect();
                if (r.width > 20 && r.height > 20) return true;
            }
            return false;
        }""", [px, py]))
    except Exception:
        return False


def _frame_click_at_pct(frame, idx_alvo: int) -> Optional[str]:
    """Clica no elemento que está na posição percentual da carta idx_alvo no frame.

    Usa elementFromPoint — funciona sem conhecer seletores específicos.
    Retorna descrição do elemento clicado ou None.
    """
    px, py = _CARD_PCT[idx_alvo]
    try:
        return frame.evaluate("""([px, py]) => {
            const x = Math.round(document.documentElement.clientWidth  * px);
            const y = Math.round(document.documentElement.clientHeight * py);
            const el = document.elementFromPoint(x, y);
            if (!el || el === document.documentElement || el === document.body) return null;
            el.dispatchEvent(new MouseEvent('click', {bubbles: true, cancelable: true}));
            const r = el.getBoundingClientRect();
            return el.tagName + ' class="' + el.className + '" at (' + x + ',' + y + ') size=' + Math.round(r.width) + 'x' + Math.round(r.height);
        }""", [px, py])
    except Exception as e:
        print(f"    [captcha/cartao] elementFromPoint erro: {type(e).__name__}")
        return None


def _gemini_cartao_animal(frames: list, api_key: str,
                          politica: PoliticaLatencia | None = None) -> int:
    """Analisa sequência de frames e retorna o índice (0-3) da carta com animal único.

    Envia até 20 frames ao Gemini com prompt que explica a animação sequencial.
    Returns -1 se não for possível identificar.
    """
    # Seleciona no máximo 20 frames igualmente espaçados
    if len(frames) > 20:
        step = len(frames) / 20
        sel = [frames[int(i * step)] for i in range(20)]
    else:
        sel = frames

    prompt = (
        "Estas são screenshots sequenciais de um captcha hCaptcha animado. "
        "O captcha mostra um grid 2×2 de cartas que se revelam UMA DE CADA VEZ "
        "(cada carta vira ~1s, em sequência: superior-esquerda → superior-direita → "
        "inferior-esquerda → inferior-direita, depois repete). "
        "Cada carta mostra um animal quando virada. Três cartas mostram o MESMO animal "
        "e uma mostra um animal DIFERENTE. "
        "Analise TODOS os frames, identifique o animal de cada posição e retorne "
        "qual posição tem o animal ÚNICO. "
        "Posições: 0=superior-esquerda, 1=superior-direita, "
        "2=inferior-esquerda, 3=inferior-direita."
    )

    contents: list = [prompt]
    for png in sel:
        contents.append(_parte_imagem(png))

    for attempt in range(1, MAX_GEMINI_TRIES + 1):
        try:
            # _gemini_call já tenta todos os modelos (fallback em sobrecarga 503).
            result = _gemini_call(contents, _SCHEMA_CARTAO_ANIMAL, api_key,
                                  "cartao", politica)
        except Exception as e:
            print(f"    [captcha/cartao] Gemini erro tentativa {attempt} | "
                  f"{_diagnostico_erro(e)}")
            break  # todos os modelos falharam; repetir rápido não ajuda

        idx_dif = result.get("indice_diferente")
        confianca = result.get("confidence", "low")

        if idx_dif is None or not (0 <= int(idx_dif) <= 3):
            print(f"    [captcha/cartao] Gemini: índice inválido {idx_dif} — tentativa {attempt}.")
            time.sleep(1)
            continue

        idx_dif = int(idx_dif)
        animais = [result.get(f"carta_{i}", "?") for i in range(4)]
        print(
            f"    [captcha/cartao] Carta diferente: idx={idx_dif} | "
            f"animais={animais} | confidence={confianca} | "
            f"{result.get('justificativa', '')[:80]}"
        )

        if confianca == "low":
            print(f"    [captcha/cartao] Confiança baixa — tentativa {attempt}.")
            time.sleep(1)
            continue

        return idx_dif

    return -1


_JS_CLICAR_CARTA = """
(idx) => {
    const sels = [
        '[class*="card"]', '[class*="task"]', '[class*="item"]',
        'li', '[role="listitem"]', '[role="button"]', 'div[tabindex]'
    ];
    for (const sel of sels) {
        const els = [...document.querySelectorAll(sel)];
        const viable = els.filter(el => {
            const r = el.getBoundingClientRect();
            return r.width >= 60 && r.height >= 60 && r.width <= 350 && r.height <= 350;
        });
        if (viable.length >= 4) {
            viable.sort((a, b) => {
                const ra = a.getBoundingClientRect();
                const rb = b.getBoundingClientRect();
                if (Math.abs(ra.top - rb.top) > 20) return ra.top - rb.top;
                return ra.left - rb.left;
            });
            const el = viable[idx];
            const r = el.getBoundingClientRect();
            el.dispatchEvent(new MouseEvent('click', {bubbles: true, cancelable: true}));
            return 'sel=' + sel + ' idx=' + idx + ' cx=' + Math.round(r.left + r.width/2) + ' cy=' + Math.round(r.top + r.height/2) + ' w=' + Math.round(r.width) + ' h=' + Math.round(r.height);
        }
    }
    return null;
}
"""


def _js_click_carta(page, idx_alvo: int) -> bool:
    """Clica no card idx_alvo diretamente via JS no frame — sem coordenadas de página."""
    frame = _get_challenge_frame(page)
    if not frame:
        return False
    try:
        result = frame.evaluate(_JS_CLICAR_CARTA, idx_alvo)
        if result:
            print(f"    [captcha/cartao] JS click OK: {result}")
            return True
        print(f"    [captcha/cartao] JS click: nenhum seletor encontrou 4 cards.")
    except Exception as e:
        print(f"    [captcha/cartao] JS click erro: {type(e).__name__}")
    return False


def _centros_cartas_dom(page) -> Optional[list]:
    """Busca os centros das 4 cartas via DOM do frame de desafio.

    Tenta vários seletores e retorna lista de 4 dicts {cx, cy} em coordenadas
    do viewport do frame (para converter para página: somar o offset do iframe).
    Returns None se não encontrar exatamente 4 cards.
    """
    frame = _get_challenge_frame(page)
    if not frame:
        return None
    try:
        cards = frame.evaluate("""() => {
            const sels = [
                '[class*="card"]', '[class*="task"]', '[class*="item"]',
                '[class*="challenge"]', 'li', '[role="listitem"]',
                '[role="button"]', 'div[tabindex]',
            ];
            for (const sel of sels) {
                const els = [...document.querySelectorAll(sel)];
                const viable = els.filter(el => {
                    const r = el.getBoundingClientRect();
                    return r.width >= 60 && r.height >= 60 &&
                           r.width <= 350 && r.height <= 350;
                });
                if (viable.length >= 4) {
                    viable.sort((a, b) => {
                        const ra = a.getBoundingClientRect();
                        const rb = b.getBoundingClientRect();
                        if (Math.abs(ra.top - rb.top) > 20) return ra.top - rb.top;
                        return ra.left - rb.left;
                    });
                    return viable.slice(0, 4).map(el => {
                        const r = el.getBoundingClientRect();
                        return { cx: r.left + r.width / 2, cy: r.top + r.height / 2,
                                 w: r.width, h: r.height };
                    });
                }
            }
            return null;
        }""")
        if cards and len(cards) == 4:
            return cards
    except Exception:
        pass
    return None


def _mover_cursor_suave(n_movimentos: int = 2) -> None:
    """Move o cursor do OS de forma suave (N trajetórias), simulando movimento humano."""
    try:
        import ctypes, random as _rnd, time as _t
        _u32 = ctypes.windll.user32

        class _PT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

        _sw = _u32.GetSystemMetrics(0) or 1920
        _sh = _u32.GetSystemMetrics(1) or 1080

        pt = _PT()
        _u32.GetCursorPos(ctypes.byref(pt))
        x, y = pt.x, pt.y

        for _ in range(n_movimentos):
            tx = _rnd.randint(150, _sw - 150)
            ty = _rnd.randint(150, _sh - 150)
            steps = _rnd.randint(18, 28)
            for i in range(1, steps + 1):
                nx = int(x + (tx - x) * i / steps)
                ny = int(y + (ty - y) * i / steps)
                _u32.SetCursorPos(nx, ny)
                _t.sleep(_rnd.uniform(0.008, 0.018))
            x, y = tx, ty
    except Exception:
        pass


def _clicar_posicao_cartao(page, idx_alvo: int) -> bool:
    """Clica na carta idx_alvo usando page.mouse.click() com coordenadas absolutas.

    As cartas são renderizadas num CANVAS — eventos DOM não funcionam.
    Usa bounding_box() do iframe locator + posições percentuais (_CARD_PCT)
    para calcular coordenadas absolutas na página.
    """
    _mover_cursor_suave(n_movimentos=2)

    iframe_loc = _get_challenge_element_locator(page)
    try:
        box = iframe_loc.bounding_box()
    except Exception as e:
        print(f"    [captcha/cartao] Erro ao obter bounding_box: {type(e).__name__}")
        return False

    if not box:
        print("    [captcha/cartao] bounding_box retornou None.")
        return False

    px, py = _CARD_PCT[idx_alvo]
    click_x = box["x"] + box["width"]  * px
    click_y = box["y"] + box["height"] * py

    print(
        f"    [captcha/cartao] Clicando carta {idx_alvo} em "
        f"({click_x:.0f},{click_y:.0f}) | "
        f"iframe=({box['x']:.0f},{box['y']:.0f}) "
        f"{box['width']:.0f}×{box['height']:.0f}"
    )

    try:
        page.mouse.click(click_x, click_y)
        return True
    except Exception as e:
        print(f"    [captcha/cartao] Erro no clique: {type(e).__name__}")
        return False


def _solve_cartao_animal(page, api_key: str, max_rounds: int = 3,
                         politica: PoliticaLatencia | None = None) -> bool:
    """Resolve captcha 'Selecione o cartão com um animal diferente' (grid 2×2 animado).

    Estratégia:
      1. Grava sequência de ~14s de screenshots do iframe (cobre ~3 ciclos).
      2. Envia até 20 frames ao Gemini para identificar a carta com animal único.
      3. Aguarda a carta-alvo virar (detecção visual via PIL) e clica por coordenada.
      4. Submete.
    """
    marca_submissoes = _SUBMISSOES
    for rnd in range(1, max_rounds + 1):
        if not _challenge_visible(page):
            return _sumiu("cartao", marca_submissoes)

        print(f"    [captcha/cartao] Rodada {rnd}/{max_rounds}...")

        # ── 1. Capturar sequência de frames (12 × 0.5s = 6s) ─────────────────
        frames = _capturar_sequencia_animacao(page, n_frames=12, interval_s=0.5)

        if len(frames) < 3:
            print(f"    [captcha/cartao] Frames insuficientes ({len(frames)}). Reiniciando...")
            continue

        # ── 2. Gemini identifica a carta diferente ────────────────────────────
        idx_alvo = _gemini_cartao_animal(frames, api_key, politica)

        if not (0 <= idx_alvo <= 3):
            print("    [captcha/cartao] Não foi possível identificar carta diferente.")
            continue

        # ── 3. Clicar no centro da carta quando ela virar ─────────────────────
        clicou = _clicar_posicao_cartao(page, idx_alvo)

        if not clicou:
            print(f"    [captcha/cartao] Não conseguiu clicar na carta {idx_alvo}.")
            continue

        # ── 4. Submit ─────────────────────────────────────────────────────────
        time.sleep(0.3)
        _submit_captcha(page)

        if _wait_for_resolve(page, timeout_ms=4_000):
            print("    [captcha/cartao] Captcha resolvido!")
            return True

        print(f"    [captcha/cartao] Rodada {rnd}: desafio ainda ativo após submit.")

    return False


def _solve_grade(page, api_key: str, max_rounds: int = 5,
                 politica: PoliticaLatencia | None = None) -> bool:
    """Resolve captcha de grade 3x3."""
    marca_submissoes = _SUBMISSOES
    for rnd in range(1, max_rounds + 1):
        if not _challenge_visible(page):
            return _sumiu("grade", marca_submissoes)

        print(f"    [captcha/grade] Rodada {rnd}/{max_rounds} — aguardando tiles carregarem...")
        tiles_ok = _wait_for_tiles(page)
        if not tiles_ok and not _challenge_visible(page):
            return _sumiu("grade", marca_submissoes, "enquanto aguardava tiles")

        ref_img = _get_reference_image_bytes(page)

        valid_tiles: list[int] = []
        result = None
        fingerprint: str | None = None
        for attempt in range(1, MAX_GEMINI_TRIES + 1):
            # Verifica ANTES do screenshot: o challenge pode ter sumido
            # entre o wait_for_tiles e agora (race condition pós-submit)
            if not _challenge_visible(page):
                return _sumiu("grade", marca_submissoes, "antes do screenshot")

            iframe_loc = _get_challenge_element_locator(page)
            try:
                png = iframe_loc.screenshot(timeout=8_000)
                # A identidade nasce COM a captura e dos MESMOS bytes que vão
                # ao modelo — é o que amarra a resposta a este desafio.
                fingerprint = _fingerprint_desafio(page, png)
                _pw, _ph = _png_dims(png)
                print(
                    f"    [captcha/grade] Screenshot capturado: "
                    f"{_pw}x{_ph}px, {len(png) // 1024} KB"
                )
            except Exception as e:
                print(f"    [captcha/grade] Screenshot falhou (tentativa {attempt}): {type(e).__name__}")
                # Se o iframe sumiu é porque o captcha foi resolvido
                if not _challenge_visible(page):
                    return _sumiu("grade", marca_submissoes, "após screenshot falhar")
                time.sleep(1)
                continue

            try:
                # Da 2a tentativa em diante vai ao SEGUNDO GOLEIRO.
                #
                # `attempt - 1` girava o modelo do Gemini — unico jeito de a
                # resposta mudar com `temperature=0.0`. So que girar modelo
                # dentro do mesmo provedor muda pouco, e o Gemini ja tinha
                # respondido algo inutil. Ate 11/09/2026 o astra so entrava
                # quando havia ERRO de chamada; resposta ruim voltava para o
                # Gemini. Erro e imprecisao sao a mesma coisa para quem espera
                # a resposta: ele nao fechou.
                result = _gemini_grade(png, ref_img, api_key, politica,
                                       rodizio=attempt - 1,
                                       direto_ao_segundo=(attempt > 1))
            except Exception as e:
                print(f"    [captcha/grade] Gemini erro (tentativa {attempt}) | "
                      f"{_diagnostico_erro(e)}")
                time.sleep(1)
                continue

            valid_tiles = sorted({i for i in result.get("matching_tiles", []) if 0 <= i <= 8})
            if result.get("confidence") == "low" or not valid_tiles:
                motivo = "confiança baixa" if result.get("confidence") == "low" else "tiles vazios"
                print(f"    [captcha/grade] {motivo} — indo ao segundo "
                      f"provedor (tentativa {attempt})...")
                result, valid_tiles = None, []
                time.sleep(1)
                continue
            break

        if not valid_tiles:
            # Orcamento zerado nao rende outra rodada — e cada uma que insiste
            # CONTA como tentativa frustrada.
            #
            # Medido em 11/09/2026, RUN-ef3f4b9d: da rodada 3 em diante toda
            # chamada ja tinha 0s. As rodadas 3, 4 e 5 foram encenacao —
            # screenshot, "0s", erro, repete —, quinze chamadas que nao tinham
            # como funcionar. E o desfecho delas fez o fluxo concluir que o
            # captcha nao era automatizavel, quando o enunciado era "clique em
            # todos os objetos feitos principalmente de metal", com dois baldes
            # obvios na grade.
            #
            # Parar aqui nao perde nada: sem tempo nao ha chamada possivel.
            # VIAVEL, e nao apenas "maior que zero".
            #
            # `esgotado` so e verdade em 0ms. Com 2,9s no relogio a rodada
            # seguinte roda inteira — screenshot, chamada, erro — para um
            # orcamento que nenhum provedor aceita: o Gemini recusa abaixo de
            # 10s e o astra tambem. Medido na RUN-910bd939, onde as rodadas
            # continuaram girando com 2,9s.
            if politica is not None and (
                    politica.esgotado
                    or 0 <= politica.restante_ms < GEMINI_DEADLINE_MIN_MS):
                print(f"    [captcha/grade] Rodada {rnd}: restam "
                      f"{max(0, politica.restante_ms) / 1000:.1f}s — nenhum "
                      "provedor aceita prazo tao curto. Encerrando.")
                break
            print(f"    [captcha/grade] Rodada {rnd}: sem tiles válidos. Continuando...")
            continue

        print(
            f"    [captcha/grade] '{_limpar_texto(result.get('task_summary'))}' "
            f"| {result.get('confidence')} | tiles={valid_tiles}"
        )

        # ── FRESHNESS GUARD — última coisa antes do PRIMEIRO clique ──────────
        if not _desafio_ainda_e_o_mesmo(page, fingerprint, png):
            print(f"    [captcha/grade] {MSG_DESCARTE}")
            continue   # descarta a resposta INTEIRA e recaptura na próxima rodada

        _click_grade_tiles(page, valid_tiles)
        time.sleep(0.1)
        _submit_captcha(page)

        # Polling até 3s (100ms/check) — mais preciso que sleep fixo de 1.5s
        resolveu, veredito = _veredito_do_portal(page, fingerprint, png, 3_000)
        # `task_summary` no lugar do enunciado: e o criterio que o modelo
        # ENTENDEU, e para agrupar familias no acervo ele serve melhor que o
        # texto cru — alem de o enunciado ja estar visivel no cabecalho da
        # propria imagem que vai junto.
        _registrar_licao(png, str(result.get("task_summary") or ""),
                         TIPO_GRADE, valid_tiles, veredito)
        if resolveu:
            print("    [captcha/grade] Captcha resolvido!")
            return True

    return False


def _solve_grade_fused(page, api_key: str, max_rounds: int = 5,
                       politica: PoliticaLatencia | None = None) -> bool:
    """Resolve captcha grade 3×3 com imagem fundida (tiles não separados no DOM).

    Estratégia de alta assertividade:
      1. Screenshot completo do iframe (contexto: cabeçalho com enunciado).
      2. Recorta a área exata dos tiles via DOM bounds (prompt.bottom → submit.top).
      3. Desenha overlay 3×3 numerado 0-8 sobre o recorte para guiar o Gemini.
      4. Envia AMBAS as imagens ao Gemini com _PROMPT_GRADE_FUSED especializado.
      5. Clica nos tiles usando a bbox do recorte (coordenadas precisas de página).
    """
    marca_submissoes = _SUBMISSOES
    for rnd in range(1, max_rounds + 1):
        if not _challenge_visible(page):
            return _sumiu("grade_fused", marca_submissoes)

        print(f"    [captcha/grade_fused] Rodada {rnd}/{max_rounds}...")
        time.sleep(0.5)

        if not _challenge_visible(page):
            return _sumiu("grade_fused", marca_submissoes)

        # ── 1. Screenshot completo do iframe ─────────────────────────────────
        iframe_loc = _get_challenge_element_locator(page)
        iframe_box = iframe_loc.bounding_box()
        fingerprint: str | None = None
        try:
            iframe_png = iframe_loc.screenshot(timeout=8_000)
            fingerprint = _fingerprint_desafio(page, iframe_png)
        except Exception as e:
            print(f"    [captcha/grade_fused] Screenshot falhou (rodada {rnd}): {type(e).__name__}")
            if not _challenge_visible(page):
                return True
            time.sleep(1)
            continue

        # ── 2. Recorte da área dos tiles via DOM bounds ───────────────────────
        frame = _get_challenge_frame(page)
        grid_page_bbox: Optional[dict] = None
        tiles_png: Optional[bytes] = None

        if frame and iframe_box:
            try:
                bounds = frame.evaluate("""() => {
                    const promptSels = ['.prompt-text', '.challenge-header', 'h2',
                                        '.header-text', '.task-label',
                                        '[class*="prompt"]', '[class*="label"]'];
                    let imgTop = 0;
                    for (const sel of promptSels) {
                        const el = document.querySelector(sel);
                        if (el) {
                            const b = el.getBoundingClientRect().bottom;
                            if (b > imgTop) imgTop = b;
                        }
                    }
                    const btnSels = ['.button-submit', '.button-verify',
                                     '[class*="submit"]', '[class*="verify"]'];
                    let imgBottom = document.documentElement.clientHeight;
                    for (const sel of btnSels) {
                        const el = document.querySelector(sel);
                        if (el) {
                            const t = el.getBoundingClientRect().top;
                            if (t < imgBottom) imgBottom = t;
                        }
                    }
                    const w = document.documentElement.clientWidth;
                    const h = imgBottom - imgTop;
                    if (imgTop < 10 || h < 50 || w < 50) return null;
                    return {x: 0, y: imgTop, width: w, height: h};
                }""")
                if bounds:
                    grid_page_bbox = {
                        "x":      iframe_box["x"] + bounds["x"],
                        "y":      iframe_box["y"] + bounds["y"],
                        "width":  bounds["width"],
                        "height": bounds["height"],
                    }
                    tiles_raw = page.screenshot(clip=grid_page_bbox)
                    if tiles_raw:
                        # ── 3. Overlay 3×3 numerado ──────────────────────────
                        tiles_png = _overlay_3x3_grid(tiles_raw)
                        print(
                            f"    [captcha/grade_fused] Tiles recortados: "
                            f"{bounds['width']:.0f}×{bounds['height']:.0f}px"
                        )
            except Exception as e:
                print(f"    [captcha/grade_fused] Recorte falhou: {type(e).__name__}")

        # ── 4. Gemini ─────────────────────────────────────────────────────────
        valid_tiles: list[int] = []
        result = None

        for attempt in range(1, MAX_GEMINI_TRIES + 1):
            if not _challenge_visible(page):
                return _sumiu("grade_fused", marca_submissoes, "antes do Gemini")

            try:
                if tiles_png:
                    # Envia iframe completo + recorte com overlay → prompt especializado
                    result = _gemini_grade_fused(iframe_png, tiles_png, api_key, politica)
                else:
                    # Fallback: só o iframe, prompt genérico de grade
                    ref_img = _get_reference_image_bytes(page)
                    result = _gemini_grade(iframe_png, ref_img, api_key, politica)
            except Exception as e:
                print(f"    [captcha/grade_fused] Gemini erro (tentativa {attempt}) | "
                      f"{_diagnostico_erro(e)}")
                time.sleep(1)
                continue

            valid_tiles = sorted({i for i in result.get("matching_tiles", []) if 0 <= i <= 8})
            if result.get("confidence") == "low" or not valid_tiles:
                motivo = "confiança baixa" if result.get("confidence") == "low" else "tiles vazios"
                print(f"    [captcha/grade_fused] {motivo} — retentando Gemini (tentativa {attempt})...")
                result, valid_tiles = None, []
                time.sleep(1)
                continue
            break

        if not valid_tiles:
            # Orcamento zerado nao rende outra rodada — e cada uma que insiste
            # CONTA como tentativa frustrada.
            #
            # Medido em 11/09/2026, RUN-ef3f4b9d: da rodada 3 em diante toda
            # chamada ja tinha 0s. As rodadas 3, 4 e 5 foram encenacao —
            # screenshot, "0s", erro, repete —, quinze chamadas que nao tinham
            # como funcionar. E o desfecho delas fez o fluxo concluir que o
            # captcha nao era automatizavel, quando o enunciado era "clique em
            # todos os objetos feitos principalmente de metal", com dois baldes
            # obvios na grade.
            #
            # Parar aqui nao perde nada: sem tempo nao ha chamada possivel.
            # VIAVEL, e nao apenas "maior que zero".
            #
            # `esgotado` so e verdade em 0ms. Com 2,9s no relogio a rodada
            # seguinte roda inteira — screenshot, chamada, erro — para um
            # orcamento que nenhum provedor aceita: o Gemini recusa abaixo de
            # 10s e o astra tambem. Medido na RUN-910bd939, onde as rodadas
            # continuaram girando com 2,9s.
            if politica is not None and (
                    politica.esgotado
                    or 0 <= politica.restante_ms < GEMINI_DEADLINE_MIN_MS):
                print(f"    [captcha/grade_fused] Rodada {rnd}: restam "
                      f"{max(0, politica.restante_ms) / 1000:.1f}s — nenhum "
                      "provedor aceita prazo tao curto. Encerrando.")
                break
            print(f"    [captcha/grade_fused] Rodada {rnd}: sem tiles válidos. Continuando...")
            continue

        print(
            f"    [captcha/grade_fused] '{_limpar_texto(result.get('task_summary'))}' "
            f"| {result.get('confidence')} | tiles={valid_tiles}"
        )

        # ── 5. Clique nos tiles usando bbox precisa ───────────────────────────
        # FRESHNESS GUARD, duas perguntas distintas: o desafio ainda é o mesmo,
        # e a geometria ainda é a mesma. A segunda não é redundante — aqui o
        # clique é por PIXEL, com `grid_page_bbox` calculada ANTES do Gemini.
        # Um scroll não muda um pixel da imagem e ainda assim manda o clique
        # para um ponto arbitrário da página.
        if not _desafio_ainda_e_o_mesmo(page, fingerprint, iframe_png):
            print(f"    [captcha/grade_fused] {MSG_DESCARTE}")
            continue
        if not _geometria_estavel(page, iframe_box):
            print(f"    [captcha/grade_fused] {MSG_DESCARTE}")
            continue

        frame = _get_challenge_frame(page)
        task_count = 0
        if frame:
            try:
                task_count = frame.locator(TASK_SEL).count()
            except Exception:
                pass

        if task_count >= 9:
            _click_grade_tiles(page, valid_tiles)
        else:
            # Passa a grid_page_bbox do recorte para clicks precisos
            _click_fused_grade_tiles(page, valid_tiles, grid_page_bbox)

        time.sleep(0.1)
        _submit_captcha(page)

        resolveu, veredito = _veredito_do_portal(page, fingerprint, iframe_png, 3_000)
        _registrar_licao(iframe_png, str(result.get("task_summary") or ""),
                         TIPO_GRADE_FUSED, valid_tiles, veredito)
        if resolveu:
            print("    [captcha/grade_fused] Captcha resolvido!")
            return True

    return False


def _solve_imagem(page, api_key: str, max_rounds: int = 5,
                  politica: PoliticaLatencia | None = None) -> bool:
    """Resolve captcha de imagem completa com grid 20x20."""
    marca_submissoes = _SUBMISSOES
    for rnd in range(1, max_rounds + 1):
        if not _challenge_visible(page):
            return _sumiu("imagem", marca_submissoes)

        print(f"    [captcha/imagem] Rodada {rnd}/{max_rounds}...")

        instrucao = _extrair_instrucao(page)
        print(f"    [captcha/imagem] Instrução: '{_limpar_texto(instrucao)}'")

        png_raw, area_bbox = _get_task_image_screenshot_and_bbox(page)
        if not png_raw:
            print("    [captcha/imagem] Screenshot falhou — aguardando...")
            time.sleep(1)
            continue

        # Aqui a captura enviada ao modelo vem de outro mecanismo (área da
        # imagem, não o iframe). A identidade usa a captura CANÔNICA do iframe,
        # que é a mesma dos dois lados da comparação — um screenshot a mais,
        # de propósito: comparar mecanismos diferentes rejeitaria sempre.
        _png_ident, _caixa_ident = _capturar_desafio(page)
        fingerprint = _fingerprint_desafio(page, _png_ident)

        # SEM malha. Medido em 09/09/2026 contra as amostras arquivadas, com
        # as respostas marcadas na imagem para conferencia visual:
        #
        #     com malha 20x20   caiu na agua vazia entre duas pipas;  10-26s
        #     pixel direto      caiu em cima da pipa certa;            2-5s
        #
        # A malha existia para dar ao modelo um vocabulario de posicao. Mas ela
        # DESENHA linhas e numeros sobre uma imagem que ja e camuflagem
        # deliberada — soma ruido ao problema, e este formato depende
        # justamente de enxergar a FORMA das figuras.
        #
        # A grade 3x3 nao muda: la os tiles ja sao celulas de verdade.
        print(f"    [captcha/imagem] Screenshot: {len(png_raw) // 1024} KB")

        try:
            # `rodizio` faz cada RODADA começar num modelo diferente. Sem ele,
            # rodada 2 mandava a mesma imagem para o mesmo modelo com
            # temperature=0.0 — resposta byte a byte idêntica, garantida. Eram 4
            # rodadas de repetição pura, cada uma com screenshots e uma chamada
            # ao modelo, e nenhuma chance de mudar de resultado.
            #
            # O mecanismo já existia e estava ligado nos outros resolvedores;
            # este ficou de fora.
            result = _gemini_pixel(png_raw, instrucao, api_key, politica,
                                   rodizio=rnd - 1)
        except Exception as e:
            print(f"    [captcha/imagem] Gemini falhou | {_diagnostico_erro(e)}")
            continue

        confidence = result.get("confidence", "low")
        action     = result.get("action", "click")
        tem_ponto  = result.get("x") is not None and result.get("y") is not None
        print(f"    [captcha/imagem] action={action} | confidence={confidence} | "
              f"{'1 ponto' if tem_ponto else 'sem ponto'}")

        if confidence == "low":
            print("    [captcha/imagem] Confiança baixa — retentando...")
            continue

        # ── FRESHNESS GUARD — antes de qualquer clique ou digitação ──────────
        if not _desafio_ainda_e_o_mesmo(page, fingerprint, _png_ident):
            print(f"    [captcha/imagem] {MSG_DESCARTE}")
            continue

        if action == "click":
            if not tem_ponto:
                print("    [captcha/imagem] Sem ponto — retentando...")
                continue
            # Clique por pixel sobre `area_bbox`, medida antes do modelo.
            if not _geometria_estavel(page, _caixa_ident):
                print(f"    [captcha/imagem] {MSG_DESCARTE}")
                continue
            # A guarda de "veio lista de candidatos" saiu daqui: o esquema
            # `ESQUEMA_PIXEL` devolve UM ponto por construcao, entao a lista
            # que ela recusava nao existe mais. O que ela protegia — clicar em
            # tudo quando o enunciado pede um — virou impossivel de expressar.
            _click_pixel(page, result, area_bbox, _dimensoes_png(png_raw))
        elif action == "type":
            txt = result.get("text_answer", "").strip()
            if txt:
                try:
                    cf = _get_challenge_frame_locator(page)
                    cf.locator("input").first.fill(txt)
                    print(f"    [captcha/imagem] Digitado: '{txt}'")
                except Exception as e:
                    print(f"    [captcha/imagem] Erro ao digitar: {type(e).__name__}")

        time.sleep(0.2)
        _submit_captcha(page)

        resolveu, veredito = _veredito_do_portal(page, fingerprint, _png_ident, 3_000)
        # A imagem gravada e a que FOI AO MODELO (`png_raw`), nao a de
        # identidade: o acervo precisa do que ele viu para servir de exemplo,
        # e `_png_ident` existe so para comparar frescor.
        _registrar_licao(png_raw, instrucao, TIPO_IMAGEM,
                         {"x": result.get("x"), "y": result.get("y")}, veredito)
        if resolveu:
            print("    [captcha/imagem] Captcha resolvido!")
            return True

    return False


# ──────────────────────────────────────────────────────────────────────────────
# "Clique no animal que a bola nunca toca"
# ──────────────────────────────────────────────────────────────────────────────
#
# A resposta nao esta numa imagem: a bola se move e PAUSA sobre cada animal, e a
# resposta e quem ela nunca toca. Um screenshot unico nao contem a informacao —
# e precisa a SEQUENCIA. Por isso este resolvedor captura, e nao fotografa.
#
# Ciclo da animacao medido em ~9,9s. A captura sozinha
# (BOLA_FRAMES * BOLA_INTERVALO_S = 7s) ja consome a maior parte de um orcamento
# apertado — e o motivo de `_solve_bola` exigir deadline e recusar rodar sem um.
BOLA_FRAMES = 14
BOLA_INTERVALO_S = 0.5
BOLA_N_ALTA = 2
BOLA_N_BAIXA = 6
BOLA_ESC_ALTA = 0.85
BOLA_ESC_BAIXA = 0.30
# Transicao de rodada real mede dezenas de milhares de px (medido: 42.924).
# Limiar SO relativo erra quando a media ja e ruido (desafio parado): um "7x a
# media" de 51 px vale 356 — por isso o piso absoluto.
BOLA_TRANSICAO_MIN_PX = 20_000

# CAPTURA LONGA, da segunda rodada em diante.
#
# Medido em 08/09/2026 contra os quadros reais do desafio da abelha, capturados
# pela coleta automatica: em TODAS as tentativas — Gemini e segundo provedor,
# quatro escalas diferentes, tres contagens de quadros — o modelo via a abelha
# visitar apenas 2 ou 3 de 5 flores. Sem ver a visita, a eliminacao nao fecha, e
# o criterio de seguranca corretamente impede o chute.
#
# Nao era escala (0,30 a 0,85 deram o mesmo) nem modelo (os dois falharam
# igual). Sao os 7 s de captura, que pegam um trecho do ciclo em que ela nao
# passa nas outras flores. A abelha e MAIS DIFICIL que a bola, nao uma variacao
# dela: a bola mudava 0,90-2,12% dos pixels entre quadros e a abelha muda
# 0,27-0,42% — menor, mais rapida, e pausa menos.
#
# PROGRESSIVO, e nao fixo: a rodada 1 segue curta, e so quem nao fechou paga a
# captura longa. O formato que ja funciona nao fica mais lento.
#
# 30 x 0,5 s = 15 s cobrem ~1,5 ciclo de 9,9 s, contra 0,7 ciclo dos 7 s. O
# payload NAO cresce: a selecao continua mandando 8 quadros ao modelo — o que
# muda e de qual janela eles saem.
#
# A JANELA E LIMITADA PELO PORTAL, nao pelo que seria ideal. Aritmetica do pior
# caso na representacao, com os tempos medidos:
#
#     rodada 1 (7 s)   abertura 5,6 + captura 7 + preparo 1 + chamada 14 + 3
#                      = 30,6 s
#     rodada 2 (15 s)  captura 15 + preparo 1 + chamada 14 + espera 3 = 33,0 s
#     total                                                            63,6 s
#
# Com 21 s na segunda seriam 69,6 s — acima do teto de 60 s e encostando nos
# 70,3 s da maior representacao CONFIRMADA no historico. 15 s e o maximo que
# cabe ali.
#
# No LOGIN esse relogio nao existe, e a janela longa caberia com folga. Mas o
# login hoje chama `solve_hcaptcha` SEM deadline, e `_solve_bola` se recusa a
# rodar sem orcamento — entao o formato animado nao e nem tentado la. Enquanto
# isso nao mudar, a captura longa so existe na representacao, apertada.
BOLA_FRAMES_LONGO = 30
BOLA_INTERVALO_LONGO_S = 0.5

_SCHEMA_BOLA = {
    "type": "object",
    "properties": {
        "animais": {"type": "array", "items": {"type": "string"}},
        "tocados": {"type": "array", "items": {"type": "string"}},
        "resposta": {"type": "string"},
        "col": {
            "type": "integer",
            "description": f"Coluna do grid, 0-based (0=esquerda, {GRID_COLS - 1}=direita).",
        },
        "row": {
            "type": "integer",
            "description": f"Linha do grid, 0-based (0=topo, {GRID_ROWS - 1}=baixo).",
        },
        "justificativa": {"type": "string"},
        "confidence": {"type": "string"},
    },
    "required": ["animais", "tocados", "resposta", "col", "row", "confidence"],
}

# O prompt NAO nomeia bola nem animal — e essa a diferenca entre um resolvedor
# e um catalogo de formatos.
#
# A versao anterior citava "bola" e "animal" 14 vezes e nunca lia o enunciado:
# todo o conhecimento do desafio estava aqui, escrito a mao. Funcionava para
# "clique no animal que a bola nunca toca" e falhava em "clique na flor em que a
# abelha nunca pousa" — mesma mecanica, substantivos outros. Cada variante nova
# custava uma sessao de desenvolvimento.
#
# O `_solve_imagem` ja tinha resolvido isso do outro lado: repassa o enunciado
# LITERAL e pergunta a celula, sem saber o que e o desafio. Foi por isso que ele
# resolveu "quebra o padrao" e "figura diferente" sem ninguem mapear nada. Aqui
# a mesma ideia, para desafios que se movem.
#
# O que fica: metodo de eliminacao, celula da grade, exigencia de eliminacao
# FECHADA. Isso e mecanica, nao vocabulario.
_PROMPT_BOLA = """Você está resolvendo um captcha hCaptcha. A instrução exata, como aparece na
tela, é:

    "{instrucao}"

=== O QUE VOCÊ RECEBEU ===
{cabecalho}

O PRIMEIRO quadro tem uma GRADE vermelha sobreposta, com rótulos "coluna,linha":
{cols} colunas (0 a {max_col}) e {rows} linhas (0 a {max_row}). Os demais quadros
NÃO têm grade — são a mesma cena, nos instantes seguintes.

=== COMO ESTE DESAFIO FUNCIONA ===
Há vários ALVOS fixos, espalhados pela área — podem ser animais, flores,
objetos, símbolos, qualquer coisa. E há UM elemento que se MOVE de quadro a
quadro, passando por cima de alguns alvos.

A resposta é o único alvo que satisfaz a condição do enunciado acima. Na forma
mais comum, é o alvo que o elemento móvel NUNCA alcança em NENHUM quadro — mas
LEIA A INSTRUÇÃO: é ela que define a condição, não esta descrição.

Não assuma cor, forma ou espécie de nada. O que identifica o elemento móvel é
que ele MUDA DE POSIÇÃO entre os quadros; os alvos ficam parados.

=== MÉTODO OBRIGATÓRIO ===
1. Liste os alvos presentes (são os MESMOS em todos os quadros).
2. Para CADA quadro, diga sobre qual alvo o elemento móvel está (ou "nenhum").
3. Elimine todo alvo que a instrução exclui — na forma mais comum, todo alvo
   alcançado em pelo menos um quadro.
4. Sobra um: é a resposta.

=== REGRAS CRÍTICAS ===
  !! NÃO adivinhe pelo primeiro quadro. A informação só existe na SEQUÊNCIA.
  !! Um alvo PARCIALMENTE sobreposto conta como alcançado.
  !! Se sobrar mais de um candidato, diga a confiança como "low" — não force
     uma escolha.
  !! O fundo pode ter textura animada. Ignore o fundo — só importam os alvos e
     o elemento que se move.
  !! IGNORE a barra de botões no rodapé: ali não há alvo nenhum.

=== ONDE ELE ESTÁ ===
`col` e `row` são a CÉLULA DA GRADE em que fica o CENTRO do alvo-resposta, lida
no primeiro quadro. Responda a célula, não pixels.

=== RETORNE ===
  animais: lista dos ALVOS identificados, em português, minúsculas
           (o nome do campo é histórico; vale para alvo de qualquer tipo)
  tocados: lista dos alvos que o elemento móvel alcançou em algum quadro
  resposta: o alvo que satisfaz o enunciado (um só, em português, minúsculas)
  col, row: célula do centro desse alvo, na grade do primeiro quadro
  justificativa: uma frase curta
  confidence: "high" | "medium" | "low"
"""


def _capturar_frames_bola(page, n: int = BOLA_FRAMES,
                          intervalo_s: float = BOLA_INTERVALO_S
                          ) -> tuple[list[bytes], dict | None]:
    """Captura N screenshots da area do desafio, por CLIP — nao por elemento.

    `locator.screenshot()` espera o elemento ficar ESTAVEL antes de capturar, e
    numa area animada isso devolve quase sempre o mesmo quadro: medido, 51 px de
    diferenca media entre "quadros" com a bola visivelmente se movendo na tela.
    `page.screenshot(clip=..., animations="allow")` captura pela geometria, sem
    estabilizar nada — a diferenca media sobe para a ordem de milhares.
    """
    loc = _get_challenge_element_locator(page)
    try:
        caixa = loc.bounding_box()
    except Exception:  # noqa: BLE001
        return [], None
    if not caixa:
        return [], None
    clip = {"x": caixa["x"], "y": caixa["y"],
            "width": caixa["width"], "height": caixa["height"]}
    frames: list[bytes] = []
    for _ in range(n):
        try:
            frames.append(page.screenshot(clip=clip, animations="allow", timeout=4_000))
        except Exception:  # noqa: BLE001
            break
        time.sleep(intervalo_s)
    return frames, caixa


def _maior_trecho_sem_transicao(frames: list[bytes]) -> list[bytes]:
    """Isola uma unica rodada dentro dos frames capturados.

    O hCaptcha faz DUAS rodadas por desafio, e a janela de captura pode
    atravessar a virada — ela troca a cena inteira, o que aparece como um pico
    de dezenas de milhares de pixels mudados entre dois frames consecutivos
    (medido: 42.924 px, contra ~5.000 de media numa rodada em movimento). Sem
    isolar, o modelo recebe frames de dois desafios diferentes misturados e a
    eliminacao nao pode fechar.
    """
    if not _PIL or len(frames) < 3:
        return frames
    imgs = [Image.open(io.BytesIO(f)).convert("RGB") for f in frames]
    difs = []
    for a, b in pairwise(imgs):
        d = ImageChops.difference(a, b).convert("L").point(lambda p: 255 if p > 40 else 0)
        difs.append(sum(d.point(bool).getdata()))
    if not difs:
        return frames
    media = sum(difs) / len(difs)
    cortes = [i + 1 for i, d in enumerate(difs)
              if d > BOLA_TRANSICAO_MIN_PX and d > 7 * media]
    if not cortes:
        return frames
    limites = [0, *cortes, len(frames)]
    trechos = [frames[a:b] for a, b in pairwise(limites)]
    return max(trechos, key=len)


def _amostrar_frames_distintos(frames: list[bytes], n: int) -> list[bytes]:
    """Escolhe `n` frames priorizando os que MUDAM entre si.

    A bola PAUSA sobre cada animal, entao amostragem uniforme reamostra o mesmo
    instante: na primeira coleta, 5 frames a 0,7s devolveram a bola em apenas 3
    posicoes distintas. Mandar quadros repetidos ao modelo gasta payload sem
    acrescentar informacao — e a informacao aqui e justamente o movimento.
    """
    if not _PIL or len(frames) <= n:
        return frames
    imgs = [Image.open(io.BytesIO(f)).convert("RGB").resize((80, 80)) for f in frames]
    escolhidos = [0]
    while len(escolhidos) < n:
        melhor, melhor_d = None, -1
        for i in range(len(frames)):
            if i in escolhidos:
                continue
            d = min(sum(ImageChops.difference(imgs[i], imgs[j])
                        .convert("L").point(bool).getdata())
                    for j in escolhidos)
            if d > melhor_d:
                melhor, melhor_d = i, d
        if melhor is None:
            break
        escolhidos.append(melhor)
    return [frames[i] for i in sorted(escolhidos)]


def _preparar_partes_bola(frames: list[bytes], n_alta: int, esc_alta: float,
                          esc_baixa: float) -> list[bytes]:
    """Recorte hibrido: os primeiros `n_alta` em resolucao maior — para
    IDENTIFICAR especies —, o resto pequenos — so para RASTREAR a bola.

    Identificar especie precisa de resolucao; rastrear posicao nao. Essa
    separacao derrubou o payload de ~1,1 MB (5 quadros grandes, que sempre
    estourava ReadTimeout) para ~130 KB com o dobro de quadros.

    A GRADE vai so no primeiro, que e o quadro a que o prompt se refere para
    pedir a celula. Nos demais ela seria ruido visual sobre a bola.
    """
    saida = []
    for idx, png in enumerate(frames):
        esc = esc_alta if idx < n_alta else esc_baixa
        im = Image.open(io.BytesIO(png)).convert("RGB")
        w, h = im.size
        im = im.resize((max(1, int(w * esc)), max(1, int(h * esc))), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "PNG")
        bruto = buf.getvalue()
        if idx == 0:
            bruto = _overlay_grid(bruto, GRID_COLS, GRID_ROWS)
        im = Image.open(io.BytesIO(bruto)).convert("RGB")
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=80)
        saida.append(buf.getvalue())
    return saida


def _gemini_bola(partes_bin: list[bytes], n_alta: int, api_key: str,
                 politica: PoliticaLatencia | None = None,
                 rodizio: int = 0, instrucao: str = "") -> dict:
    cabecalho = (
        f"{min(n_alta, len(partes_bin))} quadros em ALTA resolução (os primeiros) e "
        f"{max(0, len(partes_bin) - n_alta)} em BAIXA resolução (os seguintes), "
        "todos da MESMA animação, em ordem cronológica."
    )
    contents: list = [_PROMPT_BOLA.format(
        instrucao=(instrucao or "").strip()
        or "Clique no alvo que o elemento em movimento nunca alcança.",
        cabecalho=cabecalho, cols=GRID_COLS, rows=GRID_ROWS,
        max_col=GRID_COLS - 1, max_row=GRID_ROWS - 1)]
    for png in partes_bin:
        contents.append(_gt.Part.from_bytes(data=png, mime_type="image/jpeg"))
    return _gemini_call(contents, _SCHEMA_BOLA, api_key, "bola", politica,
                        rodizio=rodizio)


def _solve_bola(page, api_key: str, max_rounds: int = 2,
                politica: PoliticaLatencia | None = None) -> bool:
    """Resolve "Clique no animal que a bola nunca toca".

    Criterio de seguranca: so clica com a eliminacao FECHADA (exatamente um
    candidato restante). `confidence: "high"` NAO serve de criterio — o modelo
    relatou "high" nos casos em que sobraram dois candidatos e chutou errado.
    Errar aqui tem custo real: alimenta a escalada de dificuldade do hCaptcha
    para a sessao inteira.

    A posicao sai como CELULA DA GRADE, nao como pixel, e a razao e medida.
    Pedindo pixel, das 13 coordenadas devolvidas em 3 amostras, 6 cairam em
    lugar impossivel — dentro da faixa de botoes do rodape, ou a 4-6 px da borda
    de uma imagem de 651x714. O modelo NOMEIA os animais certo e nao sabe
    aponta-los. Com grade, sobre as mesmas 3 amostras, 3/3 com a resposta
    utilizavel e correta. E a mesma tecnica que `_solve_imagem` ja usa aqui, e
    pelo mesmo motivo.

    EXIGE orcamento de tempo LIMITADO (`politica.fim` definido) e desiste sem
    ele. A captura sozinha leva ~7s ANTES da primeira chamada; sem teto, duas
    rodadas completas com a cadeia de modelos inteira ficam livres para levar o
    tempo que os timeouts por chamada permitirem, e o desafio expira na tela
    antes disso — ja observado, com o portal fechando o captcha no meio.
    """
    marca_submissoes = _SUBMISSOES
    if not _PIL:
        print("    [captcha/bola] Pillow indisponível — não é possível resolver.")
        return False
    if politica is None or politica.fim is None:
        print("    [captcha/bola] Sem orçamento de tempo definido — "
              "recusando (este resolvedor só roda com deadline).")
        return False

    for rnd in range(1, max_rounds + 1):
        if _politica(politica).esgotado:
            print(f"    [captcha/bola] Orçamento total esgotado na rodada {rnd} — parando.")
            return False
        if not _challenge_visible(page):
            return _sumiu("bola", marca_submissoes)

        # Rodada 1 curta; da 2a em diante, janela longa. Quem fechou na
        # primeira nao paga por isto.
        longa = rnd > 1
        n_quadros = BOLA_FRAMES_LONGO if longa else BOLA_FRAMES
        intervalo = BOLA_INTERVALO_LONGO_S if longa else BOLA_INTERVALO_S
        print(f"    [captcha/bola] Rodada {rnd}/{max_rounds} — capturando "
              f"{n_quadros} quadros em {n_quadros * intervalo:.0f}s"
              f"{' (janela LONGA)' if longa else ''}...")
        enunciado_origem = _prompt_do_desafio(page)
        # O enunciado vai para o modelo LITERAL. E o que faz este resolvedor
        # servir "a bola nunca toca" e "a abelha nunca pousa" sem saber o que e
        # bola nem abelha.
        instrucao = _extrair_instrucao(page)
        print(f"    [captcha/bola] Instrução: '{_limpar_texto(instrucao)}'")
        frames, caixa = _capturar_frames_bola(page, n=n_quadros,
                                              intervalo_s=intervalo)
        if len(frames) < 5 or not caixa:
            print("    [captcha/bola] Poucos quadros capturados — retentando.")
            continue
        if not _challenge_visible(page):
            return _sumiu("bola", marca_submissoes, "durante a captura")

        rodada = _maior_trecho_sem_transicao(frames)
        selecionados = _amostrar_frames_distintos(rodada, BOLA_N_ALTA + BOLA_N_BAIXA)
        partes_bin = _preparar_partes_bola(
            selecionados, BOLA_N_ALTA, BOLA_ESC_ALTA, BOLA_ESC_BAIXA)

        try:
            result = _gemini_bola(partes_bin, BOLA_N_ALTA, api_key, politica,
                                  rodizio=rnd - 1, instrucao=instrucao)
        except Exception as e:  # noqa: BLE001
            print(f"    [captcha/bola] Gemini falhou | {_diagnostico_erro(e)}")
            continue

        animais = [str(a).strip().lower() for a in (result.get("animais") or [])]
        tocados = {str(a).strip().lower() for a in (result.get("tocados") or [])}
        restantes = [a for a in animais if a not in tocados]
        print(f"    [captcha/bola] animais={animais} tocados={sorted(tocados)} "
              f"restantes={restantes} confidence={result.get('confidence')}")

        if len(restantes) != 1:
            print(f"    [captcha/bola] eliminação não fechou ({len(restantes)} "
                  "candidatos) — retentando em vez de chutar.")
            continue

        # Freshness leve: fingerprint de pixel exato NAO SERVE aqui — a area e
        # animada, dois frames do MESMO desafio nunca batem byte a byte. O
        # enunciado ainda igual + geometria estavel e o que da para checar.
        if not _challenge_visible(page) or _prompt_do_desafio(page) != enunciado_origem:
            print(f"    [captcha/bola] {MSG_DESCARTE}")
            continue
        if not _geometria_estavel(page, caixa):
            print(f"    [captcha/bola] {MSG_DESCARTE}")
            continue

        col, row = result.get("col"), result.get("row")
        if col is None or row is None:
            print("    [captcha/bola] Sem célula — retentando.")
            continue
        # Mesma conta de `_click_grid_positions`: a celula e uma FRACAO da
        # caixa, entao a escala com que a imagem foi enviada nao entra aqui.
        col = max(0, min(GRID_COLS - 1, int(col)))
        row = max(0, min(GRID_ROWS - 1, int(row)))
        x_real = caixa["x"] + (col + 0.5) * (caixa["width"] / GRID_COLS)
        y_real = caixa["y"] + (row + 0.5) * (caixa["height"] / GRID_ROWS)

        _mover_cursor_suave(1)
        try:
            page.mouse.click(x_real, y_real)
            print(f"    [captcha/bola] Clicado em '{result.get('resposta')}' "
                  f"célula=({col},{row}) -> ({x_real:.0f},{y_real:.0f})")
        except Exception:  # noqa: BLE001
            print("    [captcha/bola] Erro ao clicar.")
            continue

        time.sleep(0.2)
        _submit_captcha(page)

        if _wait_for_resolve(page, timeout_ms=3_000):
            print("    [captcha/bola] Captcha resolvido!")
            return True

    return False


# ──────────────────────────────────────────────────────────────────────────────
# Ponto de entrada público
# ──────────────────────────────────────────────────────────────────────────────

# Teto por chamada, POR TIPO de desafio.
#
# Ate 09/09/2026 o login mandava 20 s para tudo. Esse numero nasceu de uma
# medicao na grade 3x3 — `16/16, 2,2 s` — e foi aplicado a requisicoes que nao
# se parecem em nada com aquela. No mesmo dia, 14 das 24 falhas do Gemini foram
# `ReadTimeout`: teto NOSSO estourando, nao o Google recusando. Outras 6 foram
# 504 DEADLINE_EXCEEDED, que e o Google dizendo que ELE nao terminou a tempo.
#
# O que muda entre os tipos e o tamanho e a natureza do que se envia:
#
#     grade / grade_fused  9 tiles pequenos                 -> a medicao dos 2,2 s
#     imagem               1 screenshot com malha 20x20      -> 445 KB, muito mais pesado
#     bola / cartao        8 quadros de uma sequencia        -> 8 imagens numa requisicao
#
# Um teto unico ou sufoca o pesado ou desperdicia no leve. E desperdicio aqui
# nao e neutro: o orcamento e TOTAL, entao segundo gasto num modelo que nao vai
# responder e segundo roubado do proximo — e do segundo provedor, que so e
# chamado depois.
#
# Estes valores sao ponto de partida derivado do que ja se mediu, e a linha de
# log diz qual foi aplicado — para a proxima calibragem sair de numero, e nao
# de palpite.
# 09/09/2026, segunda calibragem do dia: a grade voltou de 12s para 25s.
#
# Eu tinha baixado para 12s de manha, para sobrar orcamento. Foi troca ruim, e
# os arquivos mostram: dos 30 erros do Gemini no dia, 26 sao teto NOSSO —
# 9 sao `504 DEADLINE_EXCEEDED`, que e o servidor dizendo que o prazo QUE NOS
# DEMOS expirou, e 12 sao `ReadTimeout`, que e o nosso cliente desistindo. So
# 4 sao `503 high demand`, o unico que e de fato indisponibilidade dele.
#
# Os 504 da grade se concentraram DEPOIS da reducao: cinco entre 15:01 e 15:45,
# contra dois na manha inteira com 20s. Economizar prazo criou o erro que a
# economia deveria evitar, e cada falha custa a rodada inteira — muito mais
# caro que os segundos poupados.
#
# 40s, e nao 25s: os 20s originais JA produziam 504 ocasional, entao qualquer
# numero proximo disso continua cortando o Gemini no meio. Decisao do Jean,
# explicita — "foda-se o orcamento, aumenta pra 30 ou 40s" —, e o raciocinio
# de custo dele fecha: o Gemini e barato e acerta 16/16 aqui, entao esperar
# por ele sai MUITO mais em conta do que empurrar a grade para o provedor
# pago so porque a espera incomoda.
#
# Teto por chamada so vale se o orcamento TOTAL couber: os deadlines em
# `servicos_rf_login/login.py` subiram junto, senao este numero seria enfeite.
TETO_POR_TIPO_MS = {
    TIPO_GRADE:         40_000,
    TIPO_GRADE_FUSED:   40_000,
    TIPO_IMAGEM:        30_000,
    TIPO_BOLA:          30_000,
    TIPO_CARTAO_ANIMAL: 30_000,
}


def _teto_do_tipo(tipo: str, politica: PoliticaLatencia) -> PoliticaLatencia:
    """Ajusta o teto por chamada ao tipo, sem nunca passar do orcamento total.

    Quem chama define o ORCAMENTO (`fim`); o tipo define quanto vale a pena
    esperar por UMA resposta dentro dele. Sao decisoes diferentes e estavam
    coladas no mesmo numero.
    """
    novo = TETO_POR_TIPO_MS.get(tipo)
    if novo is None or novo == politica.timeout_ms:
        return politica
    print(f"    [captcha] Teto por chamada ajustado ao tipo: "
          f"{politica.timeout_ms / 1000:.0f}s -> {novo / 1000:.0f}s ({tipo}).")
    return politica._replace(timeout_ms=novo)


def solve_hcaptcha(page, max_rounds: int = 6, *,
                   gemini_timeout_ms: int | None = None,
                   deadline_s: float | None = None,
                   deadline_max_s: float | None = None,
                   tipo_ja_classificado: str | None = None) -> bool:
    """Resolve hCaptcha na página.

    `gemini_timeout_ms` e `deadline_s` são o ORÇAMENTO DE TEMPO desta chamada.
    `None` nos dois mantém o comportamento de sempre — nenhum consumidor existente
    muda de política sem pedir. Quando informados, valem só aqui: nada de
    variável de ambiente temporária, nada de estado global.

    `deadline_s` é o teto TOTAL, e é ele que manda: o timeout que chega a cada
    request é `min(gemini_timeout_ms, tempo restante)`. Esgotado o orçamento, a
    resolução termina como NÃO CONCLUÍDA — de forma controlada, para quem chamou
    reavaliar o estado da página em vez de esperar mais um minuto.

    Returns:
        True  — captcha resolvido ou ausente
        False — não resolvido no orçamento ou após max_rounds iterações
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key or api_key.startswith("cole-"):
        raise RuntimeError("GEMINI_API_KEY não configurada no ambiente.")

    politica = PoliticaLatencia(
        timeout_ms=GEMINI_TIMEOUT_MS if gemini_timeout_ms is None else gemini_timeout_ms,
        fim=None if deadline_s is None else time.monotonic() + deadline_s,
    )

    ultimo_tipo = TIPO_DESCONHECIDO
    # Instante ZERO do orcamento, para o teto duro ser medido a partir do
    # inicio e nao a partir da ultima extensao.
    inicio_orcamento = time.monotonic() if deadline_s is not None else None

    # Checkbox OU desafio, no mesmo prazo — o que aparecer primeiro. Uma grade
    # já aberta começa a ser classificada de imediato.
    inicio = _aguardar_desafio_ou_checkbox(page, timeout_ms=10_000)
    if inicio == INICIO_NENHUM:
        print("    [captcha] Nenhum captcha na página.")
        return True
    if inicio == INICIO_CHECKBOX:
        # Já sabemos que está visível: o clique não precisa reesperar por ele.
        _click_checkbox_widget(page, timeout_ms=2_000)

    for rnd in range(1, max_rounds + 1):
        if politica.esgotado:
            print("    [captcha] Orçamento de tempo esgotado — resolução não concluída.")
            return False
        print(f"    [captcha] === Iteração {rnd}/{max_rounds} ===")

        # Quem chama pode JA ter classificado — e nesse caso reclassificar e
        # pior do que redundante.
        #
        # A lib de login classifica para decidir a politica ("este tipo pode ser
        # tentado?") e o comentario dela diz "classificacao UMA vez, aqui". Mas
        # este laco reclassificava por dentro, em silencio, e era a SEGUNDA
        # decisao que escolhia o resolvedor. Duas decisoes independentes sobre a
        # mesma tela podem discordar — e discordaram em producao em 08/09/2026:
        #
        #     14:45:38  sonda 0,33%  ->  bola_em_movimento   (lib de login)
        #     14:45:43  sonda 0,25%  ->  grade_fused         (aqui)
        #
        # Limiar em 0,3%: caiu dos dois lados. O desafio animado foi para o
        # resolvedor de quadro parado, que respondeu tres vezes bem e teve as
        # tres respostas descartadas pelo guardiao de frescor — porque num
        # desafio que se mexe a impressao digital muda sempre.
        #
        # So vale para a PRIMEIRA iteracao: da segunda em diante o desafio pode
        # ter mudado de verdade, e ai reclassificar e o certo.
        if rnd == 1 and tipo_ja_classificado:
            tipo = tipo_ja_classificado
            print(f"    [captcha] Tipo informado por quem chamou: {tipo} "
                  "(sem reclassificar).")
        else:
            timeout_det = 10_000 if rnd == 1 else 5_000
            tipo = _detect_challenge_type(page, timeout_ms=timeout_det)
        ultimo_tipo = tipo
        politica = _teto_do_tipo(tipo, politica)

        if tipo == "nenhum":
            print("    [captcha] Nenhum desafio ativo. Captcha concluído.")
            return True

        if tipo == "grade":
            ok = _solve_grade(page, api_key, politica=politica)
        elif tipo == "grade_fused":
            ok = _solve_grade_fused(page, api_key, politica=politica)
        elif tipo == "cartao_animal":
            ok = _solve_cartao_animal(page, api_key, politica=politica)
        elif tipo == TIPO_BOLA:
            ok = _solve_bola(page, api_key, politica=politica)
        else:
            ok = _solve_imagem(page, api_key, politica=politica)

        if not ok:
            # SOLVER QUE DESISTIU NAO SE REINICIA AQUI.
            #
            # Cada resolvedor JA repete por dentro — 5 rodadas em grade, imagem
            # e grade_fused, 3 no cartao, 2 na sequencia. Multiplicar pelas 6
            # iteracoes deste laco dava ate 30 tentativas para o mesmo desafio,
            # e o orcamento cobre umas 3.
            #
            # As outras 27 nasciam com o relogio zerado, e ai `timeout_efetivo`
            # vai a zero e TODA chamada estoura na hora — produzindo um log de
            # "falha na chamada ao modelo" que parece problema do provedor e e
            # do orcamento. Foi o que o Jean via como "sempre falha na segunda
            # leva": a segunda leva ja comecava sem tempo.
            #
            # Este laco existe para o desafio que MUDA (o hCaptcha tem duas
            # rodadas), e esse caso e tratado adiante, depois de `ok`. Repetir
            # um solver que acabou de desistir sobre a MESMA tela nao acrescenta
            # nada: ele ja tentou com todos os modelos e com o segundo provedor.
            print(f"    [captcha] Iteração {rnd}: solver não resolveu — ele já "
                  "repetiu internamente; encerrando.")
            # Inerte sem `CAPTCHA_DEBUG_AMOSTRAS_DIR`.
            _guardar_amostra(page, tipo)
            _diagnosticar_desafio(page, api_key, tipo, _extrair_instrucao(page))
            return False

        # Os solvers já fazem _wait_for_resolve (polling) antes de retornar True,
        # então aqui basta uma folga curta antes de reconfirmar.
        page.wait_for_timeout(500)
        if not _challenge_visible(page):
            print(f"    [captcha] Captcha resolvido na iteração {rnd}!")
            return True

        # PROGRESSO COMPROVADO: uma rodada foi submetida com sucesso e OUTRA
        # apareceu. O hCaptcha faz duas rodadas por desafio, e um teto contado
        # desde o inicio nao sabe disso — resolve a primeira e e cortado no meio
        # da segunda, jogando fora o trabalho ja feito.
        #
        # E o mesmo erro que a bola sofreu (teto de 35 s dimensionado para uma
        # rodada), e aqui a correcao e melhor do que aumentar o numero fixo:
        # so ganha tempo quem MOSTROU que esta avancando. Um desafio que nunca
        # fecha uma rodada nao recebe extensao nenhuma.
        #
        # `deadline_max_s` e o teto duro, e quem chama o define — ele conhece o
        # limite do consumidor (na representacao, o do portal). Sem ele, nao ha
        # extensao: o comportamento e o de sempre.
        if (deadline_max_s is not None and politica.fim is not None
                and inicio_orcamento is not None):
            teto_duro = inicio_orcamento + deadline_max_s
            if politica.fim < teto_duro:
                novo_fim = min(teto_duro, politica.fim + (deadline_s or 0.0))
                ganho = novo_fim - politica.fim
                if ganho > 0.5:
                    politica = politica._replace(fim=novo_fim)
                    print(f"    [captcha] Rodada concluída e outra apareceu — "
                          f"+{ganho:.0f}s de orçamento (progresso comprovado, "
                          f"teto duro {deadline_max_s:.0f}s).")
        print(f"    [captcha] Desafio ainda ativo após iteração {rnd}. Continuando...")

    print(f"    [captcha] Limite de {max_rounds} iterações atingido.")
    # Autopsia: aqui a resolucao JA falhou, entao nao ha orcamento a proteger.
    # Sem CAPTCHA_DEBUG_AMOSTRAS_DIR isto nao acontece.
    _diagnosticar_desafio(page, api_key, ultimo_tipo, _extrair_instrucao(page))
    return False


# Aliases de compatibilidade
solve_captcha = solve_hcaptcha


def cell_to_viewport(cell: str, base_x: float, base_y: float, cell_size_css: float):
    """Stub de compatibilidade — não utilizado nesta implementação."""
    raise NotImplementedError("cell_to_viewport não é usado nesta implementação.")
