"""Resolvedor de hCaptcha.

Combina o melhor de duas implementações:
  - Detecção robusta do frame ATIVO (múltiplos iframes pré-carregados pelo hCaptcha)
  - Grade 3x3: screenshot do iframe ativo → Gemini (response_schema) → índices 0-8
               → clique direto em .task[n] no DOM do frame
  - Imagem completa: 3 estratégias de screenshot + grid 20x20 (PIL) → Gemini → col/row → pixels
  - Imagem de referência extraída separadamente para prompt mais específico
  - Submit com 5 estratégias em cascata (JS, frame.locator, frame_locator, coords, XPath)
  - google.genai SDK com response_schema → JSON sempre estruturado e válido
  - Thinking habilitado (thinkingBudget=4096) para maior acurácia
"""

from __future__ import annotations

import io
import json
import os
import time
from typing import Optional

try:
    from PIL import Image, ImageDraw, ImageFont
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

GEMINI_MODEL      = "gemini-2.5-flash"
GRID_COLS         = 20
GRID_ROWS         = 20
MAX_GEMINI_TRIES  = 5

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
  !! Se voce retornou [] nas tentativas anteriores, AMPLIE a categoria e seja mais generoso.
  !! Se a referencia e um aviao e a grade tem trens e onibus → INCLUA (todos sao transportes).
  !! Tipicamente de 2 a 5 tiles correspondem ao criterio em cada rodada.
  !! Retornar todos os 9 tiles tambem esta errado.

=== RETORNE ===
  task_summary: criterio identificado de forma clara (para TIPO B: use a categoria ampla, ex.: "transportes / veiculos")
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
        print(f"    [captcha/debug] Erro ao salvar: {e}")


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


def _make_config(schema: dict):
    """GenerateContentConfig com response_schema + thinking (4096 tokens)."""
    kwargs: dict = {
        "temperature": 0.0,
        "response_mime_type": "application/json",
        "response_schema": schema,
    }
    try:
        kwargs["thinking_config"] = _gt.ThinkingConfig(thinking_budget=4096)
    except Exception:
        pass
    try:
        return _gt.GenerateContentConfig(**kwargs)
    except Exception:
        # Fallback sem thinking se a versão instalada não suportar
        kwargs_safe = {k: v for k, v in kwargs.items() if k != "thinking_config"}
        return _gt.GenerateContentConfig(**kwargs_safe)


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


def _detect_challenge_type(page, timeout_ms: int = 12_000) -> str:
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
        page.wait_for_timeout(1200)  # aguarda conteúdo do iframe carregar

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
                # Grade 3x3 é aproximadamente quadrada (0.75–1.4); imagem livre é mais retangular
                if 0.75 <= ratio <= 1.4:
                    print(
                        f"    [captcha] Tipo: grade fused (0 tiles, "
                        f"{bounds['width']:.0f}x{bounds['height']:.0f}px ratio={ratio:.2f})."
                    )
                    return "grade_fused"
        except Exception:
            pass

        print(f"    [captcha] Tipo: imagem completa ({count} tile(s)).")
        return "imagem"
    except Exception as e:
        print(f"    [captcha] Erro ao detectar tipo: {e}. Assumindo grade.")
        return "grade"


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
        print(f"    [captcha/imagem] JS img fallback: {e}")

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
        print(f"    [captcha/imagem] DOM bounds falhou: {e}")

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
        print(f"    [captcha] Erro no grid overlay: {e}")
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
        print(f"    [captcha] Erro no 3x3 overlay: {e}")
        return png


# ──────────────────────────────────────────────────────────────────────────────
# Chamadas ao Gemini
# ──────────────────────────────────────────────────────────────────────────────

def _gemini_grade(png: bytes, ref_img: Optional[bytes], api_key: str) -> dict:
    """Grade 3x3 → Gemini → {task_summary, matching_tiles, confidence}."""
    client = _get_client(api_key)
    if ref_img:
        contents = [
            _gt.Part.from_bytes(data=png,     mime_type="image/png"),
            _gt.Part.from_bytes(data=ref_img, mime_type="image/png"),
            _PROMPT_GRADE_COM_REF,
        ]
    else:
        contents = [
            _gt.Part.from_bytes(data=png, mime_type="image/png"),
            _PROMPT_GRADE,
        ]
    for attempt in range(1, MAX_GEMINI_TRIES + 1):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=_make_config(_SCHEMA_GRADE),
            )
            return json.loads(resp.text)
        except Exception as e:
            print(f"    [captcha/grade] Gemini tentativa {attempt}/{MAX_GEMINI_TRIES}: {e}")
            if attempt < MAX_GEMINI_TRIES:
                time.sleep(min(2 ** attempt, 30))
    raise RuntimeError("Gemini grade: todas as tentativas falharam.")


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


def _gemini_grade_fused(iframe_png: bytes, tiles_png: bytes, api_key: str) -> dict:
    """Grade fused: envia iframe completo (contexto) + tiles recortados com overlay 3x3 → Gemini."""
    client = _get_client(api_key)
    contents = [
        _gt.Part.from_bytes(data=iframe_png, mime_type="image/png"),
        _gt.Part.from_bytes(data=tiles_png,  mime_type="image/png"),
        _PROMPT_GRADE_FUSED,
    ]
    for attempt in range(1, MAX_GEMINI_TRIES + 1):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=_make_config(_SCHEMA_GRADE),
            )
            return json.loads(resp.text)
        except Exception as e:
            print(f"    [captcha/grade_fused] Gemini tentativa {attempt}/{MAX_GEMINI_TRIES}: {e}")
            if attempt < MAX_GEMINI_TRIES:
                time.sleep(min(2 ** attempt, 30))
    raise RuntimeError("Gemini grade_fused: todas as tentativas falharam.")


def _gemini_grid(png: bytes, instrucao: str, api_key: str) -> dict:
    """Imagem+grid → Gemini → {instruction, action, click_positions, confidence}."""
    client = _get_client(api_key)
    prompt = _PROMPT_GRID_TMPL.format(
        cols=GRID_COLS,
        rows=GRID_ROWS,
        max_col=GRID_COLS - 1,
        max_row=GRID_ROWS - 1,
        instruction=instrucao or "Leia a instrucao que aparece na imagem.",
    )
    contents = [
        _gt.Part.from_bytes(data=png, mime_type="image/png"),
        prompt,
    ]
    for attempt in range(1, MAX_GEMINI_TRIES + 1):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=_make_config(_SCHEMA_GRID),
            )
            return json.loads(resp.text)
        except Exception as e:
            print(f"    [captcha/grid] Gemini tentativa {attempt}/{MAX_GEMINI_TRIES}: {e}")
            if attempt < MAX_GEMINI_TRIES:
                time.sleep(min(2 ** attempt, 30))
    raise RuntimeError("Gemini grid: todas as tentativas falharam.")


# ──────────────────────────────────────────────────────────────────────────────
# Execução de cliques
# ──────────────────────────────────────────────────────────────────────────────

def _click_grade_tiles(page, indices: list[int]) -> None:
    """Clica nos tiles por índice 0-8 diretamente no DOM do frame ativo."""
    if not indices:
        return
    _mover_cursor_suave(1)
    cf = _get_challenge_frame_locator(page)
    tasks = cf.locator(TASK_SEL)
    for idx in sorted(set(indices)):
        try:
            tasks.nth(idx).click(delay=30)
            time.sleep(0.05)
            print(f"    [captcha] Tile {idx} clicado.")
        except Exception as e:
            print(f"    [captcha] Erro ao clicar tile {idx}: {e}")


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
            print(f"    [captcha] Erro ao clicar tile fused {idx}: {e}")


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
            print(f"    [captcha] Erro ao clicar grid ({col},{row}): {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Submit com 5 estratégias em cascata
# ──────────────────────────────────────────────────────────────────────────────

def _submit_captcha(page) -> bool:
    """Clica no botão Verificar — 5 estratégias em cascata."""
    page.wait_for_timeout(300)
    print("    [captcha] Submetendo desafio...")

    # 1. JavaScript direto no frame real
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
                print("    [captcha] Submit via JS/frame real.")
                return True
    except Exception:
        pass

    # 2. Frame real + locator Playwright
    try:
        frame = _get_challenge_frame(page)
        if frame:
            for sel in SUBMIT_SELS + ['[role="button"][title*="ximo"]']:
                try:
                    frame.locator(sel).first.click(timeout=2_000)
                    print(f"    [captcha] Submit via frame.locator({sel}).")
                    return True
                except Exception:
                    continue
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

def _click_checkbox_widget(page, timeout_ms: int = 10_000) -> bool:
    """Aguarda o checkbox 'Sou humano' aparecer e clica nele."""
    print("    [captcha] Aguardando checkbox hCaptcha (até 10 s)...")
    try:
        page.locator(CHECKBOX_SEL).first.wait_for(state="visible", timeout=timeout_ms)
    except Exception:
        print("    [captcha] Checkbox não detectado — pode ser desafio direto.")
        return False

    for tentativa in range(1, 4):
        try:
            cf = page.frame_locator(CHECKBOX_SEL)
            cf.locator("#checkbox").first.click(timeout=3_000)
            print(f"    [captcha] Checkbox clicado (tentativa {tentativa}/3).")
            page.wait_for_timeout(2_000)
            return True
        except Exception as e:
            print(f"    [captcha] Checkbox tentativa {tentativa}/3: {e}")
            page.wait_for_timeout(600)

    print("    [captcha] Não foi possível clicar no checkbox.")
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Polling pós-submit
# ──────────────────────────────────────────────────────────────────────────────

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
            print(f"    [captcha/cartao] Frame {i + 1} erro: {type(e).__name__}: {e}")
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
        print(f"    [captcha/cartao] elementFromPoint erro: {e}")
        return None


def _gemini_cartao_animal(frames: list, api_key: str) -> int:
    """Analisa sequência de frames e retorna o índice (0-3) da carta com animal único.

    Envia até 20 frames ao Gemini com prompt que explica a animação sequencial.
    Returns -1 se não for possível identificar.
    """
    client = _get_client(api_key)

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
        contents.append(_gt.Part.from_bytes(data=png, mime_type="image/png"))

    for attempt in range(1, MAX_GEMINI_TRIES + 1):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=_make_config(_SCHEMA_CARTAO_ANIMAL),
            )
            result = json.loads(resp.text)
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

        except Exception as e:
            print(f"    [captcha/cartao] Gemini erro tentativa {attempt}: {e}")
            if attempt < MAX_GEMINI_TRIES:
                time.sleep(min(2 ** attempt, 10))

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
        print(f"    [captcha/cartao] JS click erro: {e}")
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
        print(f"    [captcha/cartao] Erro ao obter bounding_box: {e}")
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
        print(f"    [captcha/cartao] Erro no clique: {e}")
        return False


def _solve_cartao_animal(page, api_key: str, max_rounds: int = 3) -> bool:
    """Resolve captcha 'Selecione o cartão com um animal diferente' (grid 2×2 animado).

    Estratégia:
      1. Grava sequência de ~14s de screenshots do iframe (cobre ~3 ciclos).
      2. Envia até 20 frames ao Gemini para identificar a carta com animal único.
      3. Aguarda a carta-alvo virar (detecção visual via PIL) e clica por coordenada.
      4. Submete.
    """
    for rnd in range(1, max_rounds + 1):
        if not _challenge_visible(page):
            print("    [captcha/cartao] Desafio sumiu — resolvido!")
            return True

        print(f"    [captcha/cartao] Rodada {rnd}/{max_rounds}...")

        # ── 1. Capturar sequência de frames (12 × 0.5s = 6s) ─────────────────
        frames = _capturar_sequencia_animacao(page, n_frames=12, interval_s=0.5)

        if len(frames) < 3:
            print(f"    [captcha/cartao] Frames insuficientes ({len(frames)}). Reiniciando...")
            continue

        # ── 2. Gemini identifica a carta diferente ────────────────────────────
        idx_alvo = _gemini_cartao_animal(frames, api_key)

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


def _solve_grade(page, api_key: str, max_rounds: int = 5) -> bool:
    """Resolve captcha de grade 3x3."""
    for rnd in range(1, max_rounds + 1):
        if not _challenge_visible(page):
            print("    [captcha/grade] Desafio sumiu — resolvido!")
            return True

        print(f"    [captcha/grade] Rodada {rnd}/{max_rounds} — aguardando tiles carregarem...")
        tiles_ok = _wait_for_tiles(page)
        if not tiles_ok and not _challenge_visible(page):
            print("    [captcha/grade] Desafio sumiu enquanto aguardava tiles — resolvido!")
            return True

        ref_img = _get_reference_image_bytes(page)

        valid_tiles: list[int] = []
        result = None
        for attempt in range(1, MAX_GEMINI_TRIES + 1):
            # Verifica ANTES do screenshot: o challenge pode ter sumido
            # entre o wait_for_tiles e agora (race condition pós-submit)
            if not _challenge_visible(page):
                print("    [captcha/grade] Desafio sumiu antes do screenshot — resolvido!")
                return True

            iframe_loc = _get_challenge_element_locator(page)
            try:
                png = iframe_loc.screenshot(timeout=8_000)
            except Exception as e:
                print(f"    [captcha/grade] Screenshot falhou (tentativa {attempt}): {type(e).__name__}")
                # Se o iframe sumiu é porque o captcha foi resolvido
                if not _challenge_visible(page):
                    print("    [captcha/grade] Desafio sumiu após screenshot falhar — resolvido!")
                    return True
                time.sleep(1)
                continue

            try:
                result = _gemini_grade(png, ref_img, api_key)
            except Exception as e:
                print(f"    [captcha/grade] Gemini erro (tentativa {attempt}): {e}")
                time.sleep(1)
                continue

            valid_tiles = sorted({i for i in result.get("matching_tiles", []) if 0 <= i <= 8})
            if result.get("confidence") == "low" or not valid_tiles:
                motivo = "confiança baixa" if result.get("confidence") == "low" else "tiles vazios"
                print(f"    [captcha/grade] {motivo} — retentando Gemini (tentativa {attempt})...")
                result, valid_tiles = None, []
                time.sleep(1)
                continue
            break

        if not valid_tiles:
            print(f"    [captcha/grade] Rodada {rnd}: sem tiles válidos. Continuando...")
            continue

        print(
            f"    [captcha/grade] '{result.get('task_summary')}' "
            f"| {result.get('confidence')} | tiles={valid_tiles}"
        )
        _click_grade_tiles(page, valid_tiles)
        time.sleep(0.1)
        _submit_captcha(page)

        # Polling até 3s (100ms/check) — mais preciso que sleep fixo de 1.5s
        if _wait_for_resolve(page, timeout_ms=3_000):
            print("    [captcha/grade] Captcha resolvido!")
            return True

    return False


def _solve_grade_fused(page, api_key: str, max_rounds: int = 5) -> bool:
    """Resolve captcha grade 3×3 com imagem fundida (tiles não separados no DOM).

    Estratégia de alta assertividade:
      1. Screenshot completo do iframe (contexto: cabeçalho com enunciado).
      2. Recorta a área exata dos tiles via DOM bounds (prompt.bottom → submit.top).
      3. Desenha overlay 3×3 numerado 0-8 sobre o recorte para guiar o Gemini.
      4. Envia AMBAS as imagens ao Gemini com _PROMPT_GRADE_FUSED especializado.
      5. Clica nos tiles usando a bbox do recorte (coordenadas precisas de página).
    """
    for rnd in range(1, max_rounds + 1):
        if not _challenge_visible(page):
            print("    [captcha/grade_fused] Desafio sumiu — resolvido!")
            return True

        print(f"    [captcha/grade_fused] Rodada {rnd}/{max_rounds}...")
        time.sleep(0.5)

        if not _challenge_visible(page):
            print("    [captcha/grade_fused] Desafio sumiu — resolvido!")
            return True

        # ── 1. Screenshot completo do iframe ─────────────────────────────────
        iframe_loc = _get_challenge_element_locator(page)
        iframe_box = iframe_loc.bounding_box()
        try:
            iframe_png = iframe_loc.screenshot(timeout=8_000)
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
                print(f"    [captcha/grade_fused] Recorte falhou: {e}")

        # ── 4. Gemini ─────────────────────────────────────────────────────────
        valid_tiles: list[int] = []
        result = None

        for attempt in range(1, MAX_GEMINI_TRIES + 1):
            if not _challenge_visible(page):
                print("    [captcha/grade_fused] Desafio sumiu antes do Gemini — resolvido!")
                return True

            try:
                if tiles_png:
                    # Envia iframe completo + recorte com overlay → prompt especializado
                    result = _gemini_grade_fused(iframe_png, tiles_png, api_key)
                else:
                    # Fallback: só o iframe, prompt genérico de grade
                    ref_img = _get_reference_image_bytes(page)
                    result = _gemini_grade(iframe_png, ref_img, api_key)
            except Exception as e:
                print(f"    [captcha/grade_fused] Gemini erro (tentativa {attempt}): {e}")
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
            print(f"    [captcha/grade_fused] Rodada {rnd}: sem tiles válidos. Continuando...")
            continue

        print(
            f"    [captcha/grade_fused] '{result.get('task_summary')}' "
            f"| {result.get('confidence')} | tiles={valid_tiles}"
        )

        # ── 5. Clique nos tiles usando bbox precisa ───────────────────────────
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

        if _wait_for_resolve(page, timeout_ms=3_000):
            print("    [captcha/grade_fused] Captcha resolvido!")
            return True

    return False


def _solve_imagem(page, api_key: str, max_rounds: int = 5) -> bool:
    """Resolve captcha de imagem completa com grid 20x20."""
    for rnd in range(1, max_rounds + 1):
        if not _challenge_visible(page):
            print("    [captcha/imagem] Desafio sumiu — resolvido!")
            return True

        print(f"    [captcha/imagem] Rodada {rnd}/{max_rounds}...")

        instrucao = _extrair_instrucao(page)
        print(f"    [captcha/imagem] Instrução: '{instrucao}'")

        png_raw, area_bbox = _get_task_image_screenshot_and_bbox(page)
        if not png_raw:
            print("    [captcha/imagem] Screenshot falhou — aguardando...")
            time.sleep(1)
            continue

        png_grid = _overlay_grid(png_raw)
        print(f"    [captcha/imagem] Screenshot com grid: {len(png_grid) // 1024} KB")

        try:
            result = _gemini_grid(png_grid, instrucao, api_key)
        except Exception as e:
            print(f"    [captcha/imagem] Gemini falhou: {e}")
            continue

        positions  = result.get("click_positions") or []
        confidence = result.get("confidence", "low")
        action     = result.get("action", "click")
        print(f"    [captcha/imagem] action={action} | confidence={confidence} | {len(positions)} pontos")

        if confidence == "low":
            print("    [captcha/imagem] Confiança baixa — retentando...")
            continue

        if action == "click":
            if not positions:
                print("    [captcha/imagem] Sem pontos — retentando...")
                continue
            _click_grid_positions(page, positions, area_bbox)
        elif action == "type":
            txt = result.get("text_answer", "").strip()
            if txt:
                try:
                    cf = _get_challenge_frame_locator(page)
                    cf.locator("input").first.fill(txt)
                    print(f"    [captcha/imagem] Digitado: '{txt}'")
                except Exception as e:
                    print(f"    [captcha/imagem] Erro ao digitar: {e}")

        time.sleep(0.2)
        _submit_captcha(page)

        if _wait_for_resolve(page, timeout_ms=3_000):
            print("    [captcha/imagem] Captcha resolvido!")
            return True

    return False


# ──────────────────────────────────────────────────────────────────────────────
# Ponto de entrada público
# ──────────────────────────────────────────────────────────────────────────────

def solve_hcaptcha(page, max_rounds: int = 6) -> bool:
    """Resolve hCaptcha na página.

    Returns:
        True  — captcha resolvido ou ausente
        False — não resolvido após max_rounds iterações
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key or api_key.startswith("cole-"):
        raise RuntimeError("GEMINI_API_KEY não configurada no ambiente.")

    # Clica checkbox "Sou humano" e aguarda desafio abrir
    _click_checkbox_widget(page, timeout_ms=10_000)

    for rnd in range(1, max_rounds + 1):
        print(f"    [captcha] === Iteração {rnd}/{max_rounds} ===")

        timeout_det = 10_000 if rnd == 1 else 5_000
        tipo = _detect_challenge_type(page, timeout_ms=timeout_det)

        if tipo == "nenhum":
            print("    [captcha] Nenhum desafio ativo. Captcha concluído.")
            return True

        if tipo == "grade":
            ok = _solve_grade(page, api_key)
        elif tipo == "grade_fused":
            ok = _solve_grade_fused(page, api_key)
        elif tipo == "cartao_animal":
            ok = _solve_cartao_animal(page, api_key)
        else:
            ok = _solve_imagem(page, api_key)

        if not ok:
            print(f"    [captcha] Iteração {rnd}: solver não resolveu. Próxima tentativa...")
            continue

        page.wait_for_timeout(1_500)
        if not _challenge_visible(page):
            print(f"    [captcha] Captcha resolvido na iteração {rnd}!")
            return True

        print(f"    [captcha] Desafio ainda ativo após iteração {rnd}. Continuando...")

    print(f"    [captcha] Limite de {max_rounds} iterações atingido.")
    return False


# Aliases de compatibilidade
solve_captcha = solve_hcaptcha


def cell_to_viewport(cell: str, base_x: float, base_y: float, cell_size_css: float):
    """Stub de compatibilidade — não utilizado nesta implementação."""
    raise NotImplementedError("cell_to_viewport não é usado nesta implementação.")
