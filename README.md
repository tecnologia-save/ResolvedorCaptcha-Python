# ResolvedorCaptcha (`resolvedor_captcha`)

Resolvedor automático de **hCaptcha** em Python, projetado para automações com
**Playwright** e integração com **UiPath**. O pacote recebe uma página
(`page`) já controlada pela automação, detecta o tipo de desafio do hCaptcha,
usa o **Google Gemini 2.5 Flash** (visão computacional multimodal) para
interpretar as imagens e executa os cliques/envios necessários até resolver — ou
esgotar as tentativas.

> ⚠️ **Aviso de uso responsável.** Este projeto foi desenvolvido para automação
> de fluxos internos autorizados da empresa. Utilize-o apenas em sistemas e
> contas que você tem permissão para automatizar, respeitando os Termos de Uso
> dos serviços acessados.

---

## Índice

- [Como funciona](#como-funciona)
- [Tipos de desafio suportados](#tipos-de-desafio-suportados)
- [Arquitetura](#arquitetura)
- [Instalação](#instalação)
- [Configuração](#configuração)
- [Uso](#uso)
- [API pública](#api-pública)
- [Depuração](#depuração)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Limitações e notas](#limitações-e-notas)

---

## Como funciona

O fluxo principal (`solve_hcaptcha`) executa os seguintes passos:

1. **Checkbox "Sou humano"** — localiza o iframe do checkbox hCaptcha e clica
   nele para abrir o desafio.
2. **Detecção do frame ativo** — o hCaptcha pré-carrega vários iframes
   `frame=challenge` no DOM, mas apenas um está ativo. O resolvedor identifica o
   frame correto verificando, via JavaScript, dimensões reais do container,
   presença de texto de instrução e botão de envio habilitado.
3. **Detecção do tipo de desafio** — analisa o DOM e a instrução para classificar
   o desafio (grade 3x3, grade fundida, cartas com animais ou imagem livre).
4. **Captura + interpretação** — tira screenshots da área relevante, opcionalmente
   desenha um grid numerado por cima (via Pillow) e envia ao Gemini com um
   `response_schema` JSON estruturado e *thinking* habilitado (4096 tokens).
5. **Execução** — converte a resposta do Gemini em cliques (por índice de tile no
   DOM ou por coordenadas de pixel) e envia o desafio.
6. **Envio em cascata** — o botão "Verificar/Próximo" é acionado por 5 estratégias
   sucessivas (JS direto, `frame.locator`, `frame_locator`, coordenadas físicas e
   XPath de fallback), garantindo robustez contra variações de layout.
7. **Polling pós-envio** — verifica a cada 100 ms se o desafio desapareceu
   (resolvido), em vez de usar esperas fixas.

O ciclo se repete por até `max_rounds` iterações, pois o hCaptcha frequentemente
encadeia múltiplas rodadas de imagens antes de aprovar.

---

## Tipos de desafio suportados

| Tipo | Descrição | Estratégia |
|------|-----------|------------|
| **`grade`** | Grade 3x3 clássica com 9 tiles separados no DOM | Screenshot do iframe → Gemini → clique direto em `.task[n]` |
| **`grade_fused`** | Grade 3x3 em que os 9 tiles formam **uma única imagem** (sem 9 elementos no DOM) | Recorta a área dos tiles, desenha overlay 3x3 numerado, Gemini → clique por coordenada de pixel |
| **`cartao_animal`** | Grid 2x2 animado: "selecione o cartão com um animal diferente" | Captura sequência de frames durante a animação → Gemini analisa todos → clique na carta única |
| **`imagem`** | Imagem livre para clique por coordenadas | Overlay de grid 20x20 (Pillow) → Gemini retorna `col,row` → conversão para pixels |

O resolvedor também distingue automaticamente desafios do tipo **categoria com
imagem de referência** (ex.: *"selecione a imagem da mesma categoria que a
referência"*), extraindo a imagem de referência separadamente e usando um prompt
especializado que generaliza para a categoria ampla (ex.: avião → "transportes").

---

## Arquitetura

```
solve_hcaptcha(page)
        │
        ├─ _click_checkbox_widget        → abre o desafio
        │
        └─ loop (max_rounds)
              ├─ _detect_challenge_type   → "grade" | "grade_fused" | "cartao_animal" | "imagem" | "nenhum"
              │
              ├─ _solve_grade ───────────┐
              ├─ _solve_grade_fused      │   cada um:
              ├─ _solve_cartao_animal    ├──→ screenshot → Gemini (response_schema) → clique → _submit_captcha
              └─ _solve_imagem ──────────┘
```

**Componentes-chave:**

- **Cliente Gemini** (`_get_client`, `_make_config`) — usa o SDK `google.genai`
  com `response_mime_type=application/json` e `response_schema`, garantindo
  saída sempre estruturada e válida. `temperature=0.0` para determinismo.
- **Schemas JSON** — um por tipo de desafio (`_SCHEMA_GRADE`, `_SCHEMA_GRID`,
  `_SCHEMA_CARTAO_ANIMAL`), definindo exatamente os campos que o Gemini deve
  retornar.
- **Prompts especializados** — instruções detalhadas por tipo, incluindo tabelas
  de categorias e regras críticas (ex.: "lista vazia é quase sempre errada").
- **Overlays Pillow** (`_overlay_grid`, `_overlay_3x3_grid`) — desenham grids
  numerados sobre os screenshots para ancorar as respostas do Gemini em
  coordenadas precisas.
- **Detecção de frame ativo** — lida com os múltiplos iframes pré-carregados do
  hCaptcha.

---

## Instalação

Requer **Python 3.10+**.

```bash
# Clonar o repositório
git clone https://github.com/tecnologia-save/ResolvedorCaptcha-Python.git
cd ResolvedorCaptcha-Python

# Instalar dependências
pip install -r requirements.txt

# Instalar o pacote (modo editável durante desenvolvimento)
pip install -e .

# Instalar o navegador do Playwright (Chromium)
playwright install chromium
```

### Dependências

| Pacote | Para quê |
|--------|----------|
| `google-genai` | Cliente do Gemini (visão, `response_schema`, *thinking*) |
| `Pillow` | Desenho dos grids (3x3 e 20x20) sobre os screenshots |
| `playwright` | Fornece o objeto `page` controlado pela automação |

> Se `Pillow` ou `google-genai` não estiverem instalados, o módulo importa sem
> erro, mas os recursos correspondentes ficam indisponíveis em runtime.

---

## Configuração

O resolvedor lê a chave da API do Gemini da variável de ambiente
**`GEMINI_API_KEY`**:

```bash
# Linux / macOS
export GEMINI_API_KEY="sua-chave-aqui"

# Windows (PowerShell)
$env:GEMINI_API_KEY = "sua-chave-aqui"
```

Obtenha uma chave em [Google AI Studio](https://aistudio.google.com/apikey).

Se a variável não estiver definida (ou começar com `cole-`), `solve_hcaptcha`
lança `RuntimeError`.

---

## Uso

O pacote **não abre o navegador** — ele opera sobre uma `page` do Playwright que
sua automação já controla. Exemplo mínimo:

```python
import os
from playwright.sync_api import sync_playwright
from resolvedor_captcha import solve_hcaptcha

os.environ["GEMINI_API_KEY"] = "sua-chave-aqui"

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    page.goto("https://site-com-hcaptcha.exemplo")

    resolvido = solve_hcaptcha(page, max_rounds=6)
    if resolvido:
        print("Captcha resolvido — seguindo o fluxo...")
    else:
        print("Não foi possível resolver o captcha.")

    browser.close()
```

### Integração com UiPath

Em um workflow UiPath com a atividade *Invoke Python Method* (ou *Python Scope*),
passe a referência da `page` do Playwright já instanciado e chame
`solve_hcaptcha`. O pacote foi empacotado (`resolvedor_captcha`) justamente para ser
importável como dependência do ambiente Python configurado no UiPath.

---

## API pública

Exportada em [`resolvedor_captcha/__init__.py`](resolvedor_captcha/__init__.py):

```python
from resolvedor_captcha import solve_hcaptcha, solve_captcha, cell_to_viewport
```

| Função | Assinatura | Descrição |
|--------|-----------|-----------|
| `solve_hcaptcha` | `solve_hcaptcha(page, max_rounds=6) -> bool` | Resolve o hCaptcha presente na página. Retorna `True` se resolvido (ou ausente), `False` se não resolveu após `max_rounds`. |
| `solve_captcha` | *(alias de `solve_hcaptcha`)* | Mantido por compatibilidade. |
| `cell_to_viewport` | `cell_to_viewport(cell, base_x, base_y, cell_size_css)` | **Stub de compatibilidade** — não utilizado nesta implementação (lança `NotImplementedError`). |

---

## Depuração

Cada screenshot enviado ao Gemini é salvo em
[`resolvedor_captcha/debug_screenshots/`](resolvedor_captcha/debug_screenshots/) com um
contador sequencial e um sufixo descritivo (ex.: `007_cartao_frame07.png`). Isso
permite inspecionar visualmente exatamente o que o modelo recebeu em cada etapa.

> Esses arquivos são gerados em runtime e estão no `.gitignore` — não são
> versionados.

O resolvedor também emite logs detalhados no `stdout` com o prefixo `[captcha]`
(e sub-prefixos como `[captcha/grade]`, `[captcha/cartao]`), incluindo o tipo
detectado, os tiles escolhidos e qual estratégia de envio funcionou.

---

## Estrutura do projeto

```
ResolvedorCaptcha/
├── README.md
├── requirements.txt
├── setup.py                      # Empacotamento (resolvedor_captcha, v1.0.0)
├── .gitignore
└── resolvedor_captcha/
    ├── __init__.py               # API pública
    ├── solver.py                 # Toda a lógica de resolução
    ├── prompt.md                 # Prompt de referência (sistema de grade A1/B2)
    └── debug_screenshots/        # Screenshots de depuração (runtime, ignorado)
```

---

## Limitações e notas

- **Dependente de modelo visual** — a acurácia depende do Gemini 2.5 Flash; o
  modelo é configurado em `GEMINI_MODEL` no [solver.py](resolvedor_captcha/solver.py).
- **Custo de API** — cada rodada faz uma ou mais chamadas ao Gemini (com até 5
  retentativas). Desafios encadeados consomem múltiplas chamadas.
- **Sensível a layout** — seletores CSS e posições percentuais (ex.: `_CARD_PCT`
  para o desafio de cartas) foram calibrados para o layout atual do hCaptcha;
  mudanças no provedor podem exigir ajustes.
- **Sincronização síncrona** — usa a API **síncrona** do Playwright
  (`page.evaluate`, `page.mouse.click`, etc.).
- **`cell_to_viewport` é um stub** — herdado de uma implementação anterior, não
  é usado.

---

**Versão:** 1.0.0 · **Python:** ≥ 3.10 · **Pacote:** `resolvedor_captcha`
