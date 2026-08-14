"""Duble deterministico do hCaptcha — sem navegador, sem rede, sem Gemini.

O solver so conversa com o mundo por tres canais: `page`, `frame` e o cliente do
google.genai. Este arquivo modela os dois primeiros com fidelidade suficiente
para exercitar o defeito real que o QA encontrou — uma resposta produzida para o
desafio A sendo clicada no desafio B.

Duas decisoes deliberadas:

* **O desafio e um OBJETO.** Trocar de desafio e trocar de instancia. Assim o
  teste expressa "o hCaptcha mudou" sem depender de detalhe de implementacao, e
  o fake nao tem como "quase" mudar.
* **JS nao modelado levanta.** `FakeFrame.evaluate` recusa qualquer script que
  este arquivo nao reconheca, em vez de devolver `None`. Um fake silencioso
  deixaria um caminho novo do solver passar sem cobertura nenhuma.
"""
from resolvedor_captcha import solver

URL_ATIVO = "https://newassets.hcaptcha.com/captcha/v1/x/static/hcaptcha.html#frame=challenge"
URL_INATIVO = "https://newassets.hcaptcha.com/captcha/v1/y/static/hcaptcha.html#frame=challenge"

CAIXA_PADRAO = {"x": 100.0, "y": 200.0, "width": 400.0, "height": 600.0}


class Desafio:
    """Um desafio concreto do hCaptcha.

    `pixels` e o que qualquer screenshot da regiao devolve. Dois desafios
    diferentes tem pixels diferentes — que e exatamente a premissa em que o
    freshness guard se apoia.
    """

    def __init__(self, prompt="selecione todas as imagens com onibus",
                 pixels=b"PNG-DESAFIO-A", caixa=None, n_tiles=9):
        self.prompt = prompt
        self.pixels = pixels
        self.caixa = dict(caixa or CAIXA_PADRAO)
        self.n_tiles = n_tiles


class Captcha:
    """Estado observavel do hCaptcha. E o que o teste manipula."""

    def __init__(self, desafio=None, n_iframes=2, idx_ativo=1):
        self.desafio = desafio or Desafio()
        self.ativo = True
        self.n_iframes = n_iframes      # o hCaptcha pre-carrega varios
        self.idx_ativo = idx_ativo
        self.tiles_clicados = []
        self.cliques_pixel = []
        self.submits = 0
        self.historico = [self.desafio]
        # Gancho do teste: o que o hCaptcha faz DEPOIS de um submit — apresentar
        # outro desafio, ou aceitar e sumir.
        self.ao_submeter = None

    # ── acoes do teste ───────────────────────────────────────────────────────
    def trocar_desafio(self, novo):
        """O hCaptcha apresentou outro desafio (auto-refresh, reload, submit)."""
        self.desafio = novo
        self.historico.append(novo)

    def recarregar(self, novo=None):
        """Reload da pagina: iframes sao substituidos e o desafio e outro."""
        self.idx_ativo = 0
        self.n_iframes = 1
        self.trocar_desafio(novo or Desafio(pixels=b"PNG-DESAFIO-APOS-RELOAD"))

    def resolver(self):
        self.ativo = False

    def submeter(self):
        self.submits += 1
        if callable(self.ao_submeter):
            self.ao_submeter(self)


class _Handle:
    def __init__(self, frame):
        self._frame = frame

    def content_frame(self):
        return self._frame


class FakeFrame:
    def __init__(self, captcha, url, ativo):
        self.captcha, self.url, self._ativo = captcha, url, ativo

    def _esta_ativo(self):
        return bool(self.captcha.ativo and self._ativo)

    def evaluate(self, js, *_a):
        # A ordem importa: o script de "captcha ativo" tambem cita .prompt-text.
        if ".challenge-container" in js:
            return self._esta_ativo()
        if "promptSels" in js:                       # recorte da area de tiles
            if not self._esta_ativo():
                return None
            return {"x": 0, "y": 60, "width": 400, "height": 400}
        if ".task-image" in js:                      # _wait_for_tiles
            return self._esta_ativo()
        if "taskSrcs" in js:                         # imagem de referencia
            return -1
        if ".prompt-text" in js:                     # enunciado do desafio
            return self.captcha.desafio.prompt if self._esta_ativo() else ""
        if ".button-submit" in js:                   # submit via JS
            if self._esta_ativo():
                self.captcha.submeter()
                return True
            return False
        raise AssertionError(f"JS nao modelado pelo fake: {js.strip()[:90]}")

    def locator(self, seletor):
        return FakeLocator(self.captcha, seletor, frame=self)


class FakeLocator:
    """Locator preguicoso, como o do Playwright: resolve so na hora do uso."""

    def __init__(self, captcha, seletor, idx=None, frame=None):
        self.captcha, self.seletor, self.idx, self.frame = captcha, seletor, idx, frame

    # ── navegacao ────────────────────────────────────────────────────────────
    def nth(self, i):
        return FakeLocator(self.captcha, self.seletor, idx=i, frame=self.frame)

    @property
    def first(self):
        return self.nth(0)

    def count(self):
        if self.seletor == solver.CHALLENGE_SEL:
            return self.captcha.n_iframes
        if self.seletor == solver.TASK_SEL or "task" in self.seletor:
            return self.captcha.desafio.n_tiles if self.captcha.ativo else 0
        return 0

    # ── iframe do desafio ────────────────────────────────────────────────────
    def element_handle(self, timeout=None):
        if self.seletor != solver.CHALLENGE_SEL:
            raise RuntimeError("sem element_handle para este seletor")
        i = self.idx or 0
        ativo = self.captcha.ativo and i == self.captcha.idx_ativo
        url = URL_ATIVO if ativo else URL_INATIVO
        return _Handle(FakeFrame(self.captcha, url, ativo))

    def bounding_box(self):
        if not self.captcha.ativo:
            return None
        return dict(self.captcha.desafio.caixa)

    def screenshot(self, timeout=None):
        if not self.captcha.ativo:
            raise RuntimeError("elemento nao esta mais na pagina")
        return self.captcha.desafio.pixels

    # ── tiles e submit ───────────────────────────────────────────────────────
    def click(self, delay=None, timeout=None):
        if not self.captcha.ativo:
            raise RuntimeError("elemento nao esta mais na pagina")
        if self.seletor == solver.TASK_SEL:
            self.captcha.tiles_clicados.append((self.captcha.desafio, self.idx))
            return
        if self.seletor in solver.SUBMIT_SELS:
            self.captcha.submeter()
            return
        raise RuntimeError("elemento nao clicavel")

    def wait_for(self, state=None, timeout=None):
        if not self.captcha.ativo:
            raise RuntimeError("timeout")

    def is_visible(self, timeout=None):
        return bool(self.captcha.ativo)

    def fill(self, texto):
        raise RuntimeError("sem input neste desafio")


class FakeFrameLocator:
    def __init__(self, captcha, idx=0):
        self.captcha, self.idx = captcha, idx

    def nth(self, i):
        return FakeFrameLocator(self.captcha, i)

    @property
    def first(self):
        return self.nth(0)

    def locator(self, seletor):
        return FakeLocator(self.captcha, seletor)


class FakeMouse:
    def __init__(self, captcha):
        self.captcha = captcha

    def click(self, x, y):
        self.captcha.cliques_pixel.append((self.captcha.desafio, x, y))

    def move(self, x, y):
        pass


class FakePage:
    def __init__(self, captcha):
        self.captcha = captcha
        self.mouse = FakeMouse(captcha)

    @property
    def frames(self):
        if not self.captcha.ativo:
            return []
        return [FakeFrame(self.captcha, URL_ATIVO if i == self.captcha.idx_ativo
                          else URL_INATIVO, i == self.captcha.idx_ativo)
                for i in range(self.captcha.n_iframes)]

    def locator(self, seletor):
        return FakeLocator(self.captcha, seletor)

    def frame_locator(self, seletor):
        return FakeFrameLocator(self.captcha)

    def screenshot(self, clip=None):
        if not self.captcha.ativo:
            raise RuntimeError("pagina sem desafio")
        # Recorte da regiao dos tiles: deriva dos MESMOS pixels do desafio, para
        # que trocar de desafio troque tambem o recorte.
        return b"CLIP:" + self.captcha.desafio.pixels

    def wait_for_timeout(self, ms):
        pass
