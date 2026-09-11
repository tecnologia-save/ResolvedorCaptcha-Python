"""Quanto dura uma chamada ao modelo — separando as que RESPONDERAM das que nao.

O teto por chamada precisa ficar logo acima da pior resposta BOA. O numero em
uso (19,8s) e de 26/08/2026 e nunca foi remedido.

Metodo: o log tem uma linha de screenshot imediatamente antes de cada chamada,
e a linha seguinte do mesmo bloco e o desfecho dela. A diferenca dos carimbos e
a latencia, com um pequeno vies para cima (inclui montar o request).
"""
import glob, io, json, os, re
from datetime import datetime

SP = os.path.dirname(os.path.abspath(__file__))
CAIXA = re.compile(r"^(\S+) (\w+) (.*)$")


def quando(linha):
    m = CAIXA.match(linha)
    if not m:
        return None, None, None
    try:
        return datetime.fromisoformat(m.group(1)), m.group(2), m.group(3)
    except ValueError:
        return None, None, None


boas, falhas = [], []
for arq in sorted(glob.glob(os.path.join(SP, "medicao", "RUN-*.json"))):
    d = json.load(io.open(arq, encoding="utf-8"))
    linhas = [quando(str(l)) for l in (d.get("logs") or [])]
    linhas = [x for x in linhas if x[0]]

    inicio = None
    for t, _lvl, msg in linhas:
        if "Screenshot capturado" in msg or "Screenshot:" in msg:
            inicio = t
            continue
        if inicio is None:
            continue
        dt = (t - inicio).total_seconds()

        # Resposta BOA: o solver imprime o veredito do modelo.
        if re.search(r"\| (high|medium|low) \| tiles=|action=click \| confidence=", msg):
            boas.append(dt); inicio = None
        elif re.search(r"falha na chamada ao modelo|Gemini erro|segundo provedor", msg):
            falhas.append(dt); inicio = None
        elif "Screenshot" in msg:
            inicio = t

def resumo(nome, xs):
    if not xs:
        print(f"{nome}: nenhuma amostra"); return
    xs = sorted(xs)
    def pct(p): return xs[min(len(xs) - 1, int(len(xs) * p))]
    print(f"{nome}: n={len(xs)}  min={xs[0]:.1f}s  p50={pct(.5):.1f}s  "
          f"p90={pct(.9):.1f}s  p95={pct(.95):.1f}s  max={xs[-1]:.1f}s")

resumo("RESPONDERAM ", boas)
resumo("FALHARAM    ", falhas)
if boas:
    xs = sorted(boas)
    print()
    print(f"pior resposta BOA observada: {xs[-1]:.1f}s")
    for teto in (12, 15, 20, 25, 27, 30, 40):
        perdidas = sum(1 for x in xs if x > teto)
        print(f"  teto {teto:>2}s -> cortaria {perdidas}/{len(xs)} "
              f"({100*perdidas/len(xs):.0f}%) das respostas boas")
