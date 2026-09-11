"""As 76 "falhas" sao o que? Infra do provedor, ou contabilidade nossa?

Se a maioria for recusa por orcamento — chamada que nem chegou a sair —, o
numero real de falhas de INFRA e muito menor, e o diagnostico muda de lado.
"""
import glob, io, json, os, re
from collections import Counter

SP = os.path.dirname(os.path.abspath(__file__))

# Recusa NOSSA: o segundo provedor nao foi chamado por falta de tempo. Nos logs
# antigos ela sai mascarada, porque `_diagnostico_erro` nao imprime o texto.
NOSSA_RECUSA = re.compile(
    r"segundo provedor (também|tambem) falhou \| categoria=desconhecido \| tipo=RuntimeError"
    r"|segundo provedor NAO chamado")
PARAMOS = re.compile(r"parando a cadeia do Gemini com|orçamento de tempo esgotado"
                     r"|não vale a ida|nao vale a ida")
FALHA_MODELO = re.compile(r"falha na chamada ao modelo \| (.+?)$")
GEMINI_ERRO = re.compile(r"Gemini erro \(tentativa \d+\) \| (.+?)$")

causas, contexto = Counter(), Counter()
for arq in sorted(glob.glob(os.path.join(SP, "medicao", "RUN-*.json"))):
    d = json.load(io.open(arq, encoding="utf-8"))
    for linha in (d.get("logs") or []):
        m = str(linha)
        if NOSSA_RECUSA.search(m):
            causas["NOSSA: sem orçamento para o 2º provedor"] += 1
        elif PARAMOS.search(m):
            contexto["NOSSA: cadeia interrompida por orçamento"] += 1
        elif FALHA_MODELO.search(m) or GEMINI_ERRO.search(m):
            det = (FALHA_MODELO.search(m) or GEMINI_ERRO.search(m)).group(1)
            cat = re.search(r"categoria=(\w+)", det)
            tipo = re.search(r"tipo=(\w+)", det)
            sts = re.search(r"status=(\d+)", det)
            if sts:
                causas[f"PROVEDOR: HTTP {sts.group(1)}"] += 1
            elif tipo and tipo.group(1) == "RuntimeError" and cat and cat.group(1) == "desconhecido":
                causas["NOSSA (propagada): cadeia sem orçamento"] += 1
            else:
                causas[f"PROVEDOR: {tipo.group(1) if tipo else '?'} "
                       f"({cat.group(1) if cat else '?'})"] += 1

total = sum(causas.values())
print(f"=== {total} eventos de falha, 9 runs ===\n")
nossas = 0
for causa, n in causas.most_common():
    marca = "  <-- nossa" if causa.startswith("NOSSA") else ""
    if causa.startswith("NOSSA"):
        nossas += n
    print(f"  {n:>3}  ({100*n/total:>4.1f}%)  {causa}{marca}")
print(f"\n  nossas (nao chegaram a sair): {nossas}/{total} = {100*nossas/total:.0f}%")
print(f"  provedor de verdade         : {total-nossas}/{total} = {100*(total-nossas)/total:.0f}%")
print()
for k, v in contexto.most_common():
    print(f"  [contexto] {v:>3}  {k}")
