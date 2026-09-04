"""Latencia da chamada ao modelo e politica de fallback.

O freshness guard impede o clique errado, mas cada analise que envelhece e uma
rodada jogada fora. Estes testes fecham a outra ponta: quanto tempo uma unica
tentativa pode consumir, e quando vale trocar de modelo em vez de insistir.

Aritmetica do pior caso, antes: 4 modelos x 2 tentativas x ~143 s medidos +
backoff = ~19 minutos sobre UM screenshot. Depois: 4 modelos x 1 tentativa x
30 s = 2 minutos.
"""
import json

import pytest

from resolvedor_captcha import solver


class _Resposta:
    def __init__(self, texto):
        self.text = texto


def _cliente(comportamento):
    """Cliente falso. `comportamento[modelo]` = excecao ou dict de resposta."""
    chamadas = []

    class _Models:
        def generate_content(self, model, contents, config):
            chamadas.append(model)
            acao = comportamento.get(model)
            if isinstance(acao, BaseException):
                raise acao
            return _Resposta(json.dumps(acao if acao is not None else {"ok": True}))

    class _Cliente:
        models = _Models()

    return _Cliente(), chamadas


# ── Timeout explicito ────────────────────────────────────────────────────────

def test_timeout_padrao_e_de_20_segundos():
    # Era 30s ate 67ac1f8 ("teto de 20s por chamada"), que mexeu no solver e
    # deixou este teste para tras. O numero segue fixado a mao de proposito: e
    # ele que multiplica pelo numero de modelos no pior caso, entao mudar o teto
    # tem de doer aqui.
    assert solver.GEMINI_TIMEOUT_MS == 20_000


def test_config_carrega_o_timeout_em_milissegundos():
    """Sem isto o SDK usa o default e uma tentativa dura minutos."""
    config = solver._make_config(solver._SCHEMA_GRADE, solver.GEMINI_MODELS[0])
    assert config.http_options is not None
    assert config.http_options.timeout == solver.GEMINI_TIMEOUT_MS


def test_timeout_acompanha_toda_chamada(monkeypatch):
    """O teto vale por tentativa, em qualquer modelo da lista."""
    for modelo in solver.GEMINI_MODELS:
        config = solver._make_config(solver._SCHEMA_GRADE, modelo)
        assert config.http_options.timeout == solver.GEMINI_TIMEOUT_MS


def test_config_preserva_determinismo_e_schema():
    """O timeout nao pode ter custado nenhuma garantia antiga."""
    config = solver._make_config(solver._SCHEMA_GRADE, solver.GEMINI_MODELS[0])
    assert config.temperature == 0.0
    assert config.response_mime_type == "application/json"
    assert config.response_schema is not None


# ── Politica de sobrecarga ───────────────────────────────────────────────────

def test_sobrecarga_nao_repete_o_mesmo_modelo(monkeypatch):
    """Uma tentativa por modelo sobrecarregado — o pool e dele, nao da chamada."""
    cliente, chamadas = _cliente({
        solver.GEMINI_MODELS[0]: RuntimeError("503 UNAVAILABLE"),
        solver.GEMINI_MODELS[1]: {"ok": 2},
    })
    monkeypatch.setattr(solver, "_get_client", lambda _k: cliente)

    assert solver._gemini_call([], {}, "k", "grade") == {"ok": 2}
    assert chamadas == [solver.GEMINI_MODELS[0], solver.GEMINI_MODELS[1]]
    assert chamadas.count(solver.GEMINI_MODELS[0]) == 1


def test_timeout_tambem_avanca_para_o_proximo_modelo(monkeypatch):
    """A lista esta ordenada por latencia medida: o proximo e o mais rapido
    dos que restam. Insistir no que acabou de estourar e a pior escolha.
    """
    cliente, chamadas = _cliente({
        solver.GEMINI_MODELS[0]: TimeoutError("deadline exceeded"),
        solver.GEMINI_MODELS[1]: {"ok": 3},
    })
    monkeypatch.setattr(solver, "_get_client", lambda _k: cliente)

    assert solver._gemini_call([], {}, "k", "grade") == {"ok": 3}
    assert chamadas.count(solver.GEMINI_MODELS[0]) == 1


@pytest.mark.parametrize("mensagem", ["timeout", "request timed out",
                                      "deadline exceeded"])
def test_timeout_e_classificado_como_transitorio(mensagem):
    assert solver._is_overloaded_error(Exception(mensagem)) is True


def test_todos_sobrecarregados_faz_uma_tentativa_por_modelo(monkeypatch):
    cliente, chamadas = _cliente({m: RuntimeError("503 UNAVAILABLE")
                                  for m in solver.GEMINI_MODELS})
    monkeypatch.setattr(solver, "_get_client", lambda _k: cliente)

    with pytest.raises(RuntimeError):
        solver._gemini_call([], {}, "k", "grade")
    assert chamadas == list(solver.GEMINI_MODELS)   # uma cada, na ordem


# ── Erros que NAO sao transitorios ───────────────────────────────────────────

@pytest.mark.parametrize("mensagem", [
    "API key not valid", "401 unauthorized", "400 INVALID_ARGUMENT",
    "PERMISSION_DENIED",
])
def test_erro_de_configuracao_nao_circula_pelos_modelos(monkeypatch, mensagem):
    """Chave invalida nao melhora no modelo seguinte.

    Deixar circular gastaria quatro chamadas inuteis e ainda pioraria o
    diagnostico, porque o erro reportado seria o do ultimo modelo.
    """
    cliente, chamadas = _cliente({m: RuntimeError(mensagem)
                                  for m in solver.GEMINI_MODELS})
    monkeypatch.setattr(solver, "_get_client", lambda _k: cliente)

    with pytest.raises(RuntimeError):
        solver._gemini_call([], {}, "k", "grade")
    assert set(chamadas) == {solver.GEMINI_MODELS[0]}


def test_erro_de_configuracao_ainda_repete_no_mesmo_modelo(monkeypatch):
    """Retry local continua valendo para o que pode ser intermitente."""
    cliente, chamadas = _cliente({m: RuntimeError("400 INVALID_ARGUMENT")
                                  for m in solver.GEMINI_MODELS})
    monkeypatch.setattr(solver, "_get_client", lambda _k: cliente)

    with pytest.raises(RuntimeError):
        solver._gemini_call([], {}, "k", "grade")
    assert len(chamadas) == solver.GEMINI_TRIES_PER_MODEL


# ── Caminhos de sucesso ──────────────────────────────────────────────────────

def test_sucesso_no_primeiro_modelo_faz_uma_chamada(monkeypatch):
    cliente, chamadas = _cliente({solver.GEMINI_MODELS[0]: {"ok": 1}})
    monkeypatch.setattr(solver, "_get_client", lambda _k: cliente)
    assert solver._gemini_call([], {}, "k", "grade") == {"ok": 1}
    assert chamadas == [solver.GEMINI_MODELS[0]]


def test_ordem_dos_modelos_preservada():
    """A ordem veio de medicao (17/08/2026), filtrada por IDs estaveis."""
    assert solver.GEMINI_MODELS == [
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash",
        "gemini-3.1-flash-lite",
    ]
    assert solver.GEMINI_MODEL == solver.GEMINI_MODELS[0]
    assert len(solver.GEMINI_MODELS) == 3


# ── So ID estavel no caminho quente ──────────────────────────────────────────
#
# A run de 18/08/2026 rodou `flash-latest` (503/ReadTimeout), `3.6-flash` (504),
# `pro-latest` (429) e `3.1-flash-lite` — que e exatamente a lista da 1.0.3. O
# runtime estava com a versao anterior; a 1.0.4 ja tinha tirado os tres. O que
# esta versao acrescenta e a REGRA, para nao depender de lembrar dela.

def test_nenhum_alias_latest_no_caminho_quente():
    """`-latest` troca de versao por tras: o que roda deixa de ser o medido."""
    assert [m for m in solver.GEMINI_MODELS if m.endswith("-latest")] == []


def test_nenhum_id_de_preview_no_caminho_quente():
    """`-preview` pode ser aposentado sem aviso."""
    assert [m for m in solver.GEMINI_MODELS if "preview" in m] == []


def test_nenhum_modelo_pro_no_caminho_quente():
    """Peso desnecessario para uma tarefa visual simples, e o pool menos
    disponivel dos medidos — 3/16 e 4/16."""
    assert [m for m in solver.GEMINI_MODELS if "pro" in m] == []


def test_todo_modelo_do_caminho_quente_tem_medicao_de_imagem():
    """Nenhum entra sem ter concluido chamada REAL com screenshot de captcha.

    A confirmacao de entrada multimodal e por MEDICAO propria, nao por
    documentacao: cada um destes fechou 16/16 na campanha de 17/08/2026, com
    grade 3x3 + response_schema + thinking, acertando o gabarito.
    """
    medidos_com_imagem = {
        "gemini-3.5-flash": "16/16 2,7s",
        "gemini-3.5-flash-lite": "16/16 2,2s",
        "gemini-3.1-flash-lite": "16/16 4,7s (pior 25,4s)",
        "gemini-3-flash-preview": "16/16 3,1s",
        "gemini-flash-lite-latest": "16/16 2,2s",
    }
    sem_medicao = [m for m in solver.GEMINI_MODELS if m not in medidos_com_imagem]
    assert sem_medicao == []


# ── Uma unica camada de retry ────────────────────────────────────────────────

def test_o_sdk_nao_faz_retry_por_conta_propria():
    """PROVA EXECUTAVEL de que nao ha retry em duas camadas.

    `retry_args(None)` do google-genai devolve `stop_after_attempt(1)`, e o
    cliente so ganha politica de retry se `http_options.retry_options` for
    preenchido — o que nao fazemos, nem no cliente nem por requisicao. Logo a
    aritmetica de tentativas e so a do solver, e o teto de tempo por tentativa
    (`GEMINI_TIMEOUT_MS`) vale de verdade.

    Se um dia o SDK passar a retentar por padrao, este teste quebra antes de a
    conta dobrar em producao.
    """
    import tenacity
    from google import genai
    from google.genai import types as gt

    api = genai.Client(api_key="chave-de-teste")._api_client
    assert api._http_options.retry_options is None
    assert isinstance(api._retry.stop, tenacity.stop_after_attempt)
    assert api._retry.stop.max_attempt_number == 1
    # E o http_options que NOS montamos tambem nao liga retry.
    assert gt.HttpOptions(timeout=solver.GEMINI_TIMEOUT_MS).retry_options is None


def test_pior_caso_de_tempo_e_um_timeout_por_modelo():
    """Tres modelos x uma tentativa x 20s = 60s, e nao minutos."""
    assert len(solver.GEMINI_MODELS) * solver.GEMINI_TIMEOUT_MS == 60_000


# Modelos reprovados por MEDICAO — nao voltam ao caminho quente sem nova medida.
# Ou o pool vive saturado (timeout/503 na maioria das chamadas) ou o ID ja foi
# aposentado pelo Google (404). Ver o comentario de GEMINI_MODELS em solver.py.
MODELOS_REPROVADOS = (
    "gemini-pro-latest",       # 3/16 — 11 timeouts; 429 na run de 18/08/2026
    "gemini-3.1-pro-preview",  # 4/16 — 12 timeouts
    "gemini-flash-latest",     # 3/16 — 9 timeouts, 4x 503; 503/ReadTimeout na run
    "gemini-3.6-flash",        # 3/8; 504 na run de 18/08/2026
    "gemini-3.7-flash",        # 2/8
    "gemini-2.5-flash",        # 0/6 — 404 "no longer available"
    "gemini-2.0-flash",        # 404 "no longer available"
)


def test_nenhum_modelo_reprovado_no_caminho_quente():
    """Esta automacao nao roda com modelo instavel: a lista padrao e so aprovado."""
    reprovados_na_lista = [m for m in solver.GEMINI_MODELS if m in MODELOS_REPROVADOS]
    assert reprovados_na_lista == []


def test_pior_caso_de_chamadas_e_um_por_modelo(monkeypatch):
    """Teto de chamadas quando tudo esta sobrecarregado."""
    cliente, chamadas = _cliente({m: RuntimeError("overloaded")
                                  for m in solver.GEMINI_MODELS})
    monkeypatch.setattr(solver, "_get_client", lambda _k: cliente)
    with pytest.raises(RuntimeError):
        solver._gemini_call([], {}, "k", "grade")
    assert len(chamadas) == len(solver.GEMINI_MODELS)


# ── Classificacao por STATUS, nao por substring ──────────────────────────────
#
# O historico do orquestrador (04/09/2026) mostrava a MESMA sequencia de erro
# saindo como `tempo_esgotado` numa tentativa e `requisicao_invalida` na
# seguinte. Causa: a classificacao varria substrings no texto inteiro do erro, e
# ("timeout", "timed out", "deadline") vinha ANTES de "400" na lista.

def test_400_com_deadline_no_corpo_e_requisicao_invalida():
    """O corpo do provedor nao decide a categoria — o status decide."""
    erro = RuntimeError(
        "400 INVALID_ARGUMENT: request deadline field is not supported here")
    assert solver._categoria_do_erro(erro) == "requisicao_invalida"


def test_400_com_deadline_no_corpo_nao_troca_de_modelo():
    """Era o efeito caro: circulava a requisicao invalida por todos os modelos."""
    erro = RuntimeError(
        "400 INVALID_ARGUMENT: request deadline field is not supported here")
    assert solver._is_overloaded_error(erro) is False


def test_400_com_deadline_no_corpo_nao_manda_modelo_para_o_banco(monkeypatch):
    """O pior efeito: um modelo saudavel suspenso por erro que era nosso."""
    erro = RuntimeError(
        "400 INVALID_ARGUMENT: request deadline field is not supported here")
    cliente, chamadas = _cliente({m: erro for m in solver.GEMINI_MODELS})
    monkeypatch.setattr(solver, "_get_client", lambda _k: cliente)
    with pytest.raises(RuntimeError):
        solver._gemini_call([], {}, "k", "grade")
    assert solver._BANCO == {}
    # So o primeiro modelo, com o retry local que o contrato ja previa.
    assert len(chamadas) == solver.GEMINI_TRIES_PER_MODEL


def test_status_do_atributo_vence_o_texto():
    """Excecao do SDK traz `code`; o texto nem precisa dizer o numero."""
    class ErroDoSDK(Exception):
        code = 429
    assert solver._categoria_do_erro(ErroDoSDK("overloaded")) == "limite_de_uso"


def test_timeout_sem_texto_ainda_e_tempo_esgotado():
    """`ReadTimeout` do httpx chega com `str(e)` vazio — so o nome da classe fala."""
    class ReadTimeout(Exception):
        pass
    assert solver._categoria_do_erro(ReadTimeout()) == "tempo_esgotado"
    assert solver._is_overloaded_error(ReadTimeout()) is True


# ── Rodizio: cada rodada ouve um modelo diferente ───────────────────────────
#
# `temperature=0.0` torna a chamada deterministica: mesma imagem, mesmo modelo,
# resposta identica byte a byte. Repetir sem trocar de modelo e' gasto puro.
# O mecanismo existia e estava ligado em grade, grade_fused, cartao_animal e
# bola; `_solve_imagem` ficou de fora — 4 rodadas de repeticao garantida, cada
# uma com screenshots e uma chamada.

def test_gemini_grid_aceita_rodizio():
    """Sem isto, `_solve_imagem` nao tem como pedir outro modelo."""
    import inspect
    assert "rodizio" in inspect.signature(solver._gemini_grid).parameters


def test_gemini_grid_repassa_o_rodizio(monkeypatch):
    visto = {}
    monkeypatch.setattr(solver, "_gemini_call",
                        lambda *a, **kw: visto.update(kw) or {"ok": 1})
    solver._gemini_grid(b"", "instrucao", "k", None, rodizio=2)
    assert visto.get("rodizio") == 2


def test_rodizio_gira_a_ordem_dos_modelos(monkeypatch):
    """Rodada N comeca no modelo N — e o que faz a repeticao valer alguma coisa."""
    primeiros = []

    def cliente_que_falha(_k):
        class C:
            class models:
                @staticmethod
                def generate_content(model, **_kw):
                    primeiros.append(model)
                    raise RuntimeError("503 unavailable")
        return C()

    monkeypatch.setattr(solver, "_get_client", cliente_que_falha)
    for rodada in range(3):
        # O banco de reservas precisa ser zerado A CADA iteracao: as falhas
        # simuladas suspendem modelos, e a rodada seguinte comecaria de uma
        # lista de ativos ja diferente — mediria o banco, nao o rodizio.
        solver._BANCO.clear()
        primeiros.clear()
        with pytest.raises(RuntimeError):
            solver._gemini_call([], {}, "k", "grid", rodizio=rodada)
        assert primeiros[0] == solver.GEMINI_MODELS[rodada % len(solver.GEMINI_MODELS)]


# ── 400 se auto-diagnostica: repete SEM os campos opcionais ─────────────────
#
# Timeline da RUN-74db9dba: `status=400 | categoria=requisicao_invalida` em
# gemini-3.1-flash-lite, nas duas tentativas, e o captcha nao foi resolvido.
# O corpo do erro nunca chega ao log (regra de higiene), entao a unica forma de
# saber QUAL campo a API recusa e tentar sem ele.
#
# O suspeito esta nomeado no docstring de `_make_config`: estes modelos
# respondem INVALID_ARGUMENT a valores de thinking_config que nao aceitam, e a
# lista de exclusao so cobre "2.0-flash". Manter lista estatica de quem suporta
# o que envelhece a cada release — a verdade vem da resposta.

def test_400_repete_sem_thinking_no_mesmo_modelo(monkeypatch):
    """Segunda tentativa vai sem thinking_config, e no MESMO modelo."""
    vistos = []

    def cliente(_k):
        class C:
            class models:
                @staticmethod
                def generate_content(model, contents=None, config=None):
                    tem_thinking = getattr(config, "thinking_config", None) is not None
                    vistos.append((model, tem_thinking))
                    if tem_thinking:
                        raise RuntimeError("400 INVALID_ARGUMENT: thinking_config")
                    class R:
                        text = '{"ok": 1}'
                    return R()
        return C()

    monkeypatch.setattr(solver, "_get_client", cliente)
    monkeypatch.setattr(solver, "THINKING_BUDGET", 4096)
    assert solver._gemini_call([], {}, "k", "grade_fused") == {"ok": 1}
    assert len(vistos) == 2, vistos
    assert vistos[0] == (vistos[1][0], True), "a 1a tentativa levava thinking"
    assert vistos[1][1] is False, "a 2a tentativa tinha de ir sem thinking"


def test_400_que_nao_e_thinking_ainda_falha(monkeypatch):
    """Se remover o opcional nao resolve, o 400 e de outra coisa — nao mascara."""
    def cliente(_k):
        class C:
            class models:
                @staticmethod
                def generate_content(**_kw):
                    raise RuntimeError("400 INVALID_ARGUMENT: outra coisa")
        return C()

    monkeypatch.setattr(solver, "_get_client", cliente)
    with pytest.raises(RuntimeError):
        solver._gemini_call([], {}, "k", "grade_fused")


def test_make_config_sem_opcionais_omite_thinking(monkeypatch):
    monkeypatch.setattr(solver, "THINKING_BUDGET", 4096)
    com = solver._make_config({}, solver.GEMINI_MODELS[0])
    sem = solver._make_config({}, solver.GEMINI_MODELS[0], sem_opcionais=True)
    assert getattr(com, "thinking_config", None) is not None
    assert getattr(sem, "thinking_config", None) is None
