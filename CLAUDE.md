# Data Overdrive — instruções para o Claude

Blog do Gabs (Quarto, publicado em https://rfgabs.github.io a partir de
`docs/`) sobre matemática e equacionamento em jogos.

## Como um post é produzido

Use a skill **`/novo-post <tema>`** — ela é o supervisor e orquestra três
subagentes (`.claude/agents/`):

| Agente | Faz | Ferramentas |
|---|---|---|
| `matematico` | **explora** ≥ 3 abordagens em níveis diferentes (N1 probabilidade → N5 RL); o Gabs escolhe; depois **executa** a escolhida | `tools/CATALOGO.md`: `mathbox` (SymPy), `mdp` (DP), `rl` (Q-learning), `nashpy`, `statsmodels`, `networkx` |
| `validador` | confere cada afirmação de forma independente, **sem ler a derivação** | `tools/montecarlo.py` (numpy/scipy), `mdp`/`rl` com modelo próprio |
| `escritor` | escreve o `index.qmd` no tom do guia | `.claude/estilo/guia-de-estilo.md` |

Depois que o Gabs edita o post, **`/aprender-estilo <slug>`** atualiza o
guia de estilo a partir da diferença entre rascunho e versão final.

## Layout de um post

```
posts/<slug>/
  index.qmd                 ← o post (draft: true até o Gabs publicar)
  _work/                    ← ignorado pelo Quarto (prefixo _)
    pauta.md                ← supervisor + decisões do Gabs
    abordagens.md           ← matemático (exploração): opções para o Gabs escolher
    modelo.md, derivacao.py ← matemático (execução)
    resultados/             ← saídas caras (treinos de RL, buscas) + como regerá-las
    claims.yaml             ← contrato: afirmações e status
    validacao.py, figuras.py← validador
    rascunho-escritor.qmd   ← cópia do que o escritor entregou
```

`uv run python tools/claims.py status|gate|set posts/<slug>` controla o ciclo
proposta → derivada → validada/divergente. O escritor só começa com `gate`
LIBERADO.

## Regras invariáveis

- **Números no post vêm de código.** Expressões inline `` `{python} x` ``
  alimentadas por `tools/` — nunca digitados. `execute-dir: project` faz
  `sys.path.insert(0, "tools")` funcionar em qualquer post.
- **Exato no matemático, semente fixa no validador** (`montecarlo.SEED`).
- **Publicar é do Gabs:** não remover `draft: true`, não dar push na `main`
  sem pedido explícito.

## Ambiente e render

```bash
uv sync                  # Python + Quarto (pacote quarto-cli)
uv run pytest            # testes das ferramentas
uv run quarto preview    # servidor local
uv run quarto render     # gera docs/ — commitar docs/ e _freeze/ junto
```

- Posts usam `freeze: true` (`posts/_metadata.yml`): o resultado computado
  fica em `_freeze/` e precisa ser commitado; para recalcular um post,
  `uv run quarto render posts/<slug>/index.qmd`.
- **O Quarto ignora arquivos dentro de diretórios ocultos.** Numa worktree em
  `.claude/worktrees/...` ele não encontra nada. Renderize copiando o projeto
  (sem `.venv`) para um diretório temporário sem ponto no caminho, com o
  `.venv/Scripts` da worktree no PATH, e traga `docs/` e `_freeze/` de volta.
- `core.autocrlf=true`: diffs de `docs/` que somem com `--ignore-cr-at-eol`
  são só quebra de linha.
