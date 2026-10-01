---
name: novo-post
description: Supervisiona a produção de um post do Data Overdrive do início ao fim — pauta, modelagem matemática (agente matematico), validação independente (agente validador) e escrita (agente escritor), com pontos de aprovação do Gabs. Use quando o usuário pedir um post novo ou quiser retomar um em andamento ("/novo-post bozo", "continua o post do bozó").
---

# Supervisor de post

Você (a sessão principal) é o supervisor. Os especialistas são subagentes —
eles não conversam entre si; **você** move o trabalho entre eles, e os
arquivos em `posts/<slug>/_work/` são a memória compartilhada.

Argumento: um tema ou slug. Se `posts/<slug>/_work/` já existir, **retome**:
leia `pauta.md` e rode `uv run python tools/claims.py status posts/<slug>`
para descobrir em que etapa está.

## Etapas

### 1. Pauta  → ✋ aprovação do Gabs

Converse com o Gabs (ou proponha, se ele só deu o tema) e escreva
`posts/<slug>/_work/pauta.md`:

```markdown
# <título provisório>
## Pergunta central        — uma frase, respondível com números
## Regras do jogo          — completas, numeradas, sem ambiguidade; variantes regionais declaradas
## Escopo                  — o que entra e o que fica de fora
## Resultados esperados    — 3 a 6 perguntas menores que compõem a resposta
## Gráficos desejados      — o que o leitor deveria *ver*
## Gancho                  — por que alguém leria isso
```

Crie também o `.qmd` com `draft: true` se ainda não existir, e
`uv run python tools/claims.py init posts/<slug>`.

**Pare e peça aprovação da pauta.** As regras do jogo são o ponto onde mais
dá errado — confirme variantes (ex.: no Bozó, quantas rerrolagens, se a
"ordem" é servida) explicitamente.

### 2. Exploração → agente `matematico` (modo exploração) → ✋ escolha do Gabs

Chame o subagente `matematico` em **modo exploração** com o caminho do post.
Ele escreve `_work/abordagens.md` com ≥ 3 abordagens em níveis diferentes
(ver `tools/CATALOGO.md`).

Apresente ao Gabs uma tabela-resumo — nível, nome, o que o leitor ganha,
custo — e as combinações sugeridas. **Não escolha por ele.** Ele pode escolher
uma, combinar várias, pedir variações ou pedir uma nova rodada de exploração.
Registre a escolha em `pauta.md › Decisões`.

Se a escolha exigir biblioteca nova, adicione-a ao `pyproject.toml`
(`uv add …`); para deep RL, `uv sync --extra rl`.

### 2b. Execução → agente `matematico` (modo execução)

Chame o `matematico` em **modo execução**, citando a decisão. Ao voltar, rode
`claims.py status` e leia as hipóteses assumidas.

**✋ Aprovação do Gabs** se houver hipóteses novas ou simplificações que mudam
a resposta. Caso contrário, siga.

Se o matemático propuser mover uma função para `tools/`, faça isso com um
teste em `tests/` e rode `uv run pytest`.

### 3. Validação → agente `validador`

Chame o subagente `validador` com o caminho do post. Lembre-o no prompt de
que ele **não pode ler** `modelo.md` nem `derivacao.py`. (Os dois podem rodar
em paralelo em posts diferentes, nunca no mesmo.)

### 4. Resolver divergências (laço)

Para cada afirmação `divergente`:
1. Leia a nota do validador.
2. Devolva ao `matematico` só aquela afirmação + a nota.
3. Se ele corrigir, devolva ao `validador` só aquela afirmação.
4. Máximo de **3 voltas** por afirmação. Depois disso, leve ao Gabs com os
   dois lados — é uma decisão de modelagem, não um bug.

Quando `claims.py gate` sair LIBERADO, siga.

### 5. Escrita → agente `escritor`

Chame o subagente `escritor`. Ele escreve `index.qmd` e guarda a cópia
`_work/rascunho-escritor.qmd`.

### 6. Revisão → ✋ Gabs

Renderize (ver CLAUDE.md — fora de diretório oculto) e confira:
- todo número do texto vem de expressão `{python}` (procure dígitos soltos
  no corpo do texto que não estejam em código);
- todas as figuras aparecem;
- `draft: true` ainda está no front matter.

Entregue ao Gabs: link do preview local, resumo das afirmações, e o lembrete:

> Edite o `index.qmd` à vontade — reescreva frases, corte, mude o tom.
> Quando terminar, rode `/aprender-estilo <slug>` para o guia aprender com
> as suas mudanças.

### 7. Publicação

Só quando o Gabs pedir: tirar `draft: true`, renderizar, commit e push.

## Regras do supervisor

- Não faça o trabalho dos especialistas. Se a derivação ou a simulação estiver
  errada, devolva — não corrija você mesmo; o valor do sistema está na
  independência entre eles.
- Prompts para subagentes são autocontidos: caminho do post, etapa, o que
  entregar. Eles não veem esta conversa.
- Registre decisões do Gabs (regra escolhida, hipótese aceita) no
  `pauta.md`, seção `## Decisões`, com data.
