# Estratégia, Bozó e Cadeias de Markov

> Estado: **v2 em andamento (2026-10-02)** — v1 escrita e validada (35 afirmações); v2 acrescenta formulação, narrativa, heurísticas e visualizações interativas. Próxima etapa: validação de C39–C59, depois escrita da v2.

## Pergunta central

Qual é a estratégia ótima para jogar Bozó — tratado como um Processo de
Decisão de Markov (MDP) de um jogador — e quantos pontos ela vale em média,
comparada com o jeito "intuitivo" de jogar?

## Regras do jogo

### Fonte de referência

**PIANO, D. L.; TOILLIER, J. S. *A Matemática do jogo Bozó*.** XXIV Semana
Acadêmica de Matemática, Universidade Estadual do Oeste do Paraná (Unioeste),
Cascavel, 2010. Seção 4.1 "Regras" e Figuras 4–5.
Original: <https://projetos.unioeste.br/cursos/cascavel/matematica/xxivsam/artigos/46.pdf>
(fora do ar em 2026-10-01) · arquivo:
<https://web.archive.org/web/20240510005431/https://projetos.unioeste.br/cursos/cascavel/matematica/xxivsam/artigos/46.pdf>

Por que esta fonte: é a única com autoria identificada e publicação acadêmica,
traz o regulamento completo (não só a lista de casas) com tabela de pontuação e
desenho do tabuleiro, e cita como base o material do MEC *A probabilidade do
Bozó* (PAULA; LOPES, Portal do Professor, 2009). A pontuação dela coincide com
a da Wikipédia (pt e en).

### Regras (texto da fonte, numerado; entre colchetes, o que a fonte não diz e precisa de decisão)

1. Cinco dados de seis faces, lançados de uma vez com um copo opaco.
2. Cada jogador faz até **três lançamentos por rodada**, podendo parar antes.
3. Após o 1º lançamento, o jogador **separa os dados que quiser** e relança só
   os outros; o mesmo vale após o 2º. [D3: pode devolver ao copo um dado
   separado antes?]
4. **BAIXO:** antes de levantar o copo (antes de ver), o jogador pode pedir
   BAIXO — valem as faces de baixo dos dados. [D2]
5. Ao fim da rodada, marca **uma casa** do tabuleiro ainda livre:

   | Casa | Requisito | Pontos | De boca (+5) |
   |---|---|---|---|
   | Ás, Duque, Terno, Quadra, Quina, Sena | — | soma dos dados com a face 1…6 (ex.: três 4 na Quadra = 12) | não |
   | Fú | três iguais + dois iguais | 20 | 25 |
   | Seguida | cinco faces em sequência [D5: 1-2-3-4-5 e 2-3-4-5-6] | 30 | 35 |
   | Quadrada | quatro iguais + um diferente [D4] | 40 | 45 |
   | General | cinco iguais | 50 | 55 [D1] |

6. **Boca:** combinação (Fú, Seguida, Quadrada, General) conseguida no
   **primeiro lançamento** da rodada ganha +5.
7. **Riscar:** quando não tiver opção de marcação, o jogador elimina uma casa
   livre, que fica com zero. [D6: pode riscar por escolha, mesmo tendo opção?]
8. O jogo termina quando todas as 10 casas estão preenchidas (10 rodadas).
   Vence a maior soma.

### Variantes conhecidas

| # | Ponto | Fonte de referência | Variante(s) | Onde aparece |
|---|---|---|---|---|
| V1 | Pontos das combinações | Fú 20 · Seguida 30 · Quadrada 40 · General 50 | Full 10 · Seguida 20 · Quadrada 30 · General 40 (+5 de boca) | [Trevo de 7 Folhas](https://trevo-7folhas.blogspot.com/p/bozo.html) |
| V2 | General de boca | vale 55 pontos | **vitória imediata** | [bozo.com.br](https://www.bozo.com.br/p/o-bozo.html), [nh8ano](https://nh8ano.blogspot.com/2009/06/regras-bozo.html), [Trevo](https://trevo-7folhas.blogspot.com/p/bozo.html), [Esporte Uai](https://esporteuai.com.br/glossario/como-jogar-bozo/) |
| V3 | BAIXO | pedido **antes** de ver os dados | pedido **depois** do 1º lançamento, vendo os dados | [Wikipedia (en)](https://en.wikipedia.org/wiki/Boz%C3%B3_(dice_game)); "embaixo" em [nh8ano](https://nh8ano.blogspot.com/2009/06/regras-bozo.html) |
| V4 | Boca vale para | só as 4 combinações | qualquer casa | [Trevo](https://trevo-7folhas.blogspot.com/p/bozo.html) |
| V5 | Riscar | só quando não há opção | a qualquer momento ("torar") | [Wikipedia (en)](https://en.wikipedia.org/wiki/Boz%C3%B3_(dice_game)) |
| V6 | Número de rodadas | 10 | 9 | [nh8ano](https://nh8ano.blogspot.com/2009/06/regras-bozo.html) (provável erro/variante local) |

Outras fontes consultadas: [Wikipédia (pt)](https://pt.wikipedia.org/wiki/Boz%C3%B3_(jogo)).

## Pontos que a fonte deixa em aberto

- **D1 — General de boca:** 55 pontos (fonte) ou vitória imediata (V2)? Para
  um MDP de um jogador que maximiza pontos, 55 é o natural; vitória imediata
  só faz sentido num modelo de partida contra adversários.
- **D2 — BAIXO:** pedido às cegas (fonte) é matematicamente irrelevante — a
  face de baixo de um dado justo também é uniforme (7 − x). Pedido depois de
  ver (V3) vira uma decisão real ("inverter todos os dados?") que muda a
  estratégia. Qual usamos?
- **D3 — Dados separados:** um dado separado pode voltar ao copo no lançamento
  seguinte (como no Yahtzee), ou fica travado?
- **D4 — Quadrada e Fú com cinco iguais:** a fonte diz "quatro iguais **mais
  um diferente**" e "duas iguais **mais** três iguais". Cinco iguais podem ser
  marcados como Quadrada ou Fú?
- **D5 — Seguida:** 1-2-3-4-5 e 2-3-4-5-6 (Wikipédia). A fonte só diz "em
  sequência".
- **D6 — Riscar por escolha:** pode marcar zero numa casa por estratégia, mesmo
  tendo outra opção válida? Uma casa de número (ex.: Ás sem nenhum 1) conta
  como "opção" com 0 pontos?

## Escopo

- Entra: Bozó **solo** (um jogador maximizando a pontuação esperada), regras
  acima, como MDP.
- Fica de fora (talvez num post futuro): partida contra adversários
  (maximizar chance de vencer ≠ maximizar pontos), variantes V1–V6 além de uma
  comparação pontual.

## Resultados esperados

1. Quantos estados tem o MDP do Bozó — e por que ele cabe num computador
   (o Yahtzee, com 13 casas e bônus, já foi resolvido).
2. Pontuação esperada jogando de forma ótima.
3. Quanto se perde com estratégias simples (ex.: "gulosa": marca sempre a casa
   que dá mais pontos agora).
4. Decisões contraintuitivas da política ótima (ex.: quando riscar o General
   cedo; quando guardar par em vez de trinca).
5. Valor de cada regra: quanto vale o bônus de boca; quanto valeria o BAIXO
   pedido depois de ver (se D2 = antes, como comparação).

## Gráficos desejados

- Distribuição da pontuação final: ótima × gulosa.
- Probabilidade de completar cada casa sob a política ótima.
- Um "mapa de decisão": dado um lançamento, o que a política ótima guarda.

## Gancho

O Bozó tem 10 casas e cabe inteiro num computador: dá para calcular a jogada
perfeita para qualquer situação — e comparar com o que fazemos na mesa.

## Decisões

**2026-10-01 — pauta aprovada** (Bozó solo, maximizar pontuação esperada; contra adversários fica para outro post).

**2026-10-01 — regras do modelo** (fonte: Piano & Toillier, 2010, com os pontos abaixo):

| # | Decisão | Origem |
|---|---|---|
| D1 | General de boca vale **55 pontos** (sem vitória imediata) | Gabs, = fonte |
| D2 | **BAIXO pedido antes de ver** — no modelo, não altera nenhuma probabilidade; entra no post como curiosidade (e a variante "depois de ver" pode aparecer como comparação) | Gabs, = fonte |
| D3 | Dado separado **pode voltar ao copo** no lançamento seguinte (o jogador escolhe livremente quais dados relançar a cada vez) | Gabs (a fonte é ambígua; = Yahtzee) |
| D4 | **Cinco iguais não valem** como Quadrada nem como Fú (leitura literal: "quatro iguais mais um diferente", "duas iguais mais três iguais") | Gabs, = fonte |
| D5 | Seguida aceita **1-2-3-4-5 e 2-3-4-5-6** | Wikipédia pt/en; aceito pelo Gabs com a pauta |
| D6 | **Riscar é livre:** qualquer casa livre pode receber zero a qualquer momento, mesmo havendo opção que pontue | Gabs (= V5 "torar"; a fonte só prevê riscar sem opção) |

Consequência de D4+D6: com cinco iguais, as opções são General, a casa de
número correspondente, ou riscar alguma casa.

**2026-10-01 — abordagem e escopo de cálculo** (ver `abordagens.md`):

| # | Decisão | Origem |
|---|---|---|
| E1 | Post segue **A + B + C + D1**: Markov de uma rodada (A), jogador guloso simulado (B), MDP completo por DP (C), RL tabular num Bozó reduzido comparado com a DP (D1) | Gabs |
| E2 | O "jeito intuitivo" é **só a heurística gulosa** (sem míope nem variantes) | Gabs |
| E3 | **BAIXO fora das contas.** Só pode ser pedido antes de ver; não entra no modelo nem em comparações (a variante "depois de ver" não é calculada) | Gabs |
| E4 | Valor ótimo do jogo completo pode ser **float64 com tolerância 1e-9** declarada; frações exatas nas pontes pequenas | Gabs |
| E5 | **RL (D1) fora deste post, por enquanto.** Afirmações C36–C38 arquivadas em `claims.yaml › arquivadas` (resultados e cache preservados para um post futuro). Post segue A + B + C | Gabs, 2026-10-02 |
| E6 | **Gulosa, quando nada pontua, risca a casa de número mais baixo livre** (Ás → Sena) = regra G2 do matemático; números C29–C34 inalterados. Sem casa de número livre: Fú → Seguida → Quadrada → General | Gabs (1ª parte), padrão provisório do supervisor (2ª parte), 2026-10-02 |

## Versão 2 (2026-10-02) — pedido do Gabs após ler o 1º rascunho

**Narrativa (F1):** "da mesa ao ótimo, e de volta à mesa" —
1. a mesa: jogamos no instinto; jogamos bem?
2. o modelo em degraus: cadeia de Markov → MDP (S, A, P, R) → equação de Bellman;
3. o ótimo (146,7) e as jogadas contraintuitivas;
4. o problema: a política ótima é uma tabela de ~773 mil decisões — ninguém joga com isso;
5. de volta à mesa: que heurísticas simples recuperam a maior parte do ganho? Que "cola" cabe num guardanapo?

**Formulação (F2), no corpo do texto** (intuição antes de cada equação; só derivações longas em callout):
- cadeia de Markov (sem decisão) → MDP (alguém escolhe a ação; (S, A, P, R));
- ponte: **fixar uma política num MDP gera uma cadeia de Markov** (é assim que se avalia a gulosa);
- equação de Bellman de **avaliação** (V^π, política fixa) × de **otimalidade** (V*, com max); programação dinâmica = resolver a de otimalidade de trás para frente.

**Heurísticas (F3):** dado que a ótima é impraticável sem cola, propor heurísticas executáveis à mesa, medir o valor exato de cada uma (avaliação de política) e a relação **complexidade da cola × pontos esperados**. Famílias: o matemático propõe (exploração), o Gabs escolhe.

**Visualizações (F4): interativas permitidas** (Observable JS / Plotly no Quarto, dados pré-computados em JSON; nada de DP no navegador). Ideias: "pergunte ao ótimo" (escolha os dados, veja a jogada ótima × heurísticas), curva complexidade × pontos, diagrama da cadeia do General, "preço" de cada casa ao longo da partida, decomposição da distribuição (ex.: General feito ou não — exige afirmação validada).

**Mantém-se:** decisões D1–D6, E1–E6 (RL fora, BAIXO fora das contas, gulosa com E6) e as 35 afirmações validadas.

**2026-10-02 — escolhas da v2** (ver `abordagens-v2.md`):

| # | Decisão | Origem |
|---|---|---|
| H1 | Entram **§0 (diagnóstico pelo lema da diferença de desempenho) + A (gulosa + emendas) + B (cola de preços) + C (rodada perfeita com preços, teto) + C′ (iteração de política em 5 passos)**. D (árvores) e E ficam fora | Gabs |
| H2 | **As duas colas são protagonistas:** a tabela "preço = valor da casa jogada sozinha" como origem da ideia (liga ao começo do post) e a regra "número = 2× a face, combinações 8, General 2" + 4 regras de guarda como versão de bolso | Gabs |
| H3 | **"Pergunte ao ótimo" interativo:** cartela vazia + ~20 cartelas típicas de meio/fim de jogo (~1 MB JSON) | Gabs |
| H4 | Eixo de complexidade = **"itens de cola"** (definição de `abordagens-v2.md`); curva com nº de regras como checagem | padrão provisório (supervisor) |
| H5 | Desempate das colas: entre casas empatadas em pontos − preço, a de menor índice | padrão provisório (supervisor) |
| H6 | Corrigir em `modelo.md` a conclusão de C28 ("diferença quase toda na marcação"): o lema mostra que as guardas respondem pela maior parte | supervisor (correção de erro) |
| H7 | Cola (i) usa **Quina 11** (arredondamento correto de C10–C14; 141,173) | padrão provisório (supervisor) |
| H8 | Melhor cola no texto = a de bolso (ii); busca local (141,29) citada só como prova de que há pouco a ganhar | padrão provisório (supervisor) |
| H9 | **Entram as duas surpresas:** riscar o General piora a gulosa sozinha mas ajuda depois de outras regras (interação); Shapley negativo do preço das casas de número → "dê preço às combinações" | Gabs, 2026-10-02 |
| H10 | "Pergunte ao ótimo" mantém as cartelas **mais prováveis** sob a ótima | Gabs, 2026-10-02 |
| H11 | Replays entram como "partidas de exemplo", não como resultado | padrão provisório (supervisor) |
