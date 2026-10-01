# Estratégia, Bozó e Cadeias de Markov

> Estado: **pauta aprovada pelo Gabs em 2026-10-01**. Próxima etapa: exploração de abordagens.

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
