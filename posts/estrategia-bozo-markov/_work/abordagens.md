# Abordagens: Estratégia, Bozó e Cadeias de Markov

> Modo exploração (matemático), 2026-10-01. Regras: pauta.md, decisões D1–D6.
> Números de viabilidade: `_work/scratch/viabilidade.py` (descartável, em
> float64, ainda **não** é derivação; tudo que for para o post será refeito em
> `derivacao.py`).

## Leitura das regras para o modelo (o que muda o tamanho do estado)

- **Estado da cartela = conjunto de casas livres. Confirmado.** A pontuação é
  a soma das casas, não há bônus da seção superior nem coringa, e os pontos de
  uma casa dependem só de (casa, dados finais, se é de boca). A recompensa
  futura depende portanto só de quais casas sobram:
  $E^*(S) = \mathbb{E}\big[\max_{b\in S}\{\text{pts}(b,\text{dados}) + E^*(S\setminus b)\}\big]$
  (com a otimização dos lançamentos dentro da rodada). São $2^{10} = 1024$
  cartelas (1023 não terminais). O total já marcado **não** entra no estado
  (isso só mudaria se o objetivo fosse vencer um adversário, fora do escopo).
- **D6 + casas de número:** toda casa livre é sempre marcável (número com 0,
  combinação inválida = riscar com 0). A ação "marcar" tem sempre $|S|$
  opções e não existe o caso especial "sem opção".
- **Boca não exige informação extra.** O +5 só vale se o jogador marca logo
  após o 1º lançamento, isto é, quando ainda restam 2 lançamentos. Esse
  contador já faz parte do estado intra-rodada. Guardar os 5 dados e "relançar
  zero" é o mesmo que parar.
- **D3 (dado separado volta ao copo):** a ação depois de cada lançamento é
  escolher um submulticonjunto qualquer dos dados atuais para guardar. Não é
  preciso lembrar o que foi guardado antes.
- **Dados como multiconjunto:** 252 resultados sem ordem (contra $6^5=7776$
  ordenados, 31× menos). Guardas possíveis (multiconjuntos de 0 a 5 dados):
  462. A partir de um lançamento há em média 17,3 guardas distintas (21,4
  ponderando pela probabilidade; máximo 32).
- **D2 (BAIXO às cegas):** não muda nada no modelo, porque $7-X$ também é
  uniforme. A variante "depois de ver" (V3) entra como uma ação extra: trocar
  o multiconjunto $d$ por $7-d$.

### Contagem do MDP completo (Bozó solo)

| Componente | Contagem |
|---|---|
| Cartelas (casas livres) | 1024 (1023 não terminais) |
| Estados de decisão (cartela, nº do lançamento 1–3, dados) | 1023 × 3 × 252 = **773.388** |
| Nós de acaso (cartela, guarda, antes do 2º/3º lançamento) | 1023 × 2 × 462 = **945.252** |
| Total (com início de rodada e terminal) | **≈ 1,72 milhão** |
| Transições não nulas dos nós de acaso | ≈ 8,9 milhões |

**Viabilidade:** não é preciso enumerar esse grafo. A DP específica por
rodada, que percorre as cartelas da menor para a maior com duas tabelas
pré-computadas (guarda → distribuição do multiconjunto final, 462×252; e
submulticonjuntos de cada lançamento, 252×32), resolve o jogo inteiro em
**0,11 s** em numpy (o pré-cálculo leva 0,04 s). Em Python puro, sem numpy, a
estimativa é de dezenas de segundos, ainda viável. O `tools/mdp.py` genérico
chegaria perto do limite `max_states = 2·10⁶` e guardaria cerca de 9·10⁶
transições como tuplas Python (na ordem de 1 GB). Seria lento (minutos a horas
de enumeração mais iteração de valor) e não ganha nada, porque o jogo é
acíclico por rodada. **Recomendo uma DP específica** e o `mdp.py` só para os
jogos reduzidos (ponte com RL).

### Resultados preliminares (float64, para dimensionar, não para publicar)

| Quantidade | Valor |
|---|---|
| $E^*$ (jogo completo, política ótima) | **146,73** |
| Mesmo jogo **sem** bônus de boca | 145,10 → a boca vale ≈ **1,63** ponto |
| Com BAIXO **depois de ver** (V3, só no 1º lançamento, inverte todos) | 151,26 → valeria ≈ **4,53** pontos |
| Gulosa ingênua (persegue a face mais frequente, para ao fazer combinação, marca a casa que dá mais pontos agora, risca a de menor índice) | **120,5 ± 0,45** (MC, 20 mil partidas) → custa ≈ **26** pontos |
| Ponte: só General livre, DP | 2,30529 = $50\cdot P(\text{general em 3}) + 5/1296$ ✓ |
| Ponte: só Ás livre, DP | 2,1065 = $5\,(1-(5/6)^3)$ ✓ |

Valor de cada casa jogada **sozinha** numa rodada (DP com $|S|=1$): Ás 2,11 ·
Duque 4,21 · Terno 6,32 · Quadra 8,43 · Quina 10,53 · Sena 12,64 · Fú 7,45 ·
Seguida 7,99 · Quadrada 11,13 · General 2,31. A soma (73,1) fica muito abaixo
de $E^*$ (146,7): jogar dez rodadas com flexibilidade vale o dobro de dez
rodadas "com alvo fixo". Isso é um bom gancho.

---

## A. Uma rodada, um alvo — nível N1–N2

**Pergunta que responde:** reformulada como "qual a chance de fazer cada
combinação numa rodada, e quanto vale cada casa se for a única que resta?".
É o tijolo do MDP.
**Método:** probabilidades exatas do 1º lançamento por contagem (General
6/7776, Quadrada 150/7776, Fú 300/7776, Seguida 240/7776), que dão a chance de
cada combinação "de boca". Para três lançamentos perseguindo um alvo fixo:
cadeia de Markov absorvente no "nº de dados já certos" (General: estados 1…5,
o clássico $P\approx 0{,}04603$ do Yahtzee; casa de número: binomial,
$\mathbb{E}=5k(1-(5/6)^3)$). Fú, Quadrada e Seguida com cadeias pequenas sobre
o "perfil" dos dados, com a política natural para cada alvo.
**Ferramentas:** existentes: `mathbox.dice_pattern_probability`,
`absorbing_analysis`, `transition_matrix`, `R`. Nenhuma a criar.
**O que o leitor ganha:** tabela "chance de fazer cada combinação de boca / em
até 3 lançamentos" e "pontos esperados de cada casa sozinha"; uma matriz de
transição 5×5 desenhável (grafo da cadeia do General).
**Custo:** segundos · explicar: baixa.
**Riscos e limites:** não responde à pergunta central (escolher entre casas).
Para Fú e Seguida, a política "ótima para um alvo" não é óbvia; a cadeia exige
cuidado ou se usa a DP com $|S|=1$ (ver pontes).
**Como validar:** Monte Carlo direto da rodada com a mesma política de alvo;
força bruta sobre os $6^5$ resultados no 1º lançamento.

## B. Jogadores de mesa simulados — nível N3

**Pergunta que responde:** "quantos pontos fazem as estratégias intuitivas, e
como se distribuem?" É a linha de base do "jeito intuitivo" da pauta.
**Método:** Monte Carlo de 2–4 heurísticas explícitas, por exemplo (i) a
gulosa acima; (ii) gulosa que risca primeiro General/Ás em vez da menor
índice; (iii) "rodada ótima míope", que maximiza os pontos **desta** rodada
sem olhar o futuro (é a DP de C com $E^*(S\setminus b)=0$, ótima dentro da
rodada e cega para a cartela). Distribuição da pontuação final, média com IC,
frequência de cada casa preenchida.
**Ferramentas:** existentes: `montecarlo` (IC, semente fixa), numpy. A criar:
simulador vetorizado do Bozó (o protótipo em Python puro leva 42 s para 20 mil
partidas; vetorizado, < 5 s para 10⁶).
**O que o leitor ganha:** "jogando como a maioria joga você faz ~120 pontos" e
o histograma, mais o contraste míope × ótimo, que isola o **valor de
planejar a cartela** (separado do valor de rolar bem).
**Custo:** segundos a 1 min · explicar: baixa.
**Riscos e limites:** "intuitivo" é arbitrário. O número depende de como a
heurística é definida, e precisa de uma definição verificável e defensável
(pergunta para o Gabs).
**Como validar:** as médias das heurísticas também saem **exatas** pela
mesma recursão de C (avaliação de política, sem o max), então Monte Carlo ×
exato é uma ponte gratuita.

## C. O MDP completo resolvido por programação dinâmica — nível N4 (pedido do Gabs)

**Pergunta que responde:** a pergunta central, como está: política ótima e
$E^*$.
**Método:** decomposição por rodada (estados "pós-marcação"). Para cada
cartela $S$, em ordem crescente de $|S|$:
$V_3(d)=\max_{b\in S}[\text{pts}(b,d)+E^*(S\setminus b)]$;
$V_2(d)=\max\big(V_3(d),\ \max_{k\subseteq d}\sum_{d'}T(k,d')V_3(d')\big)$;
$V_1(d)=\max\big(\max_b[\text{pts}^{\text{boca}}(b,d)+E^*(S\setminus b)],\ \max_{k\subseteq d}\sum_{d'}T(k,d')V_2(d')\big)$;
$E^*(S)=\sum_d P_0(d)V_1(d)$. Daí saem:
- $E^*$ e a política (guarda e marcação para cada situação);
- **distribuição exata da pontuação final** sob a política ótima (e sob a
  gulosa), propagando para frente a distribuição de (cartela, pontos
  acumulados). Os pontos cabem em 0…~350, então é barato;
- **probabilidade de cada casa terminar preenchida com pontos / riscada**, e
  em que rodada, pela mesma propagação para frente;
- **valor das regras:** refazer a DP sem boca (−1,63), com BAIXO depois de
  ver (+4,53), e uma variante de V1 ou V5 se quiser;
- decisões contraintuitivas: busca automática por situações em que a ótima
  discorda da gulosa com maior perda (ex.: riscar General cedo, guardar par em
  vez de trinca, marcar Sena baixa para preservar Seguida).

**Ferramentas:** existentes: `mdp.policy_evaluation_exact` só para casos
reduzidos. A criar: `bozo_dp` (numpy, ~100 linhas). Candidato a ir para
`tools/` como `dice_mdp.py` genérico ("dados + cartela de casas aditivas"),
reaproveitável em Yahtzee/General/Bozó, com teste pela ponte do General.
**O que o leitor ganha:** "o Bozó jogado com perfeição vale ≈146,7 pontos;
jogar 'no instinto' custa ≈26"; o histograma ótima × gulosa; as barras "chance
de completar cada casa"; o "mapa de decisão" (para um lançamento e uma cartela,
o que guardar); o tamanho do MDP (1,7 milhão de estados, resolvido em 0,1 s,
contra Yahtzee com bônus).
**Custo:** < 1 s a DP; minutos com as análises para frente · explicar:
média–alta (a decomposição por rodada é o ponto didático).
**Riscos e limites:** exatidão. Em float64, $E^*$ fica com erro ~1e-12. Uma
versão **racional exata** (sympy/Fraction) teria denominadores até $6^{150}$
e ~10⁷ operações; acho viável em minutos, mas não testei. Desempates entre
ações de mesmo valor tornam a política não única, e a afirmação de política
tem de dizer "uma política ótima" e fixar o desempate. O "mapa de decisão"
completo tem 773 mil entradas; o post precisa escolher fatias (ex.: cartela
cheia, 1º lançamento).
**Como validar:** (1) Monte Carlo de 10⁶ partidas seguindo a política
exportada (tabela) → média dentro do IC de $E^*$; (2) implementação
independente do validador pela força bruta do `mdp.py` num jogo reduzido (3–4
casas); (3) pontes de A ($|S|=1$); (4) invariantes: $E^*$ monotônico em $S$;
$E^*(\text{sem boca}) \le E^* \le E^*(\text{BAIXO depois})$.

## D. Um agente aprende a jogar Bozó — nível N5

**Pergunta que responde:** "um agente que só joga, sem conhecer as
probabilidades, descobre a estratégia? Em quantas partidas? Quão perto do
ótimo de C chega?"
**Método:** duas camadas.
- **D1 — tabular num Bozó reduzido** (3–4 casas, ex.: Sena, Fú, Seguida,
  General): ≈ 7–15 cartelas × (756 + 924) estados ≈ 10–25 mil estados.
  Q-learning / MC control do `rl.py` sobre o `mdp.py`, com ≥ 5 sementes e
  curva de aprendizado com a reta do ótimo exato.
- **D2 — jogo completo, aprendendo só o valor da cartela:** o agente resolve
  a rodada exatamente, mas *estima* os 1024 valores $\hat E(S)$ por TD a partir
  de partidas jogadas (aprendizado por afterstates, como no TD-Gammon). Isso
  converge em poucas dezenas de milhares de partidas e mostra quanto do
  "segredo" está só no valor das casas. (Q-learning tabular no jogo completo,
  ~1,7 M estados × ~21 ações ≈ 3,6·10⁷ pares, não converge em tempo razoável
  em Python; deep RL com `--extra rl` é possível, mas caro e pouco
  explicável.)

**Ferramentas:** existentes: `rl.q_learning`, `mc_control`, `rollout_returns`,
`MDP.as_env`. A criar: o loop TD em afterstates de D2 (pequeno) e um ambiente
rápido do Bozó.
**O que o leitor ganha:** a curva "pontos por partida × partidas jogadas",
subindo da gulosa (~120) em direção ao ótimo (146,7). Nela se vê quantas
partidas um humano levaria para "aprender" o jogo.
**Custo:** D1 minutos; D2 minutos a ~1 h (Python), ou rápido se reaproveitar o
núcleo numpy de C · explicar: média.
**Riscos e limites:** convergência ruidosa em decisões apertadas (catálogo:
`alpha="visitas"`, `q_init` otimista); D2 não é "aprender do zero" (a rodada é
resolvida), e isso precisa ficar dito. Resultados aleatórios: afirmar sobre
média e dispersão de ≥ 5 sementes.
**Como validar:** distância para o ótimo exato (conhecido por C) em várias
sementes; o validador reavalia a política aprendida por Monte Carlo
independente.

---

## Combinações

- **A → C → B (recomendada para a pergunta da pauta):** A dá a intuição e os
  números de uma rodada; C resolve o jogo e responde a tudo dos "Resultados
  esperados" 1–5; B (com as médias exatas via C) é a linha de base "intuitiva"
  e o contraste míope × ótimo.
- **A + C + D1 (versão "Markov → MDP → RL"):** o post sobe a escada inteira. O
  RL aparece como epílogo num Bozó reduzido em que o ótimo exato é conhecido.
- **C + D2:** foco em "onde mora a estratégia": quase todo o ganho sobre a
  gulosa está em saber quanto vale cada casa livre (1024 números).

## Pontes entre níveis

1. **DP com uma casa livre = cadeia de Markov (A):** só General →
   $50\,P_3+5/1296 = 2{,}30529$ (conferido em float); casas de número →
   $5k(1-(5/6)^3)$ (conferido). É afirmação exata e fácil de validar.
2. **Recursão específica (C) = `mdp.py` genérico** num Bozó de 3–4 casas, com
   `value_iteration` e `policy_evaluation_exact` em racionais.
3. **Exato × Monte Carlo** para a ótima e para cada heurística de B.
4. **RL (D1) → ótimo de C** no jogo reduzido, com várias sementes.
5. **BAIXO às cegas = sem BAIXO:** a DP com e sem a opção dá o mesmo $E^*$
   (identidade trivial, boa curiosidade, cf. D2).

## Perguntas para o Gabs

1. **"Jeito intuitivo":** qual heurística representa como se joga na mesa?
   Proponho a gulosa descrita acima (persegue a face mais repetida, para ao
   fazer combinação, marca o que dá mais pontos agora). Ela tem um detalhe
   arbitrário, a casa a riscar quando tudo dá zero. Quer 1 heurística ou 2–3
   (inclusive a míope)?
2. **BAIXO depois de ver (comparação, V3):** assumi que inverte **todos** os
   dados, só uma vez e só logo após o 1º lançamento. É isso, ou pode ser pedido
   em qualquer lançamento?
3. **Exato ou numérico:** aceita $E^*$ em float com tolerância declarada
   (ex.: 1e-9) se a versão racional exata se mostrar lenta? Posso testar a
   racional primeiro na execução.
4. **Escopo do RL:** entra no post (D1 reduzido, D2 completo) ou fica para um
   post futuro?

**Preferência (uma linha):** C como espinha, com A como porta de entrada e B
como contraste. A surpresa de que dez casas "sozinhas" valem 73 e juntas
valem 147 me parece o melhor momento do post; D1 é bônus barato.
