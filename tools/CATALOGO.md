# Catálogo de abordagens e ferramentas

Mapa — não cerca. O matemático usa este catálogo para propor abordagens em
níveis diferentes de complexidade, mas **pode e deve** propor métodos que não
estão aqui quando a pergunta pedir (e criar a ferramenta que faltar).

## Níveis

| Nível | Família | Responde perguntas do tipo | Ferramentas |
|---|---|---|---|
| **N1** | Probabilidade elementar e estatística descritiva | "qual a chance de…", "quanto vale em média…" | `mathbox`: `dice_pattern_probability`, `sum_distribution`, `expected_value`; `scipy.stats` |
| **N2** | Modelo exato estrutural | "quantas jogadas até…", "qual a chance de terminar em…" | `mathbox`: Markov exato (`absorbing_analysis`, `stationary_distribution`), `verify_identity`; SymPy (`rsolve`, séries, funções geradoras); `networkx` (grafo de estados) |
| **N3** | Simulação Monte Carlo | o que não tem forma fechada; distribuições inteiras; sensibilidade a variações de regra | `numpy`; `montecarlo` (IC, qui-quadrado, semente fixa) |
| **N4** | Otimização de decisões | "qual a melhor estratégia", "quanto custa jogar mal" | `mdp`: `MDP`, `value_iteration`, `policy_evaluation(_exact)`; `scipy.optimize` para parâmetros contínuos |
| **N5** | Aprendizado | "um agente descobre a estratégia sozinho? em quantas partidas?", jogos grandes demais para N4 | `rl`: `q_learning`, `mc_control`, `rollout_returns`, curvas de aprendizado; extra `uv sync --extra rl` (gymnasium + stable-baselines3) para deep RL |

## Transversais

| Família | Quando | Ferramentas |
|---|---|---|
| Teoria dos jogos | mais de um jogador decidindo; blefe; estratégias mistas | `nashpy` (equilíbrios de Nash, jogos de soma zero) |
| Inferência estatística | há dados reais (partidas registradas, resultados de torneio) | `statsmodels` (regressão, testes), `scipy.stats`, `pandas` |
| Grafos | estrutura de estados, caminhos, alcançabilidade | `networkx` |

## Fora do catálogo (exemplos do que também vale propor)

Teoria da informação (quanto uma jogada revela), inferência bayesiana
(atualizar crença sobre a mão do adversário), critério de Kelly (quanto
apostar), busca em árvore / MCTS, algoritmos genéticos para estratégias,
análise de sensibilidade de regras ("e se o dado tivesse 8 faces?"),
modelos de Elo/TrueSkill. Se precisar de biblioteca nova, diga qual e por quê
— o supervisor adiciona ao `pyproject.toml`.

## Como os níveis se conferem

Níveis altos ganham credibilidade quando reproduzem um nível baixo num caso
pequeno: o Q-learning (N5) deve chegar perto da política ótima da programação
dinâmica (N4) num jogo reduzido; a DP deve bater com a fórmula exata (N2) onde
ela existe. Proponha essas pontes — elas viram afirmações fáceis de validar e
bons momentos no post.

## Armadilhas conhecidas

- **RL tabular:** use `alpha="visitas"` (padrão) e `q_init` otimista; passo
  constante deixa a política escolhendo pelo ruído em decisões apertadas.
  Treine com `as_env(exploring_starts=True)` quando a política tende a nunca
  visitar a região onde a decisão importa; **avalie** com o ambiente normal.
- **Resultados de RL são aleatórios:** afirme sobre várias sementes
  (média e dispersão), nunca sobre uma execução.
- **MDP grande:** `MDP` enumera todos os estados alcançáveis; acima de
  ~10⁶ estados, abstraia o estado (simetrias, ordenar dados) ou vá para N5.
