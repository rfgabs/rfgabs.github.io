# resultados/v2 — dados para as visualizações da v2

Gerados por `uv run python posts/estrategia-bozo-markov/_work/derivacao.py` (seção v2 em
`derivacao_v2.py`; ~7 min no total). Nada aqui é digitado à mão. JSON compacto, UTF-8.

Convenções comuns:

- **casas** `0..9` = Ás, Duque, Terno, Quadra, Quina, Sena, Fú, Seguida, Quadrada, General.
- **cartela** = conjunto de casas livres; como inteiro, bit `b` = casa `b` livre (`1023` = cartela vazia, início
  do jogo; `0` = fim de jogo).
- **mãos**: os 252 multiconjuntos de 5 dados na ordem de `maos` (`"11111"`, `"11112"`, …, `"66666"`).
- **código de ação** `a`: `a < 0` marca a casa `-(a+1)`; `a >= 0` guarda `guardas[a]` (string de faces;
  `""` = relança os 5) e relança o resto.
- **valores** = pontos esperados do resto do jogo (rodada atual incluída). Valores com 9 casas vêm de
  racionais exatos; `*_exato` traz a fração.
- **cola (ii)** = "2face_8_2+seg4x+quad_gen+alvo+doispar" (C46); definição em `claims.yaml › colas`.

| arquivo | conteúdo / esquema |
|---|---|
| `pergunte_ao_otimo.json` (~430 KB) | `casas`, `maos[252]`, `guardas[462]`, `cola` (texto), `criterio_cartelas`, `cartelas[21]`: `{livres:[b…], rodada, prob_otima, E_cartela, lancamentos[3]:{otima[252], v_otima[252], gulosa[252], perda_gulosa[252], cola[252], perda_cola[252]}}`. `perda_x[d]` = valor ótimo − valor da ação de x seguida de jogo ótimo (0 = a ação também é ótima). Cartelas: vazia + as mais prováveis sob a ótima com 9..1 casas livres (3,3,2,2,2,2,2,2,2). Valores a 4 casas. |
| `curva_colas.json` (~45 KB) | `E_otimo`, `E_gulosa`, `tetos_nao_executaveis` (rodada perfeita com 6 tabelas de preço; ótimo), `colas[38]`: `{id, precos, tabela_precos, regras, itens_cola, n_regras, valor, valor_exato, ganho_recuperado, pareto_itens, pareto_regras, texto}`; `emendas` (texto e itens); `forward_A`, `forward_B` (passos com os valores de todos os candidatos); `shapley` (cola (i) `sozinha_int` e (ii) `2face_8_2`); `precos_mq`; `precos_marginais_cheia`. |
| `diagnostico_lema.json` (~4 KB) | `formas.i` / `formas.ii`: `por_categoria` (float), `por_categoria_exato` (fração), `por_grupo` (guarda/parada/casa), `por_casas_livres`; `formas.i.pares_casa_gulosa_para_otima` = `[casa gulosa, casa ótima, pontos]`. Soma de cada forma = `gap` = E* − E_G (exato). |
| `iteracao_politica.json` (~2 KB) | `passos[6]`: `{passo, valor, valor_exato, max_dif_E_otimo, cartelas_com_E_igual_otimo}` (de 1023). |
| `distribuicao_por_general.json` (~11 KB) | `otima`/`gulosa` → `feito`/`nao_feito`: `{p, media, moda, min, max, dist}`; `dist[s]` = P(pontuação final = s **e** condição), s = 0..max. Soma das duas = distribuição de C20/C32. |
| `ordem_preenchimento.json` (~5 KB) | `otima`/`gulosa`/`cola_ii` → `P[b][t]` = P(casa b preenchida na rodada t+1) (linhas e colunas somam 1) e `rodada_media[b]`. |
| `preco_marginal.json` (~65 KB) | `preco[b][t]` = E[E*(S) − E*(S sem b) \| b livre na rodada t+1] sob a ótima; `E_cartela[1024]` = E*(S) por cartela (índice = máscara); `marginal[S]` = lista de E*(S) − E*(S sem b) para cada b livre em S, b crescente. |
| `replays.json` (~25 KB) | `partidas[3]`: `{semente, dados_sorteados[10][3][5], otima, gulosa, cola_ii}`; cada política: `{total, rodadas[10]:{casa, pontos, total, lancamentos:[{dados[5], acao, acao_otima, custo}]}}`. Dados comuns: dado relançado na posição i no lançamento r da rodada t recebe `dados_sorteados[t][r][i]`; guardas ocupam as primeiras posições com a face guardada. `custo` = valor ótimo − valor da ação tomada (seguida de jogo ótimo). Ilustrativo: não é afirmação. |

Para regerar: rode o `derivacao.py` (tudo é recalculado; não há cache na v2).
