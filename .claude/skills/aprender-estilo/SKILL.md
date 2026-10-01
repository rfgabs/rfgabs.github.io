---
name: aprender-estilo
description: Atualiza o guia de estilo do blog a partir das edições que o Gabs fez sobre o rascunho do agente escritor. Use depois que o Gabs revisar um post ("/aprender-estilo bozo", "terminei de editar o post, aprende com isso").
---

# Aprender estilo a partir das edições

O Gabs não tem um corpo grande de textos de referência. Em vez disso, o tom
dele é aprendido **pela diferença** entre o que o escritor entregou e o que
ele publicou. Cada post refina `.claude/estilo/guia-de-estilo.md`.

## Passos

1. Localize os dois textos de `posts/<slug>/`:
   - `_work/rascunho-escritor.qmd` — o que o agente entregou;
   - `index.qmd` — a versão editada pelo Gabs.
   Se o rascunho não existir, não há o que comparar: avise e pare.

2. Gere o diff por parágrafo (ignore blocos de código e front matter):
   `git diff --no-index --word-diff=plain posts/<slug>/_work/rascunho-escritor.qmd posts/<slug>/index.qmd`

3. Classifique cada edição relevante (ignore correções de digitação e
   mudanças factuais):
   - **vocabulário** — palavras trocadas, gírias adicionadas/removidas;
   - **ritmo** — frases quebradas ou fundidas, parágrafos encurtados;
   - **voz** — pessoa (eu/a gente/você), humor, ironia, digressões pessoais;
   - **estrutura** — seções movidas, cortadas, criadas; onde a matemática entra;
   - **cortes** — o que ele removeu (explicação demais? floreio? clichê?).

4. Proponha ao Gabs as mudanças no guia, **cada uma com o exemplo
   antes → depois que a motivou**. Formato:

   ```
   + [voz] Usa "a gente" em vez de "nós".
       antes: "Nós podemos calcular..."  →  depois: "A gente consegue calcular..."
   ~ [ritmo] (reforça regra existente) Frases curtas na abertura.
   - [vocabulário] Remover "fascinante" da lista de permitidas — cortado 3x.
   ```

   Só registre como regra um padrão que apareceu **2+ vezes** ou que o Gabs
   confirmar. Edições isoladas vão para a seção *Observações* do guia, com o
   slug, para virarem regra se reaparecerem.

5. Com a aprovação dele, edite `.claude/estilo/guia-de-estilo.md`:
   atualize as seções, acrescente os pares antes→depois em *Exemplos*, e
   registre o post no *Histórico*.

## Princípios

- O guia descreve o Gabs, não um ideal de "boa escrita". Se ele prefere uma
  construção que um manual condenaria, ela vira regra.
- Exemplos concretos valem mais que adjetivos. "Tom descontraído" não ensina
  nada; "abre com uma situação de mesa de bar" ensina.
- Mantenha o guia curto (< ~200 linhas). Ao crescer, consolide regras
  parecidas e mova exemplos antigos para o fim.
