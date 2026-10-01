# Data Overdrive — rfgabs.github.io

Blog de ciência e tecnologia (Quarto), publicado pelo GitHub Pages a partir de `docs/`.

## Ambiente

O ambiente Python (cálculo + Quarto) é gerenciado pelo [uv](https://docs.astral.sh/uv/).
O Quarto vem do pacote `quarto-cli`, então não precisa de instalação no sistema.

```bash
uv sync                    # cria .venv com sympy, numpy, scipy, pandas, matplotlib, jupyter, quarto
uv run quarto preview      # servidor local com recarregamento
uv run quarto render       # gera o site em docs/
```

> O Quarto ignora arquivos dentro de diretórios ocultos. Renderizar a partir de um
> caminho como `.claude/worktrees/...` não encontra nenhum `.qmd` — renderize a partir
> do checkout principal (ou de uma cópia fora de diretórios com ponto).

## Estrutura

- `posts/<slug>/index.qmd` — um post por pasta. Use `draft: true` enquanto não estiver pronto
  (fica fora da listagem, da busca e do sitemap).
- `docs/` — saída renderizada; é o que vai para o ar. Não editar à mão.
