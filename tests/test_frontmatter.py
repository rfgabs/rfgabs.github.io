"""Agentes e skills com frontmatter inválido somem do Claude Code sem aviso."""

from pathlib import Path

import pytest
import yaml

RAIZ = Path(__file__).resolve().parents[1]
ARQUIVOS = sorted(RAIZ.glob(".claude/agents/*.md")) + sorted(RAIZ.glob(".claude/skills/*/SKILL.md"))


@pytest.mark.parametrize("arquivo", ARQUIVOS, ids=lambda p: p.parent.name if p.name == "SKILL.md" else p.stem)
def test_frontmatter_valido(arquivo):
    texto = arquivo.read_text(encoding="utf-8")
    assert texto.startswith("---\n"), "frontmatter precisa abrir na primeira linha"
    meta = yaml.safe_load(texto.split("---\n")[1])
    assert isinstance(meta, dict)
    assert meta.get("name") and meta.get("description")


def test_encontrou_arquivos():
    assert len(ARQUIVOS) >= 5
