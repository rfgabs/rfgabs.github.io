"""Registro de afirmações de um post — o contrato entre os agentes.

Cada post tem `posts/<slug>/_work/claims.yaml`. O matemático cria e deriva as
afirmações, o validador marca o veredito, e o escritor só começa quando todas
estão validadas.

    uv run python tools/claims.py status posts/<slug>
    uv run python tools/claims.py gate   posts/<slug>     # sai com erro se algo não está validado
    uv run python tools/claims.py set    posts/<slug> C3 validada --nota "MC n=1e6, IC99 contém 0.0463"

Ciclo de vida de uma afirmação:
    proposta -> derivada -> validada
                         -> divergente -> (matemático revisa) -> derivada ...
    qualitativa: afirmações sem número (ex.: "a estratégia gulosa não é ótima")
                 são validadas por contraexemplo ou simulação comparativa.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import yaml

STATUSES = ["proposta", "derivada", "validada", "divergente"]

TEMPLATE = """\
# Afirmações do post. Ver tools/claims.py para o ciclo de vida.
pergunta: ""          # a pergunta central do post, em uma frase
claims: []
# Exemplo de item:
#  - id: C1
#    enunciado: "P(general servido) na primeira jogada"
#    tipo: probabilidade          # probabilidade | esperanca | distribuicao | identidade | qualitativa
#    valor: "1/1296"              # forma exata (sympy) — vazio para qualitativa
#    valor_num: 0.000771605
#    derivacao: "modelo.md#c1"    # onde está a derivação
#    status: derivada
#    validacao: null              # preenchido pelo validador: {metodo, n, estimativa, ic, nota}
#    historico: []
"""


def claims_path(target: str) -> Path:
    p = Path(target)
    if p.is_dir():
        p = p / "_work" / "claims.yaml"
    return p


def load(target: str) -> dict:
    p = claims_path(target)
    if not p.exists():
        raise SystemExit(f"{p} não existe — rode `claims.py init {target}`")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    data.setdefault("claims", [])
    return data


def save(target: str, data: dict) -> None:
    p = claims_path(target)
    p.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=100), encoding="utf-8")


def cmd_init(args) -> int:
    p = claims_path(args.target)
    if p.exists():
        print(f"{p} já existe")
        return 1
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(TEMPLATE, encoding="utf-8")
    print(f"criado {p}")
    return 0


def cmd_status(args) -> int:
    data = load(args.target)
    claims = data["claims"]
    if data.get("pergunta"):
        print(f"Pergunta: {data['pergunta']}\n")
    if not claims:
        print("(nenhuma afirmação)")
        return 0
    for c in claims:
        val = c.get("valor") or "—"
        print(f"{c['id']:>4}  {c.get('status', '?'):<10}  {val:<14}  {c.get('enunciado', '')}")
    counts = {s: sum(1 for c in claims if c.get("status") == s) for s in STATUSES}
    print("\n" + "  ".join(f"{s}: {n}" for s, n in counts.items() if n))
    return 0


def cmd_gate(args) -> int:
    data = load(args.target)
    pending = [c for c in data["claims"] if c.get("status") != "validada"]
    if not data["claims"]:
        print("BLOQUEADO: nenhuma afirmação registrada")
        return 1
    if pending:
        print("BLOQUEADO: afirmações não validadas:")
        for c in pending:
            print(f"  {c['id']} ({c.get('status')}): {c.get('enunciado')}")
        return 1
    print(f"LIBERADO: {len(data['claims'])} afirmações validadas")
    return 0


def cmd_set(args) -> int:
    if args.status not in STATUSES:
        raise SystemExit(f"status inválido: {args.status} (use {', '.join(STATUSES)})")
    data = load(args.target)
    for c in data["claims"]:
        if c["id"] == args.id:
            c.setdefault("historico", []).append(
                {"data": date.today().isoformat(), "de": c.get("status"), "para": args.status, "nota": args.nota or ""}
            )
            c["status"] = args.status
            save(args.target, data)
            print(f"{args.id}: {c['historico'][-1]['de']} -> {args.status}")
            return 0
    raise SystemExit(f"afirmação {args.id} não encontrada")


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("init", "status", "gate"):
        sp_ = sub.add_parser(name)
        sp_.add_argument("target", help="pasta do post ou caminho do claims.yaml")
    s = sub.add_parser("set")
    s.add_argument("target")
    s.add_argument("id")
    s.add_argument("status")
    s.add_argument("--nota")
    args = ap.parse_args(argv)
    return {"init": cmd_init, "status": cmd_status, "gate": cmd_gate, "set": cmd_set}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
