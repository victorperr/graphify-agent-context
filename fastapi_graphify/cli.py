"""Command-line entry point for FastAPI Graphify reports."""

from __future__ import annotations

import argparse

from .analyzer import FastAPIAnalyzer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a FastAPI report from Graphify output and source code.")
    parser.add_argument("source", help="FastAPI project directory")
    parser.add_argument("--graph", help="Optional Graphify graph.json")
    parser.add_argument("--output", default="FASTAPI_REPORT.md")
    parser.add_argument("--format", choices=("markdown",
                        "json"), default="markdown")
    args = parser.parse_args()
    analyzer = FastAPIAnalyzer(
        graph_path=args.graph) if args.graph else FastAPIAnalyzer()
    report = analyzer.analyze(args.source)
    analyzer.write_report(report, args.output, format=args.format)
    print(f"FastAPI report written to {args.output}")


if __name__ == "__main__":
    main()
