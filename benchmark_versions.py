#!/usr/bin/env python3
import argparse
import csv
import html
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


VERSION_RE = re.compile(r"\b\d+\.\d+\.\d+\b")
BENCHMARK_RE = re.compile(
    r"^(solve|grad): mean=(\d+\.\d+) ms best=(\d+\.\d+) ms over (\d+) runs$",
    re.MULTILINE,
)


@dataclass(frozen=True, order=True)
class Version:
    parts: tuple[int, int, int]

    @classmethod
    def parse(cls, value: str) -> "Version":
        match = VERSION_RE.fullmatch(value.strip())
        if not match:
            raise ValueError(f"unsupported version format: {value!r}")
        return cls(tuple(int(part) for part in value.split(".")))

    def __str__(self) -> str:
        return ".".join(str(part) for part in self.parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark script.py across matching jax/jaxlib versions."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only discover matching versions and print them",
    )
    parser.add_argument(
        "--min-version",
        type=Version.parse,
        help="inclusive lower bound for versions to benchmark",
    )
    parser.add_argument(
        "--max-version",
        type=Version.parse,
        help="inclusive upper bound for versions to benchmark",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results.csv"),
        help="output file for benchmark rows (default: benchmark_results.csv)",
    )
    return parser.parse_args()


def run_command(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=check, text=True, capture_output=True)


def discover_versions(package: str) -> list[Version]:
    result = run_command(
        [sys.executable, "-m", "pip", "index", "versions", package],
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        detail = stderr or stdout or f"pip exited with code {result.returncode}"
        raise RuntimeError(f"failed to discover {package} versions: {detail}")

    versions = {Version.parse(match) for match in VERSION_RE.findall(result.stdout)}
    if not versions:
        raise RuntimeError(f"found no stable versions for {package}")
    return sorted(versions)


def filter_versions(
    versions: list[Version],
    min_version: Version | None,
    max_version: Version | None,
) -> list[Version]:
    filtered = []
    for version in versions:
        if min_version is not None and version < min_version:
            continue
        if max_version is not None and version > max_version:
            continue
        filtered.append(version)
    return filtered


def discover_matching_versions(
    min_version: Version | None,
    max_version: Version | None,
) -> list[Version]:
    jax_versions = set(discover_versions("jax"))
    jaxlib_versions = set(discover_versions("jaxlib"))
    matching = sorted(jax_versions & jaxlib_versions)
    return filter_versions(matching, min_version, max_version)


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_header(output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=output_columns())
        writer.writeheader()


def append_row(output_path: Path, row: dict[str, str]) -> None:
    with output_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=output_columns())
        writer.writerow(row)


def output_columns() -> list[str]:
    return [
        "timestamp",
        "version",
        "status",
        "solve_mean_ms",
        "solve_best_ms",
        "solve_runs",
        "grad_mean_ms",
        "grad_best_ms",
        "grad_runs",
        "error_type",
        "error_detail",
        "stdout",
        "stderr",
    ]


def compact_text(content: str) -> str:
    return " ".join(content.split())


def escape_multiline_text(content: str) -> str:
    return html.escape(content.strip()).replace("\n", "&#10;")


def parse_benchmark_output(stdout: str) -> dict[str, str]:
    metrics = {
        "solve_mean_ms": "",
        "solve_best_ms": "",
        "solve_runs": "",
        "grad_mean_ms": "",
        "grad_best_ms": "",
        "grad_runs": "",
    }

    for label, mean_ms, best_ms, runs in BENCHMARK_RE.findall(stdout):
        metrics[f"{label}_mean_ms"] = mean_ms
        metrics[f"{label}_best_ms"] = best_ms
        metrics[f"{label}_runs"] = runs
    return metrics


def install_version(version: Version) -> subprocess.CompletedProcess[str]:
    return run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            f"jax=={version}",
            f"jaxlib=={version}",
        ],
        check=False,
    )


def benchmark_script() -> subprocess.CompletedProcess[str]:
    return run_command([sys.executable, "script.py"], check=False)


def benchmark_version(version: Version, output_path: Path) -> None:
    row = {
        "timestamp": timestamp(),
        "version": str(version),
        "status": "",
        "solve_mean_ms": "",
        "solve_best_ms": "",
        "solve_runs": "",
        "grad_mean_ms": "",
        "grad_best_ms": "",
        "grad_runs": "",
        "error_type": "",
        "error_detail": "",
        "stdout": "",
        "stderr": "",
    }

    install_result = install_version(version)
    if install_result.returncode != 0:
        row["status"] = "install_failed"
        row["error_type"] = "pip_install_failed"
        row["error_detail"] = compact_text(
            install_result.stderr or install_result.stdout or "pip install failed"
        )
        row["stdout"] = escape_multiline_text(install_result.stdout)
        row["stderr"] = escape_multiline_text(install_result.stderr)
        append_row(output_path, row)
        return

    benchmark_result = benchmark_script()
    if benchmark_result.returncode != 0:
        row["status"] = "benchmark_failed"
        row["error_type"] = "script_failed"
        row["error_detail"] = compact_text(
            benchmark_result.stderr or benchmark_result.stdout or "script.py failed"
        )
        row["stdout"] = escape_multiline_text(benchmark_result.stdout)
        row["stderr"] = escape_multiline_text(benchmark_result.stderr)
        row.update(parse_benchmark_output(benchmark_result.stdout))
        append_row(output_path, row)
        return

    row["status"] = "ok"
    row["stdout"] = escape_multiline_text(benchmark_result.stdout)
    row["stderr"] = escape_multiline_text(benchmark_result.stderr)
    row.update(parse_benchmark_output(benchmark_result.stdout))
    if not row["solve_mean_ms"] or not row["grad_mean_ms"]:
        row["status"] = "parse_failed"
        row["error_type"] = "benchmark_output_unrecognized"
        row["error_detail"] = "script.py output did not match expected benchmark format"
    append_row(output_path, row)


def main() -> int:
    args = parse_args()
    if args.min_version and args.max_version and args.min_version > args.max_version:
        print("--min-version must be <= --max-version", file=sys.stderr)
        return 2

    try:
        versions = discover_matching_versions(args.min_version, args.max_version)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if not versions:
        print("No matching versions found for the requested range.")
        return 0

    if args.dry_run:
        for version in versions:
            print(version)
        return 0

    write_header(args.output)
    for version in versions:
        print(f"Benchmarking jax=={version} jaxlib=={version}...")
        benchmark_version(version, args.output)

    print(f"Wrote results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
