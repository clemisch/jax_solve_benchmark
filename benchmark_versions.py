#!/usr/bin/env python3
import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


VERSION_RE = re.compile(r"\b\d+\.\d+\.\d+\b")


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
        default=Path("benchmark_results.txt"),
        help="output file for benchmark logs (default: benchmark_results.txt)",
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


def append_section(output_path: Path, lines: list[str]) -> None:
    with output_path.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
        fh.write("\n\n")


def format_command_output(label: str, content: str) -> list[str]:
    if not content.strip():
        return [f"{label}: <empty>"]
    lines = [f"{label}:"]
    lines.extend(content.rstrip().splitlines())
    return lines


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
    header = [
        f"version: {version}",
        f"timestamp: {timestamp()}",
    ]

    install_result = install_version(version)
    if install_result.returncode != 0:
        lines = header + [
            "status: install_failed",
            *format_command_output("stdout", install_result.stdout),
            *format_command_output("stderr", install_result.stderr),
        ]
        append_section(output_path, lines)
        return

    benchmark_result = benchmark_script()
    if benchmark_result.returncode != 0:
        lines = header + [
            "status: benchmark_failed",
            *format_command_output("stdout", benchmark_result.stdout),
            *format_command_output("stderr", benchmark_result.stderr),
        ]
        append_section(output_path, lines)
        return

    lines = header + [
        "status: ok",
        *format_command_output("stdout", benchmark_result.stdout),
        *format_command_output("stderr", benchmark_result.stderr),
    ]
    append_section(output_path, lines)


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

    args.output.write_text("", encoding="utf-8")
    for version in versions:
        print(f"Benchmarking jax=={version} jaxlib=={version}...")
        benchmark_version(version, args.output)

    print(f"Wrote results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
