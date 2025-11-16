"""Legacy entry-point for running the ARGprism pipeline locally."""

from __future__ import annotations

from pathlib import Path

from argprism.pipeline import run_pipeline


def main() -> None:
    input_fasta = Path("Input_proteins/ERR589503_PROT.faa")
    output_dir = Path("./timestamp_run")

    result = run_pipeline(
        input_fasta=input_fasta,
        output_dir=output_dir,
    )

    print(f"Predicted sequences written to {result.predicted_fasta}")
    if result.diamond_output:
        print(f"DIAMOND results written to {result.diamond_output}")
    if result.report_csv:
        print(f"Final report written to {result.report_csv}")
    print(f"Elapsed time: {result.elapsed_seconds:.2f} seconds")


if __name__ == "__main__":
    main()

