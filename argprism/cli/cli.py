# Copyright 2025 ARGPrism Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Command-line interface for ARGPrism.

This module provides the command-line interface for the ARGPrism package,
allowing users to interact with the antibiotic resistance gene prediction
functionality through command-line arguments and options.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

# Set threading environment variables BEFORE any PyTorch/ML imports
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"  
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from .. import __version__
from ..io.file_paths import (
    DEFAULT_DIAMOND_BLAST_OUTPUT_FILENAME,
    DEFAULT_PREDICTED_ARGS_FASTA_FILENAME,
    DEFAULT_FINAL_REPORT_CSV_FILENAME,
)
from ..core.pipeline import run_pipeline


class ARGPrismCLI:
    """Command-line interface for ARGPrism."""
    
    def __init__(self):
        self.logger = None
        self.console = Console()
    
    def setup_logging(self, log_file=None, verbose=False, quiet=False):
        """Configure logging based on command line arguments."""
        log_level = logging.INFO
        if verbose:
            log_level = logging.DEBUG
        elif quiet:
            log_level = logging.ERROR
        
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ))
            logging.getLogger().addHandler(file_handler)
        
        self.logger = logging.getLogger("argprism")
    
    def print_banner(self):
        """Print the ARGPrism banner with Rich formatting."""
        banner = r"""
        
 █████╗ ██████╗  ██████╗ ██████╗ ██████╗ ██╗███████╗███╗   ███╗
██╔══██╗██╔══██╗██╔════╝ ██╔══██╗██╔══██╗██║██╔════╝████╗ ████║
███████║██████╔╝██║  ███╗██████╔╝██████╔╝██║███████╗██╔████╔██║
██╔══██║██╔══██╗██║   ██║██╔═══╝ ██╔══██╗██║╚════██║██║╚██╔╝██║
██║  ██║██║  ██║╚██████╔╝██║     ██║  ██║██║███████║██║ ╚═╝ ██║
╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝  ╚═╝╚═╝╚══════╝╚═╝     ╚═╝
        """
        self.console.print(Text(banner, style="magenta"))
        self.console.print(Panel(Text("ARGPrism CLI Agent", justify="right"), style="bold blue", expand=False))
    
    def print_status(self, message, style="green"):
        """Print status messages with Rich formatting."""
        self.console.print(f"[+] {message}", style=f"bold {style}")
    
    def print_help_table(self):
        """Print help information in a tabular format using Rich."""
        table = Table(show_header=True, header_style="bold blue", box=box.ROUNDED, highlight=True)
        table.add_column("Options", style="bold cyan", width=30, min_width=25)
        table.add_column("Description", style="white", min_width=40)
        table.add_row("-h, --help", "Show this help message and exit")
        table.add_row("-i, --input", "Input protein FASTA file containing sequences to analyse")
        table.add_row("-o, --output-dir", "Directory where pipeline outputs will be written (default: 'argprism_output')")
        table.add_row("-v, --verbose", "Increase output verbosity")
        table.add_row("-q, --quiet", "Suppress all non-error output")
        table.add_row("-l, --log-file", "Path to save log file")
        table.add_row("--device", "Force execution on CPU or CUDA ('cpu, cuda')")
        table.add_row("--output-fasta", "Filename for predicted ARG sequences (default: 'predicted_ARGs.fasta')")
        table.add_row("--diamond-output", "Filename for DIAMOND BLAST results (default: 'predicted_ARGs_vs_ref.tsv')")
        table.add_row("--report", "Filename for final CSV report (default: 'final_ARG_prediction_report.csv')")
        table.add_row("--reuse-diamond-db", "Skip rebuilding the DIAMOND database if it already exists")
        table.add_row("--version", "Show package version")
        self.console.print(table)
    
    def parse_args(self):
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(
            description="ARGPrism: Deep learning-based antibiotic resistance gene prediction pipeline",
            epilog="For more information, visit: https://github.com/haseebmanzur/ARGPrism",
            add_help=False
        )
        
        parser.add_argument("-h", "--help", action="store_true", help="Show this help message and exit")
        parser.add_argument("--input", "-i", required=False,
                            help="Input protein FASTA file containing sequences to analyse")
        parser.add_argument("--output-dir", "-o", default="argprism_output",
                            help="Directory where pipeline outputs will be written (default: argprism_output)")
        parser.add_argument("--output-fasta", default=DEFAULT_PREDICTED_ARGS_FASTA_FILENAME,
                            help=f"Filename for predicted ARG sequences (default: {DEFAULT_PREDICTED_ARGS_FASTA_FILENAME})")
        parser.add_argument("--diamond-output", default=DEFAULT_DIAMOND_BLAST_OUTPUT_FILENAME,
                            help=f"Filename for DIAMOND BLAST results (default: {DEFAULT_DIAMOND_BLAST_OUTPUT_FILENAME})")
        parser.add_argument("--report", default=DEFAULT_FINAL_REPORT_CSV_FILENAME,
                            help=f"Filename for final CSV report (default: {DEFAULT_FINAL_REPORT_CSV_FILENAME})")
        parser.add_argument("--device", choices=["cpu", "cuda"],
                            help="Force execution on CPU or CUDA. Defaults to auto-detect.")
        parser.add_argument("--reuse-diamond-db", action="store_true",
                            help="Skip rebuilding the DIAMOND database if it already exists")
        parser.add_argument("--verbose", "-v", action="store_true",
                            help="Increase output verbosity")
        parser.add_argument("--quiet", "-q", action="store_true",
                            help="Suppress all non-error output")
        parser.add_argument("--log-file", "-l",
                            help="Path to save log file")
        parser.add_argument("--version", action="store_true",
                            help="Show package version")
        
        args = parser.parse_args()
        
        if args.verbose and args.quiet:
            parser.error("--verbose and --quiet cannot be used together")
        
        return args
    
    def run(self):
        """Run the ARGPrism CLI interface."""
        # Always show banner
        self.print_banner()
        
        args = self.parse_args()
        
        # Handle help or no arguments
        if args.help or len(sys.argv) == 1:
            self.print_help_table()
            self.console.print(Panel(Text("For more information, visit: https://github.com/haseebmanzur/ARGPrism", 
                                        justify="center"), style="bold green", expand=False))
            return 0
        
        # Handle version
        if args.version:
            self.console.print(f"ARGPrism version {__version__}", style="bold green")
            return 0
        
        # Validate required arguments
        if not args.input:
            self.console.print("Error: --input is required for analysis.", style="bold red")
            self.print_help_table()
            return 1
        
        # Validate input file exists
        input_path = Path(args.input)
        if not input_path.exists():
            self.console.print(f"Error: Input file '{args.input}' does not exist.", style="bold red")
            return 1
        
        self.setup_logging(log_file=args.log_file, verbose=args.verbose, quiet=args.quiet)
        
        try:
            self.print_status(f"Starting ARGPrism 'v{__version__}'", "green")
            self.print_status(f"Processing input file: {args.input}", "cyan")
            self.print_status(f"Output directory: '{args.output_dir}'", "cyan")
            
            # Run the pipeline
            result = run_pipeline(
                input_fasta=args.input,
                output_dir=args.output_dir,
                output_fasta=args.output_fasta,
                diamond_output=args.diamond_output,
                final_report=args.report,
                preferred_device=args.device,
                build_diamond_db=not args.reuse_diamond_db,
                verbose=not args.quiet,
            )
            
            # Print results summary
            if not args.quiet:
                self._print_results_summary(result)
            
            self.print_status(f"Results saved to: {args.output_dir}")
            self.print_status("ARGPrism completed successfully!", "green")
            
        except Exception as e:
            self.console.print(f"Error: {str(e)}", style="bold red")
            if args.verbose:
                import traceback
                self.console.print(traceback.format_exc(), style="bold red")
            return 1
        
        return 0
    
    def _print_results_summary(self, result):
        """Print results summary in Rich format."""
        total_predictions = len(result.predictions)
        arg_predictions = sum(1 for pred in result.predictions.values() if pred == "ARG")
        non_arg_predictions = total_predictions - arg_predictions
        
        summary_text = f"""
        [bright_green]Analysis completed successfully![/bright_green]
        
        [bright_cyan]Total sequences processed:[/bright_cyan] [yellow]{total_predictions}[/yellow]
        [bright_cyan]ARGs predicted:[/bright_cyan] [yellow]{arg_predictions}[/yellow]
        [bright_cyan]Non-ARGs predicted:[/bright_cyan] [yellow]{non_arg_predictions}[/yellow]
        [bright_cyan]Processing time:[/bright_cyan] [yellow]{result.elapsed_seconds:.2f}s[/yellow]

        [bright_cyan]Output files saved to:[/bright_cyan] [dim]{Path(result.predicted_fasta).parent}[/dim]
        """
        if result.report_csv:
            summary_text += f"\n[bright_cyan]Final report:[/bright_cyan] [dim]{result.report_csv}[/dim]"
        if result.diamond_output:
            summary_text += f"\n[bright_cyan]DIAMOND results:[/bright_cyan] [dim]{result.diamond_output}[/dim]"
        
        self.console.print(Panel(
            summary_text,
            title="[bright_cyan]Analysis Results[/bright_cyan]",
            style="bright_blue",
            padding=(1, 2),
            expand=False
        ))


def main():
    """Main entry point for the CLI."""
    cli = ARGPrismCLI()
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())
