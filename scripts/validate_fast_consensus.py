#!/usr/bin/env python3
"""
Validation script for FastConsensus vs bcftools consensus.

This script compares outputs from the Python-native FastConsensus implementation
against bcftools consensus to ensure correctness before switching to the faster
implementation.

Usage:
    python scripts/validate_fast_consensus.py \
        --vcf /path/to/sample.vcf.gz \
        --fasta /path/to/reference.fa.gz \
        --regions 100 \
        --output validation_report.json
"""

import argparse
import json
import subprocess
import random
import time
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.fast_consensus import FastConsensus
import pandas as pd


def run_bcftools_consensus(
    vcf_path: str,
    fasta_path: str,
    chrom: str,
    start: int,
    end: int,
    snp_only: bool = False,
) -> tuple[str, float]:
    """
    Run bcftools consensus and return sequence + timing.

    Returns:
        Tuple of (sequence, time_seconds)
    """
    region_str = f"{chrom}:{start}-{end}"

    if snp_only:
        filter_expr = 'ALT~"<.*>" || TYPE!="snp"'
    else:
        filter_expr = 'ALT~"<.*>"'

    cmd = (
        f'samtools faidx {fasta_path} "{region_str}" | '
        f'bcftools consensus -H I -e \'{filter_expr}\' '
        f'<(bcftools view -r "{region_str}" "{vcf_path}")'
    )

    start_time = time.perf_counter()
    result = subprocess.run(
        cmd, shell=True, executable='/bin/bash', capture_output=True, text=True
    )
    elapsed = time.perf_counter() - start_time

    if result.returncode != 0:
        # Fall back to reference
        cmd_ref = ["samtools", "faidx", fasta_path, region_str]
        result_ref = subprocess.run(cmd_ref, capture_output=True, text=True)
        if result_ref.returncode == 0:
            seq = "".join(result_ref.stdout.strip().split("\n")[1:])
            return seq, elapsed
        return "", elapsed

    seq = "".join(result.stdout.strip().split("\n")[1:])
    return seq, elapsed


def run_fast_consensus(
    consensus: FastConsensus,
    chrom: str,
    start: int,
    end: int,
    snp_only: bool = False,
) -> tuple[str, float]:
    """
    Run FastConsensus and return sequence + timing.

    Returns:
        Tuple of (sequence, time_seconds)
    """
    start_time = time.perf_counter()
    seq, _ = consensus.get_consensus(chrom, start, end, snp_only=snp_only)
    elapsed = time.perf_counter() - start_time
    return seq, elapsed


def compare_sequences(seq1: str, seq2: str) -> dict:
    """Compare two sequences and return detailed comparison."""
    if seq1 == seq2:
        return {"match": True, "differences": 0}

    # Find differences
    differences = []
    min_len = min(len(seq1), len(seq2))

    for i in range(min_len):
        if seq1[i] != seq2[i]:
            differences.append({
                "position": i,
                "bcftools": seq1[i],
                "pysam": seq2[i],
            })

    # Length difference
    len_diff = len(seq1) - len(seq2)

    return {
        "match": False,
        "differences": len(differences),
        "length_diff": len_diff,
        "bcftools_len": len(seq1),
        "pysam_len": len(seq2),
        "first_10_diffs": differences[:10],
    }


def load_gene_regions(gencode_path: str) -> pd.DataFrame:
    """Load gene regions from gencode file."""
    df = pd.read_csv(gencode_path)
    return df


def main():
    parser = argparse.ArgumentParser(description="Validate FastConsensus vs bcftools")
    parser.add_argument("--vcf", required=True, help="Path to VCF file")
    parser.add_argument("--fasta", required=True, help="Path to reference FASTA")
    parser.add_argument("--gencode", help="Path to gencode gene list CSV")
    parser.add_argument("--regions", type=int, default=100, help="Number of regions to test")
    parser.add_argument("--output", default="validation_report.json", help="Output report path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--snp-only", action="store_true", help="Test SNP-only mode")
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Loading FastConsensus from {args.vcf}...")
    load_start = time.perf_counter()
    consensus = FastConsensus(args.vcf, args.fasta)
    load_time = time.perf_counter() - load_start
    print(f"Loaded in {load_time:.2f}s")

    # Generate test regions
    if args.gencode:
        print(f"Loading gene regions from {args.gencode}...")
        genes = load_gene_regions(args.gencode)
        test_regions = []
        sample_genes = genes.sample(n=min(args.regions, len(genes)), random_state=args.seed)
        for _, gene in sample_genes.iterrows():
            test_regions.append({
                "chrom": gene["chromosome"],
                "start": int(gene["start"]),
                "end": int(gene["end"]),
                "gene_id": gene.get("gene_id", "unknown"),
            })
    else:
        # Default test regions across chromosomes
        print("Using default test regions...")
        chromosomes = [f"chr{i}" for i in range(1, 23)]
        test_regions = []
        for _ in range(args.regions):
            chrom = random.choice(chromosomes)
            start = random.randint(1000000, 100000000)
            end = start + random.randint(1000, 100000)
            test_regions.append({
                "chrom": chrom,
                "start": start,
                "end": end,
                "gene_id": f"{chrom}:{start}-{end}",
            })

    print(f"Testing {len(test_regions)} regions...")

    results = []
    matches = 0
    mismatches = 0
    bcftools_total_time = 0
    pysam_total_time = 0

    for i, region in enumerate(test_regions):
        chrom = region["chrom"]
        start = region["start"]
        end = region["end"]

        # Run both implementations
        bcftools_seq, bcftools_time = run_bcftools_consensus(
            args.vcf, args.fasta, chrom, start, end, snp_only=args.snp_only
        )
        pysam_seq, pysam_time = run_fast_consensus(
            consensus, chrom, start, end, snp_only=args.snp_only
        )

        bcftools_total_time += bcftools_time
        pysam_total_time += pysam_time

        # Compare
        comparison = compare_sequences(bcftools_seq, pysam_seq)

        result = {
            "region": f"{chrom}:{start}-{end}",
            "gene_id": region.get("gene_id", ""),
            "bcftools_time_ms": bcftools_time * 1000,
            "pysam_time_ms": pysam_time * 1000,
            "speedup": bcftools_time / pysam_time if pysam_time > 0 else float('inf'),
            **comparison,
        }
        results.append(result)

        if comparison["match"]:
            matches += 1
            status = "OK"
        else:
            mismatches += 1
            status = f"MISMATCH ({comparison['differences']} diffs)"

        # Progress
        if (i + 1) % 10 == 0 or i == 0:
            print(f"[{i+1}/{len(test_regions)}] {region['gene_id']}: {status}")

    # Summary
    summary = {
        "total_regions": len(test_regions),
        "matches": matches,
        "mismatches": mismatches,
        "match_rate": matches / len(test_regions) if test_regions else 0,
        "vcf_load_time_s": load_time,
        "bcftools_total_time_s": bcftools_total_time,
        "pysam_total_time_s": pysam_total_time,
        "avg_bcftools_time_ms": (bcftools_total_time / len(test_regions) * 1000) if test_regions else 0,
        "avg_pysam_time_ms": (pysam_total_time / len(test_regions) * 1000) if test_regions else 0,
        "avg_speedup": bcftools_total_time / pysam_total_time if pysam_total_time > 0 else float('inf'),
        "snp_only_mode": args.snp_only,
    }

    report = {
        "summary": summary,
        "results": results,
        "mismatched_regions": [r for r in results if not r["match"]],
    }

    # Save report
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total regions tested: {summary['total_regions']}")
    print(f"Matches: {summary['matches']}")
    print(f"Mismatches: {summary['mismatches']}")
    print(f"Match rate: {summary['match_rate']:.1%}")
    print()
    print(f"VCF load time: {summary['vcf_load_time_s']:.2f}s")
    print(f"Avg bcftools time: {summary['avg_bcftools_time_ms']:.1f}ms")
    print(f"Avg pysam time: {summary['avg_pysam_time_ms']:.1f}ms")
    print(f"Average speedup: {summary['avg_speedup']:.1f}x")
    print()
    print(f"Report saved to: {args.output}")

    if mismatches > 0:
        print("\nWARNING: Some regions did not match!")
        print("Check the report for details on mismatched regions.")
        return 1

    print("\nSUCCESS: All regions match!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
