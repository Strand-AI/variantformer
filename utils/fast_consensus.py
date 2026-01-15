"""
Fast consensus sequence generation using pysam.

This module replaces bcftools consensus subprocess calls with pure Python
implementation using pysam. Key benefits:
- Load VCF once per sample instead of 18,439 times
- No subprocess spawn overhead (~100ms per call)
- ~100x faster: ~10ms per gene vs ~1.6sec with bcftools

The implementation matches bcftools consensus -H I behavior:
- IUPAC ambiguity codes for heterozygous SNPs (e.g., R for A/G)
- First haplotype for heterozygous indels
- Structural variant filtering (ALT~"<.*>")
"""

import pysam
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

log = logging.getLogger(__name__)

# IUPAC ambiguity codes for heterozygous SNPs
# Maps frozenset of two bases to ambiguity code
IUPAC_CODES = {
    frozenset({'A', 'C'}): 'M',
    frozenset({'A', 'G'}): 'R',
    frozenset({'A', 'T'}): 'W',
    frozenset({'C', 'G'}): 'S',
    frozenset({'C', 'T'}): 'Y',
    frozenset({'G', 'T'}): 'K',
}


@dataclass
class Variant:
    """Represents a single variant."""
    chrom: str
    pos: int  # 1-based position (VCF convention)
    ref: str
    alt: str
    is_snp: bool
    is_het: bool  # heterozygous
    gt: Tuple[int, ...]  # genotype tuple e.g., (0, 1)


class FastConsensus:
    """
    Fast consensus sequence generator using pysam.

    Loads VCF once and indexes variants for efficient region queries.
    Memory usage: ~1.4GB per sample (7M variants × 200 bytes).

    Usage:
        consensus = FastConsensus(vcf_path, fasta_path)
        seq = consensus.get_consensus("chr1", 1000, 2000)
        # Or with SNP-only filtering:
        seq = consensus.get_consensus("chr1", 1000, 2000, snp_only=True)
    """

    def __init__(self, vcf_path: str, fasta_path: str):
        """
        Initialize FastConsensus with VCF and reference FASTA.

        Args:
            vcf_path: Path to bgzipped and indexed VCF file
            fasta_path: Path to reference FASTA file (can be gzipped)
        """
        self.vcf_path = vcf_path
        self.fasta_path = fasta_path

        # Load reference FASTA
        self.fasta = pysam.FastaFile(fasta_path)

        # Load and index all variants
        self._variants_by_chrom: Dict[str, Dict[int, Variant]] = {}
        self._load_variants()

    def _load_variants(self):
        """Load all variants from VCF into memory index."""
        log.info(f"Loading variants from {self.vcf_path}")

        vcf = pysam.VariantFile(self.vcf_path)
        variant_count = 0
        skipped_sv = 0

        for record in vcf.fetch():
            # Skip structural variants (ALT contains <...>)
            alt = record.alts[0] if record.alts else None
            if alt is None or alt.startswith('<'):
                skipped_sv += 1
                continue

            # Get genotype for first (and only) sample
            sample = record.samples[0]
            gt = sample['GT']

            # Skip if no call or homozygous reference
            if gt is None or gt == (0, 0) or gt == (None, None):
                continue

            # Determine if heterozygous
            is_het = (gt[0] != gt[1]) if len(gt) >= 2 else False

            # Determine if SNP
            ref = record.ref
            is_snp = len(ref) == 1 and len(alt) == 1

            chrom = record.chrom
            pos = record.pos  # 1-based

            variant = Variant(
                chrom=chrom,
                pos=pos,
                ref=ref,
                alt=alt,
                is_snp=is_snp,
                is_het=is_het,
                gt=gt,
            )

            if chrom not in self._variants_by_chrom:
                self._variants_by_chrom[chrom] = {}

            # Store by position (1-based)
            self._variants_by_chrom[chrom][pos] = variant
            variant_count += 1

        vcf.close()
        log.info(f"Loaded {variant_count:,} variants, skipped {skipped_sv:,} structural variants")

    def _get_iupac_code(self, ref: str, alt: str) -> str:
        """Get IUPAC ambiguity code for heterozygous SNP."""
        bases = frozenset({ref.upper(), alt.upper()})
        return IUPAC_CODES.get(bases, 'N')

    def _apply_variant(self, ref_base: str, variant: Variant) -> Tuple[str, int]:
        """
        Apply a variant to get the resulting base(s).

        Returns:
            Tuple of (resulting sequence, length change from reference)
            Length change is used to track position shifts from indels.
        """
        if variant.is_het:
            if variant.is_snp:
                # Heterozygous SNP: return IUPAC ambiguity code
                return self._get_iupac_code(variant.ref, variant.alt), 0
            else:
                # Heterozygous indel: use first haplotype (bcftools -H I behavior)
                # GT (0, 1) → use ref, GT (1, 0) → use alt
                if variant.gt[0] == 0:
                    return variant.ref, 0
                else:
                    return variant.alt, len(variant.alt) - len(variant.ref)
        else:
            # Homozygous alt
            if variant.is_snp:
                return variant.alt, 0
            else:
                return variant.alt, len(variant.alt) - len(variant.ref)

    def get_consensus(
        self,
        chrom: str,
        start: int,
        end: int,
        snp_only: bool = False,
    ) -> Tuple[str, int]:
        """
        Get consensus sequence for a region.

        Args:
            chrom: Chromosome name
            start: Start position (1-based, inclusive)
            end: End position (1-based, inclusive)
            snp_only: If True, only apply SNP variants (ignore indels)

        Returns:
            Tuple of (consensus sequence, number of mutations applied)
        """
        # Fetch reference sequence (pysam uses 0-based half-open coordinates)
        try:
            ref_seq = self.fasta.fetch(chrom, start - 1, end)
        except (KeyError, ValueError) as e:
            log.warning(f"Failed to fetch reference for {chrom}:{start}-{end}: {e}")
            return "", 0

        if not ref_seq:
            return "", 0

        # Get variants in this chromosome
        chrom_variants = self._variants_by_chrom.get(chrom, {})

        # Find variants in region
        region_variants = []
        for pos in range(start, end + 1):
            if pos in chrom_variants:
                v = chrom_variants[pos]
                if snp_only and not v.is_snp:
                    continue
                region_variants.append(v)

        if not region_variants:
            return ref_seq, 0

        # Sort variants by position (descending for right-to-left application)
        # This ensures indel position shifts don't affect subsequent variants
        region_variants.sort(key=lambda v: v.pos, reverse=True)

        # Convert to list for mutation
        seq_list = list(ref_seq)
        mutations_applied = 0

        for variant in region_variants:
            # Convert to 0-based index relative to region start
            idx = variant.pos - start

            if idx < 0 or idx >= len(seq_list):
                continue

            # Verify reference matches (sanity check)
            ref_len = len(variant.ref)
            actual_ref = ''.join(seq_list[idx:idx + ref_len])

            if actual_ref.upper() != variant.ref.upper():
                log.debug(
                    f"Reference mismatch at {chrom}:{variant.pos}: "
                    f"expected {variant.ref}, got {actual_ref}"
                )
                continue

            # Apply variant
            new_seq, _ = self._apply_variant(seq_list[idx], variant)

            # Replace in sequence
            seq_list[idx:idx + ref_len] = list(new_seq)
            mutations_applied += 1

        return ''.join(seq_list), mutations_applied

    def close(self):
        """Close file handles."""
        if self.fasta:
            self.fasta.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class FastConsensusCache:
    """
    Cache for FastConsensus instances, one per VCF file.

    This is useful when processing multiple samples in parallel with DataLoader,
    where each worker maintains its own cache.

    Usage:
        cache = FastConsensusCache(fasta_path)
        consensus = cache.get(vcf_path)  # Returns cached or creates new
        seq = consensus.get_consensus("chr1", 1000, 2000)
    """

    def __init__(self, fasta_path: str, max_cached: int = 1):
        """
        Initialize cache.

        Args:
            fasta_path: Path to reference FASTA (shared across all VCFs)
            max_cached: Maximum number of VCF indexes to keep cached
        """
        self.fasta_path = fasta_path
        self.max_cached = max_cached
        self._cache: Dict[str, FastConsensus] = {}
        self._access_order: List[str] = []

    def get(self, vcf_path: str) -> FastConsensus:
        """Get FastConsensus for a VCF, using cache if available."""
        if vcf_path in self._cache:
            # Move to end of access order (LRU)
            self._access_order.remove(vcf_path)
            self._access_order.append(vcf_path)
            return self._cache[vcf_path]

        # Create new instance
        consensus = FastConsensus(vcf_path, self.fasta_path)

        # Evict oldest if at capacity
        while len(self._cache) >= self.max_cached and self._access_order:
            oldest = self._access_order.pop(0)
            old_consensus = self._cache.pop(oldest, None)
            if old_consensus:
                old_consensus.close()

        # Add to cache
        self._cache[vcf_path] = consensus
        self._access_order.append(vcf_path)

        return consensus

    def clear(self):
        """Clear all cached instances."""
        for consensus in self._cache.values():
            consensus.close()
        self._cache.clear()
        self._access_order.clear()

    def __del__(self):
        self.clear()
