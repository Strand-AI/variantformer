# Main training script for the Seq2GenePredictor model.
from pathlib import Path
import torch  # type: ignore
from torch.utils.data import Dataset
from utils.data_process import ExtractSeqFromBed
import pandas as pd
from utils.seq import BPEEncoder
from utils.constants import (
    MAP_REF_CRE_TO_IDX,
    SPECIAL_TOKENS,
)
import yaml
from utils import constants
from utils.functions import reverse_complement, multi_try_load_csv
from utils.assets import GeneManifestLookup


def collate_fn_batching(batch):
    """Collate function for batching

    Args:
        batch: Batch of data

    Returns:
        tuple: A tuple containing:
            - X: List of CRE token IDs
            - attention_mask: List of CRE attention masks
            - tissue_context: List of tissue context
            - labels: List of labels
            - ref_labels: List of reference labels
            - strand_tensor: List of strands
            - gene_embeddings: List of gene embeddings
            - gene_attention_masks: List of gene attention masks
    """
    data = []
    attention_mask = []
    context = []
    labels = []
    ref_labels = []
    strands = []
    gene_embeddings = []
    gene_attention_masks = []
    for X, mask, ctx, label, ref_label, strand, emb, amb_att in batch:
        data.append(X)  # cre token ids
        attention_mask.append(mask)  # cre attention mask
        context.append(ctx)  # tissue context
        labels.append(label)  # label
        ref_labels.append(ref_label)  # ref cre label
        gene_embeddings.append(emb)  # gene embeddings
        gene_attention_masks.append(amb_att)  # gene attention masks
        strands.append(strand.unsqueeze(0))  # strand
    strand_val = torch.cat(strands, dim=0)
    D = {
        "cre_sequences": data,
        "cre_attention_masks": attention_mask,
        "tissue_context": context,
        "cre_labels": labels,
        "ref_cre_labels": ref_labels,
        "strand_val": strand_val,
        "gene_embeddings": gene_embeddings,
        "gene_attention_masks": gene_attention_masks,
    }
    return D


class VCFDataset(Dataset):
    """Dataset class for gene expression prediction with CRE sequences and gene embeddings.

    This dataset handles loading and preprocessing of gene data, CRE sequences,
    and gene embeddings for model training and evaluation.
    """

    def __init__(
        self,
        max_length: int,
        max_chunks: int,
        cre_neighbour_hood: int,
        gencode_v24: str,
        gene_cre_manifest: GeneManifestLookup,
        gene_upstream_neighbour_hood: int,
        gene_downstream_neighbour_hood: int,
        query_df: pd.DataFrame,
        fasta_path: str,
        vcf_path: str = None,
    ):
        # Tokenization setup
        base_dir = Path(__file__).resolve().parent.parent
        bpe_vocab_path = base_dir / "vocabs" / "bpe_vocabulary_500.json"
        self.bpe = BPEEncoder()
        self.bpe.load_vocabulary(str(bpe_vocab_path))
        self.vocab = self.bpe.tokenizer.get_vocab()
        self.pad_token_id = self.vocab.get(SPECIAL_TOKENS["pad_token"])

        # Sequence processing
        self.max_chunks = max_chunks
        self.max_length = max_length
        self.cre_neighbour_hood = cre_neighbour_hood
        self.gene_upstream_neighbour_hood = gene_upstream_neighbour_hood
        self.gene_downstream_neighbour_hood = gene_downstream_neighbour_hood
        self.query_df = query_df

        # Ref data
        self.gene_cre_manifest = gene_cre_manifest
        self.gencode_v24 = gencode_v24
        self.fasta_path = fasta_path

        # Load file list
        self.ref_cre_to_idx = MAP_REF_CRE_TO_IDX
        self.cre_to_idx = constants.MAP_CRE_TO_IDX  # Initialize cre_to_idx
        self.gencode_v24 = pd.read_csv(gencode_v24)

        # Tissue vocab
        tissue_vocab_path = base_dir / "vocabs" / "tissue_vocab.yaml"
        with open(tissue_vocab_path, "r") as f:
            tissue_vocab = yaml.safe_load(f)
        self.tissue_vocab = tissue_vocab

        # VCF data
        self.vcf_path = vcf_path

        self._check_filter_query_df()

    def _check_filter_query_df(self):
        """Check and filter the query df

        Args:
            query_df: Query dataframe
        """
        # Check if the query df is valid and filter it based on the gencode v24 and tissue vocab
        assert self.query_df is not None, "Query dataframe is not provided"
        assert (
            "gene_id" in self.query_df.columns
        ), "Query dataframe must contain gene_id column"
        assert (
            "tissues" in self.query_df.columns
        ), "Query dataframe must contain tissues column"
        len_query_df = len(self.query_df)
        D = []
        for it, row in self.query_df.iterrows():
            tissues = row["tissues"]
            gene_id = row["gene_id"]
            tissues = tissues.split(",")
            if gene_id not in self.gencode_v24["gene_id"].values:
                print(f"Gene {gene_id} not found in the training set so skipping it")
                continue
            T = []
            T_names = []
            for tissue in tissues:
                if tissue in self.tissue_vocab:
                    T.append(self.tissue_vocab[tissue])
                    T_names.append(tissue)
                else:
                    print(
                        f"Tissue {tissue} not found in the tissue vocab so skipping it"
                    )
                    continue
            if len(T) == 0:
                print(f"No tissues found for gene {gene_id}")
                continue
            D.append({"gene_id": gene_id, "tissues": T, "tissue_names": T_names})
        if len(D) == 0:
            raise ValueError(
                "No genes found in the query df that are present in the gencode v24 and have at least one tissue in the training set of VariantFormer"
            )
        self.query_df = pd.DataFrame(D)
        print(
            f"Filtered query df to {len(self.query_df)} genes reducing from {len_query_df}"
        )
        return True

    def __len__(self):
        return len(self.query_df)

    def __getitem__(self, idx):
        x = self._load_file(idx)
        return x

    def _get_gene_info(self, gene_id: str) -> dict:
        """Get the gene info for a given gene id

        Args:
            gene_id: Gene ID

        Returns:
            dict: A dictionary containing the gene info for the given gene id with keys:
                - 'gene_id'
                - 'gene_name'
                - 'chromosome'
                - 'start'
                - 'end'
                - 'strand'
        """
        gene_info = (
            self.gencode_v24[self.gencode_v24["gene_id"] == gene_id].iloc[0].to_dict()
        )
        return gene_info

    def __adjust_length(self, token_ids):
        """Adjust sequence length to max length and create attention mask.

        Args:
            token_ids: List of token IDs to adjust

        Returns:
            tuple: A tuple containing:
                - token_ids: Adjusted token IDs
                - attention_mask: Adjusted attention mask
        """
        attention_mask = [0] * len(token_ids)
        if len(token_ids) < self.max_length:
            padding_length = self.max_length - len(token_ids)
            token_ids += [self.pad_token_id] * padding_length
            attention_mask += [1] * padding_length
        else:
            token_ids = token_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]
        return token_ids, attention_mask

    def _get_cres(self, gene_id: str, gene_info: dict, vcf_path: str) -> pd.DataFrame:
        """Get the cres for a given gene id

        Args:
            gene_id: Gene ID

        Returns:
            pd.DataFrame: A dataframe containing the cres for the given gene id
        """
        gene_cre_map_path = self.gene_cre_manifest.get_file_path(gene_id)
        genes_cre_map = multi_try_load_csv(gene_cre_map_path)
        bed_regions = genes_cre_map[["chromosome", "start_cre", "end_cre", "cre_name"]]
        bed_regions = bed_regions.rename(
            columns={
                "chromosome": "chrom",
                "start_cre": "start",
                "end_cre": "end",
                "cre_name": "cCRE",
            }
        )
        cres = ExtractSeqFromBed(
            neighbour_hood=self.cre_neighbour_hood, ref_fasta=self.fasta_path
        ).process_subject(vcf_file=vcf_path, bed_regions=bed_regions)

        if gene_info["strand"] == "-":
            cres = cres.iloc[
                ::-1
            ]  # reverse the cres if the gene is on the minus strand

        # Extract sequences and labels in one pass using itertuples (10x faster than iterrows)
        is_minus_strand = gene_info["strand"] == "-"
        sequences = []
        ref_labels = []
        default_label_idx = self.cre_to_idx["Low-DNase"]
        for row in cres.itertuples():
            seq = row.sequence
            sequences.append(reverse_complement(seq) if is_minus_strand else seq)
            ref_labels.append(self.ref_cre_to_idx[row.cCRE])

        # Batch tokenize all CRE sequences at once (much faster than per-CRE encoding)
        all_token_ids = self.bpe.encode_batch_forward(sequences)

        # Vectorized padding and tensor creation
        n_cres = len(all_token_ids)
        X_tensor = torch.full((n_cres, 1, self.max_length), self.pad_token_id, dtype=torch.long)
        attention_mask_tensor = torch.ones((n_cres, 1, self.max_length), dtype=torch.bool)

        for i, token_ids in enumerate(all_token_ids):
            seq_len = min(len(token_ids), self.max_length)
            X_tensor[i, 0, :seq_len] = torch.tensor(token_ids[:seq_len], dtype=torch.long)
            attention_mask_tensor[i, 0, :seq_len] = False  # 0 = not padded

        ref_labels_tensor = torch.tensor(ref_labels, dtype=torch.long)
        labels_tensor = torch.full((n_cres,), default_label_idx, dtype=torch.long)
        return X_tensor, attention_mask_tensor, ref_labels_tensor, labels_tensor

    def _get_gene(self, gene_id: str, gene_info: dict, vcf_path: str) -> pd.DataFrame:
        mutated_seq = ExtractSeqFromBed(
            neighbour_hood=self.gene_downstream_neighbour_hood,
            ref_fasta=self.fasta_path,
            upstream_neighbour_hood=self.gene_upstream_neighbour_hood,
        ).process_gene(gene_info, vcf_path)
        assert (
            len(mutated_seq) > 1000
        ), f"Mutated sequence is less than 1000bp for gene {gene_id}"

        token_id, _, _, _ = (
            self.bpe.encode([str(mutated_seq), "A"])
            if gene_info["strand"] == "+"
            else self.bpe.encode([str(reverse_complement(mutated_seq)), "A"])
        )
        tokens_ids = torch.tensor(token_id, dtype=torch.long)
        X = tokens_ids.unsqueeze(0)
        X, attention_mask = self.chunkify_data(X)
        return X, attention_mask

    def _load_file(self, idx: int) -> tuple:
        # Load metadata and file paths based on data source
        tissues = self.query_df.iloc[idx]["tissues"]
        gene_id = self.query_df.iloc[idx]["gene_id"]
        if "vcf_path" not in self.query_df.columns:
            vcf_path = self.vcf_path
        else:
            vcf_path = self.query_df.iloc[idx]["vcf_path"]
        gene_info = self._get_gene_info(gene_id)
        strand = gene_info["strand"]
        strand = [0] if strand == "+" else [1]
        assert (
            gene_info["chromosome"] in ["chr" + str(i) for i in range(1, 23)]
        ), f"Chromosome {gene_info['chromosome']} is not a valid chromosome. Sex chromosomes are not supported"
        X, attention_mask, ref_labels, labels = self._get_cres(
            gene_id, gene_info, vcf_path
        )
        gene_embeddings, gene_attention_masks = self._get_gene(
            gene_id, gene_info, vcf_path
        )
        strand_tensor = torch.tensor(strand, dtype=torch.long)
        tissue_context = torch.tensor(tissues, dtype=torch.long)
        return (
            X,
            attention_mask,
            tissue_context,
            labels,
            ref_labels,
            strand_tensor,
            gene_embeddings,
            gene_attention_masks,
        )

    def chunkify_data(self, x):
        """Transform the input

        Args:
            x: Input to transform

        Returns:
            tuple: A tuple containing:
                - chunk_list: List of chunks
                - chunk_attention_mask_list: List of attention masks for the chunks
        """
        attention_mask = torch.zeros_like(x)
        max_token_allowed = self.max_length
        chunk_list = []
        chunk_attention_mask_list = []
        fixed_chunking = (
            True  # If extended training is not used, then fixed chunking is used
        )
        max_chunks_allowed = self.max_chunks  # default 300. Beyond 300 facing OOM error
        for start in range(0, x.size(1), max_token_allowed):
            end = start + max_token_allowed
            chunk = x[:, start:end]
            chunk_attention_mask = attention_mask[:, start:end]
            if chunk.size(1) < max_token_allowed:
                padding = self.pad_token_id * torch.ones(
                    (chunk.size(0), max_token_allowed - chunk.size(1)),
                    dtype=chunk.dtype,
                ).to(chunk.device)  # pad token is 0
                chunk = torch.cat([chunk, padding], dim=1)
                chunk_attention_mask = torch.cat(
                    [
                        chunk_attention_mask,
                        torch.ones(
                            (
                                chunk_attention_mask.size(0),
                                max_token_allowed - chunk_attention_mask.size(1),
                            ),
                            dtype=chunk_attention_mask.dtype,
                        ).to(chunk_attention_mask.device),
                    ],
                    dim=1,
                )
            if fixed_chunking:
                max_chunks_allowed -= 1
                if max_chunks_allowed < 0:
                    break

            chunk_list.append(chunk)
            chunk_attention_mask_list.append(chunk_attention_mask)

        chunk_list = torch.vstack(chunk_list).unsqueeze(1).to(dtype=torch.long)
        # chunk_list = chunk_list.repeat(1, 2, 1)

        chunk_attention_mask_list = torch.vstack(chunk_attention_mask_list).unsqueeze(1)
        chunk_attention_mask_list = chunk_attention_mask_list.to(dtype=torch.bool)
        # chunk_attention_mask_list = chunk_attention_mask_list.repeat(1, 2, 1)
        return chunk_list, chunk_attention_mask_list
