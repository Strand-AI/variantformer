"""
Combined Modulator Model for VariantFormer: Memory-Optimized Version
This model combines the epigenetics_modulator and gene_modulator into a single
processing unit that directly feeds CRE outputs to gene layers without storing
intermediate tensors, significantly reducing memory usage.
"""

import torch
import torch.nn as nn
import lightning.pytorch as pl
import logging
import numpy as np

# Import base model components
from .modules.layers import (
    ContextFlashAttentionEncoderLayer,
    ContextFlashCrossAttentionEncoderLayer,
    FlashAttentionEncoderLayer,
    MultiRegistry,
    StartToken,
    TissueExpressionHeads,
    AddContext,
)
from utils import constants
from flash_attn.bert_padding import (
    pad_input,
    unpad_input,
)
from utils.functions import precision2dtype

logger = logging.getLogger(__name__)
MAX_WINDOW_SIZE = 30000000
MAX_CHUNK_SIZE = 1024


class CombinedModulator(nn.Module):
    """
    Memory-optimized combined modulator that processes CRE and gene sequences
    in a coupled fashion without storing large intermediate tensors.

    This modulator processes both sequences layer by layer, directly feeding
    CRE outputs to gene layers without materializing intermediate results.
    """

    def __init__(
        self,
        emb_dim,
        num_heads,
        num_layers,
        use_alibi,
        mlp_dout,
        use_context,
        num_ref_cres=None,
        only_cross_attention=True,
        use_res=False,
        cross_alibi=False,
        flash_attn_3=False,
    ):
        super().__init__()

        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_context = use_context
        self.only_cross_attention = only_cross_attention
        self.use_res = use_res
        self.cross_alibi = cross_alibi

        # Context embedding for CRE sequences (if using context)
        if use_context:
            assert (
                num_ref_cres is not None
            ), "num_ref_cres must be provided when use_context is True"
            self.second_level_context_embedding = nn.Embedding(num_ref_cres, emb_dim)

        # CRE processing layers (epigenetics modulator layers)
        if use_context:
            self.cre_layers = nn.ModuleList(
                [
                    ContextFlashAttentionEncoderLayer(
                        d_model=emb_dim,
                        nhead=num_heads,
                        batch_first=True,
                        use_alibi=use_alibi,
                        mlp_dout=mlp_dout,
                        flash_attn_3=flash_attn_3,
                    )
                    for _ in range(num_layers - 1)
                ]
            )
        else:
            self.cre_layers = nn.ModuleList(
                [
                    FlashAttentionEncoderLayer(
                        d_model=emb_dim,
                        nhead=num_heads,
                        batch_first=True,
                        use_alibi=use_alibi,
                        mlp_dout=mlp_dout,
                    )
                    for _ in range(num_layers - 1)
                ]
            )

        # Gene processing layers (gene modulator layers)
        if only_cross_attention:
            self.gene_layers = nn.ModuleList(
                [
                    ContextFlashCrossAttentionEncoderLayer(
                        d_model=emb_dim,
                        nhead=num_heads,
                        batch_first=True,
                        use_alibi=use_alibi,
                        mlp_dout=mlp_dout,
                        cross_alibi=cross_alibi,
                        flash_attn_3=flash_attn_3,
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            self.gene_layers = nn.ModuleList(
                [
                    ContextFlashAttentionEncoderLayer(
                        d_model=emb_dim,
                        nhead=num_heads,
                        batch_first=True,
                        use_alibi=use_alibi,
                        mlp_dout=mlp_dout,
                        cross_alibi=cross_alibi,
                        flash_attn_3=flash_attn_3,
                    )
                    for _ in range(num_layers)
                ]
            )

    def forward(
        self,
        cre_x,
        gene_x,
        context=None,
        cre_padding_mask=None,
        gene_padding_mask=None,
        context_padding_mask=None,
        precision=None,
        cre_token_position=None,
        gene_token_position=None,
    ):
        """
        Forward pass with direct CRE→Gene coupling without intermediate storage.

        Args:
            cre_x: CRE sequence embeddings [batch, cre_seq_len, emb_dim]
            gene_x: Gene sequence embeddings [batch, gene_seq_len, emb_dim]
            context: Context tensor for CRE sequences (if use_context=True)
            cre_padding_mask: Padding mask for CRE sequences
            gene_padding_mask: Padding mask for gene sequences
            context_padding_mask: Padding mask for context
            precision: Computation precision

        Returns:
            gene_output: Final gene embeddings [batch, gene_seq_len, emb_dim]
        """

        # === Step 1: Setup context embedding ===
        context_embedding = None
        if self.use_context and context is not None:
            context_embedding = self.second_level_context_embedding(context)

        # === Step 2: Setup unpadding information ===
        cre_unpad_info = None
        gene_unpad_info = None
        context_unpad_info = None

        # Unpad CRE sequences once at the start
        if cre_padding_mask is not None:
            batch, cre_seqlen = cre_x.shape[:2]
            cre_padding_mask_inverted = ~cre_padding_mask
            cre_x_unpadded, cre_indices, cre_cu_seqlens, cre_max_seqlen_in_batch, _ = (
                unpad_input(cre_x, cre_padding_mask_inverted)
            )
            cre_unpad_info = {
                "indices": cre_indices,
                "cu_seqlens": cre_cu_seqlens,
                "max_seqlen": cre_max_seqlen_in_batch,
                "batch": batch,
                "seqlen": cre_seqlen,
            }
            cre_x = cre_x_unpadded

        # Unpad gene sequences once at the start
        if gene_padding_mask is not None:
            gene_batch, gene_seqlen = gene_x.shape[:2]
            gene_padding_mask_inverted = ~gene_padding_mask
            (
                gene_x_unpadded,
                gene_indices,
                gene_cu_seqlens,
                gene_max_seqlen_in_batch,
                _,
            ) = unpad_input(gene_x, gene_padding_mask_inverted)
            gene_unpad_info = {
                "indices": gene_indices,
                "cu_seqlens": gene_cu_seqlens,
                "max_seqlen": gene_max_seqlen_in_batch,
                "batch": gene_batch,
                "seqlen": gene_seqlen,
            }
            gene_x = gene_x_unpadded

        # Unpad context if needed
        if (
            self.use_context
            and context_padding_mask is not None
            and context_embedding is not None
        ):
            context_batch, context_seqlen = context_embedding.shape[:2]
            context_padding_mask_inverted = ~context_padding_mask
            (
                context_unpadded,
                context_indices,
                context_cu_seqlens,
                context_max_seqlen_in_batch,
                _,
            ) = unpad_input(context_embedding, context_padding_mask_inverted)
            context_unpad_info = {
                "indices": context_indices,
                "cu_seqlens": context_cu_seqlens,
                "max_seqlen": context_max_seqlen_in_batch,
                "batch": context_batch,
                "seqlen": context_seqlen,
            }
            context_embedding = context_unpadded

        # === Step 3: Setup residual connection for gene ===
        gene_res = gene_x.clone() if self.use_res else None

        # === Step 4: Process first gene layer with initial CRE ===
        # Process the first gene layer using the input CRE as context
        cre_current = cre_x  # Start with input CRE
        gene_current = gene_x  # Start with input gene

        # Apply first gene layer
        gene_current = self.gene_layers[0](
            gene_current,
            cre_current,
            gene_unpad_info=gene_unpad_info,
            context_unpad_info=cre_unpad_info,
            precision=precision,
        )

        # Apply residual connection if configured
        if self.use_res and gene_res is not None:
            gene_current = gene_current + gene_res

        # === Step 5: Process remaining layers with direct coupling ===
        # Process remaining layers: for each layer i, process CRE layer i, then Gene layer i+1
        for i in range(self.num_layers - 1):
            # Process CRE layer i
            if self.use_context:
                cre_current = self.cre_layers[i](
                    cre_current,
                    context_embedding,
                    unpad_info=cre_unpad_info,
                    context_unpad_info=context_unpad_info,
                    precision=precision,
                )
            else:
                cre_current = self.cre_layers[i](
                    cre_current, unpad_info=cre_unpad_info, precision=precision
                )

            # CRITICAL: Direct feed to gene layer (i+1) - NO INTERMEDIATE STORAGE
            # Process gene layer i+1 using the freshly computed CRE output
            gene_current = self.gene_layers[i + 1](
                gene_current,
                cre_current,  # Use fresh CRE output directly
                gene_unpad_info=gene_unpad_info,
                context_unpad_info=cre_unpad_info,
                precision=precision,
            )

            # Apply residual connection if configured
            if self.use_res and gene_res is not None:
                gene_current = gene_current + gene_res

        # === Step 6: Pad final output ===
        if gene_unpad_info is not None:
            gene_current = pad_input(
                gene_current,
                gene_unpad_info["indices"],
                gene_unpad_info["batch"],
                gene_unpad_info["seqlen"],
            )

        if gene_token_position is not None:
            gene_token_position = gene_token_position.long()
            # Use advanced indexing to get the correct positions for each sample
            batch_indices = torch.arange(
                gene_token_position.size(0), device=gene_token_position.device
            )
            gene_token_embedding = gene_current[batch_indices, gene_token_position, :]
            gene_token_embedding = gene_token_embedding.squeeze(1)
        else:
            gene_token_embedding = torch.zeros(
                gene_current.size(0), gene_current.size(2), device=gene_current.device
            )
        if cre_unpad_info is not None:
            cre_current = pad_input(
                cre_current,
                cre_unpad_info["indices"],
                cre_unpad_info["batch"],
                cre_unpad_info["seqlen"],
            )
        if cre_token_position is not None:
            cre_token_position = cre_token_position.long()
            # Use advanced indexing to get the correct positions for each sample
            batch_indices = torch.arange(
                cre_token_position.size(0), device=cre_token_position.device
            )
            cre_token_embedding = cre_current[batch_indices, cre_token_position, :]
            cre_token_embedding = cre_token_embedding.squeeze(1)
        else:
            cre_token_embedding = torch.zeros(
                cre_current.size(0), cre_current.size(2), device=cre_current.device
            )

        return gene_current, gene_token_embedding, cre_token_embedding

    def prepare_input(
        self,
        g_exp,
        gene_pooling,
        start_tkn=None,
        tissue_vector=None,
        padding_mask_gene=None,
    ):
        """
        Prepare the gene expression input by adding start tokens or registry tokens if needed.
        This method is kept for compatibility with existing code.
        """
        res = g_exp.clone()

        if gene_pooling == "start_token" and start_tkn is not None:
            start_token = start_tkn(g_exp)
            g_exp = torch.cat((start_token, g_exp), dim=1)
            res = torch.cat((torch.zeros_like(start_token), res), dim=1)
            if padding_mask_gene is not None:
                start_mask = torch.zeros(
                    (padding_mask_gene.size(0), 1),
                    dtype=padding_mask_gene.dtype,
                    device=padding_mask_gene.device,
                )
                new_mask = torch.cat((start_mask, padding_mask_gene), dim=1)
                padding_mask_gene = new_mask.clone()

        elif gene_pooling == "multi_registry" and start_tkn is not None:
            g_exp, res = start_tkn(g_exp, tissue_vector)
            if padding_mask_gene is not None:
                start_mask = torch.zeros(
                    (padding_mask_gene.size(0), 1),
                    dtype=padding_mask_gene.dtype,
                    device=padding_mask_gene.device,
                )
                new_mask = torch.cat((start_mask, padding_mask_gene), dim=1)
                padding_mask_gene = new_mask.clone()

        return g_exp, res, padding_mask_gene

    def pool_outputs(self, g_exp, gene_pooling, padding_mask_gene=None):
        """
        Pool the gene expression outputs based on the specified method.
        This method is kept for compatibility with existing code.
        """
        if gene_pooling == "mean" and padding_mask_gene is not None:
            comp_padding_mask_gene = ~padding_mask_gene
            g_exp = (
                g_exp * comp_padding_mask_gene.unsqueeze(-1)
            ) / comp_padding_mask_gene.sum(dim=1, keepdim=True).unsqueeze(-1)

        elif gene_pooling == "max" and padding_mask_gene is not None:
            inf_padding_mask = torch.zeros(padding_mask_gene.size()).to(
                padding_mask_gene.device
            )
            inf_padding_mask = inf_padding_mask.masked_fill(
                padding_mask_gene, float("-inf")
            )
            g_exp = g_exp + inf_padding_mask.unsqueeze(-1)
            g_exp = g_exp.max(dim=1).values

        elif gene_pooling in ["start_token", "multi_registry"]:
            g_exp = g_exp[:, 0, :]  # Only keep the start/registry token
        else:
            raise ValueError(f"Invalid gene pooling method: {gene_pooling}")

        return g_exp


class Seq2GenePredictorCombinedModulator(pl.LightningModule):
    """
    Standalone VariantFormer model with combined modulator for maximum memory efficiency.

    This model replaces the separate epigenetics_modulator and gene_modulator
    with a single CombinedModulator that processes sequences without storing
    intermediate tensors.

    Memory savings: Eliminates storage of 25 intermediate tensors per batch,
    saving approximately 3GB of GPU memory for typical batch sizes.
    """

    def __init__(
        self,
        num_tissues: int,
        emb_dim: int,
        gene_emb_dim: int,
        num_heads: int,
        num_layers: int,
        use_alibi: bool = True,
        mlp_dout: float = 0.1,
        weight_decay: float = 0.0,
        learning_rate: float = 1e-4,
        lr_scale: float = 1,
        use_context: bool = False,
        token_dim: int = 128,
        cre_tokenizer=None,
        gene_tokenizer=None,
        cre_tokenizer_train_mode="val",
        cre_tokenizer_val_mode="val",
        gene_tokenizer_train_mode="val",
        gene_tokenizer_val_mode="val",
        tissues: list = None,
        optimizer="adam",
        gene_pooling="mean",
        flash_attn_3=False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["cre_tokenizer", "gene_tokenizer"])
        # Save for optimizer
        self.precision = None
        self.lr_scale = lr_scale
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.pred = []
        self.target = []

        assert gene_pooling in [
            "mean",
            "max",
            "start_token",
            "multi_registry",
        ], "gene_pooling must be one of mean, max, start_token, or multi_registry"
        self.gene_pooling = gene_pooling
        if self.gene_pooling == "start_token":
            self.start_tkn = StartToken(emb_dim)
        elif self.gene_pooling == "multi_registry":
            self.start_tkn = MultiRegistry(num_tissues, emb_dim)
        else:
            self.start_tkn = None
        train_gene_tokenizer = kwargs.get("train_gene_tokenizer", False)
        self.train_gene_tokenizer = train_gene_tokenizer
        self.cre_tokenizer = cre_tokenizer
        if cre_tokenizer is not None:
            for param in self.cre_tokenizer.parameters():
                param.requires_grad = False

        self.gene_tokenizer = gene_tokenizer
        if gene_tokenizer is not None:
            if not train_gene_tokenizer:
                for param in self.gene_tokenizer.parameters():
                    param.requires_grad = False
        self.add_context_to_cres = kwargs.get("add_context_to_cres", False)
        if self.add_context_to_cres:
            self.add_context = AddContext(num_tissues, emb_dim)
        else:
            self.add_context = None
        self.cre_tokenizer_train_mode = cre_tokenizer_train_mode
        self.cre_tokenizer_val_mode = cre_tokenizer_val_mode
        self.gene_tokenizer_train_mode = gene_tokenizer_train_mode
        self.gene_tokenizer_val_mode = gene_tokenizer_val_mode
        self.emb_dim = emb_dim
        self.use_context = use_context
        self.tissues = tissues
        self.optimizer = optimizer
        self.scheduler = kwargs.get("scheduler", "plateau")
        self.warmup_ratio = kwargs.get("warmup_ratio", 0.1)
        self.min_lr = kwargs.get("min_lr", 1e-7)
        self.use_batching = kwargs.get("use_batching", False)
        self.use_res = kwargs.get("use_res", False)
        self.loss_fn = kwargs.get("loss_fn", "poisson")
        self.use_bigger_head = kwargs.get("use_bigger_head", False)

        self.multi_head = kwargs.get("multi_head", True)
        self.count_big_genes = 0
        self.only_cross_attention = kwargs.get("only_cross_attention", True)
        self.cross_alibi = kwargs.get("cross_alibi", False)
        if use_alibi == False:
            self.cross_alibi = False

        assert self.optimizer in ["adam", "adamw"], "Optimizer not supported"

        self.gene_map = nn.Linear(
            gene_emb_dim, emb_dim
        )  # Map the gene embedding to the same dimension as the cis-regulatory elements

        if token_dim != emb_dim:
            self.cre_map = nn.Linear(token_dim, emb_dim)

        self.combined_modulator = CombinedModulator(
            emb_dim=emb_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            use_alibi=use_alibi,
            mlp_dout=mlp_dout,
            use_context=use_context,
            num_ref_cres=len(constants.REF_CREs) if use_context else None,
            only_cross_attention=self.only_cross_attention,
            use_res=self.use_res,
            cross_alibi=self.cross_alibi,
            flash_attn_3=flash_attn_3,
        )

        self.tissue_heads = TissueExpressionHeads(
            emb_dim,
            num_tissues,
            use_bigger_head=self.use_bigger_head,
            multi_head=self.multi_head,
            mlp_dout=mlp_dout,
            loss_fn=self.loss_fn,
            head_type=kwargs.get("head_type", "mlp"),
        )

        if self.loss_fn == "poisson":
            self.gene_loss = nn.PoissonNLLLoss(
                log_input=False, full=True, reduction="none"
            )
        elif self.loss_fn == "mse":
            self.gene_loss = nn.MSELoss(reduction="none")

    def forward(
        self,
        inp,
        attention_mask,
        tissue_vector,
        cre_context,
        strand,
        gene_embedding,
        gene_att_mask,
        return_embedding=False,
        get_all=False,
        **kwargs,
    ):
        """
        Forward pass using the combined modulator for memory-optimized processing.
        """

        # === Data preparation (same as original) ===
        mode_finetune = kwargs.get("mode_finetune", False)
        only_embedding = kwargs.get("only_embedding", False)

        # Transform CRE sequences
        x, padding_mask_cres, context, precision, _ = self.transform_with_batching(
            inp,
            attention_mask,
            tissue_vector,
            cre_context,
            strand,
            embedder=self.cre_tokenizer,
        )

        # Transform gene sequences
        dummy_gene_tissue_vector = [
            torch.zeros(gene_embedding[i].size(0)) for i in range(len(gene_embedding))
        ]
        dummy_cre_context = [
            torch.zeros(gene_embedding[i].size(0)) for i in range(len(gene_embedding))
        ]
        if self.gene_tokenizer is not None:
            detach = not self.train_gene_tokenizer
            x_gene, padding_mask_gene, _, _, gene_donors = self.transform_with_batching(
                gene_embedding,
                gene_att_mask,
                dummy_gene_tissue_vector,
                dummy_cre_context,
                strand,
                embedder=self.gene_tokenizer,
                detach_embedding=detach,
            )
        else:
            x_gene, padding_mask_gene, _, _, gene_donors = self.transform_with_batching(
                gene_embedding,
                gene_att_mask,
                dummy_gene_tissue_vector,
                dummy_cre_context,
                strand,
                embedder=self.cre_tokenizer,
            )

        # Subset to common donors
        donors = gene_donors  # sorted(list(set(donors) & set(gene_donors)))
        x, padding_mask_cres, context = (
            x[donors],
            padding_mask_cres[donors],
            context[donors],
        )
        # x_gene, padding_mask_gene = x_gene[donors], padding_mask_gene[donors]

        tissue_vector_expanded = [tissue_vector[i] for i in donors]

        if x.size(-1) != self.emb_dim:
            x = self.cre_map(x)
        x_gene = self.gene_map(x_gene)

        # Expand tissue vectors
        gene_token_position = kwargs.get("gene_token_position", None)
        cre_token_position = kwargs.get("cre_token_position", None)
        cre_token_position_repeat = []
        gene_token_position_repeat = []
        x_repeat, padding_mask_repeat, context_repeat = [], [], []
        x_gene_repeat, padding_mask_gene_repeat = [], []
        tissue_vector_repeat = []
        for i, tissue in enumerate(tissue_vector_expanded):
            num_tissues_sample = len(tissue)
            x_repeat.append(x[i : i + 1].repeat(num_tissues_sample, 1, 1))
            padding_mask_repeat.append(
                padding_mask_cres[i : i + 1].repeat(num_tissues_sample, 1)
            )
            context_repeat.append(context[i : i + 1].repeat(num_tissues_sample, 1))
            x_gene_repeat.append(x_gene[i : i + 1].repeat(num_tissues_sample, 1, 1))
            padding_mask_gene_repeat.append(
                padding_mask_gene[i : i + 1].repeat(num_tissues_sample, 1)
            )
            tissue_vector_repeat.extend([t for t in tissue])

            # Repeat the token positions for each tissue
            if cre_token_position is not None:
                cre_token_position_repeat.append(
                    cre_token_position[i].repeat(num_tissues_sample)
                )
            if gene_token_position is not None:
                gene_token_position_repeat.append(
                    gene_token_position[i].repeat(num_tissues_sample)
                )

        x = torch.cat(x_repeat, dim=0)
        padding_mask_cres = torch.cat(padding_mask_repeat, dim=0)
        context = torch.cat(context_repeat, dim=0)
        x_gene = torch.cat(x_gene_repeat, dim=0)
        padding_mask_gene = torch.cat(padding_mask_gene_repeat, dim=0)
        tissue_vector_final = torch.tensor(
            tissue_vector_repeat, dtype=torch.long, device=x_gene.device
        ).unsqueeze(1)
        cre_token_position = (
            torch.cat(cre_token_position_repeat, dim=0)
            if cre_token_position is not None
            else None
        )
        gene_token_position = (
            torch.cat(gene_token_position_repeat, dim=0)
            if gene_token_position is not None
            else None
        )

        # Adjust gene token positions if start tokens are used
        if gene_token_position is not None and self.start_tkn is not None:
            gene_token_position = gene_token_position + 1

        # --- Add tissue specific embedding to the cis-regulatory elements ---
        if self.add_context_to_cres:
            x = self.add_context(x, tissue_vector_final)

        # === Gene input preparation ===
        g_exp = x_gene.clone()

        start_tkn = self.start_tkn

        g_exp, res, padding_mask_gene = self.combined_modulator.prepare_input(
            g_exp,
            self.gene_pooling,
            start_tkn=start_tkn,
            tissue_vector=tissue_vector_final,
            padding_mask_gene=padding_mask_gene,
        )

        # === MEMORY-OPTIMIZED COMBINED PROCESSING ===
        g_exp, gene_token_embedding, cre_token_embedding = self.combined_modulator(
            cre_x=x,
            gene_x=g_exp,
            context=context,
            cre_padding_mask=padding_mask_cres,
            gene_padding_mask=padding_mask_gene,
            context_padding_mask=padding_mask_cres,
            precision=precision,
            cre_token_position=cre_token_position,
            gene_token_position=gene_token_position,
        )

        # Pool the outputs
        g_exp = self.combined_modulator.pool_outputs(
            g_exp, self.gene_pooling, padding_mask_gene=padding_mask_gene
        )

        if only_embedding:
            D = {"embedding": g_exp, "donors": donors}
            return D

        # === Tissue specific gene expression prediction ===
        pred_gene_exp = self.tissue_heads(g_exp, tissue_vector_final)

        # === Return results ===
        if return_embedding:
            return (
                pred_gene_exp,
                donors,
                g_exp,
                gene_token_embedding,
                cre_token_embedding,
            )
        else:
            return pred_gene_exp, donors

    def transform_with_batching(
        self,
        x,
        attention_mask,
        tissue_context,
        ref_labels_tensor,
        strand,
        embedder,
        detach_embedding=True,
    ):
        """
        Process sequences (CREs or genes) in batches and embed them using the provided embedder.
        Handles padding and manages memory constraints by chunking large sequences.
        """
        try:
            precision = precision2dtype(self.trainer.precision)
        except Exception:
            precision = torch.float32

        if precision in [torch.float16, torch.bfloat16]:
            precision = None
        else:
            precision = torch.float32

        total_sequence_length = sum(v.shape[0] for v in x)
        exceeds_memory_limit = (
            total_sequence_length > MAX_WINDOW_SIZE
            and not detach_embedding
            and self.training
        )

        if exceeds_memory_limit:
            donor_windows = [v.shape[0] for v in x]
            donor_list = [np.argmax(donor_windows).item()]
            self.count_big_genes += 1
        else:
            donor_list = list(range(len(x)))

        max_cre_length = max(v.shape[0] for v in x)
        chunk_size = min(MAX_CHUNK_SIZE, max_cre_length)

        embedded_sequences, attention_masks_out, reference_labels = [], [], []

        for donor_id in donor_list:
            chunk_embeddings = []
            for chunk_start in range(0, len(x[donor_id]), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(x[donor_id]))
                x_chunk, mask_chunk, tissue_chunk, labels_chunk = (
                    x[donor_id][chunk_start:chunk_end],
                    attention_mask[donor_id][chunk_start:chunk_end],
                    tissue_context[donor_id][chunk_start:chunk_end],
                    ref_labels_tensor[donor_id][chunk_start:chunk_end],
                )

                emb = embedder(
                    x_chunk,
                    mask_chunk,
                    tissue_chunk,
                    context=labels_chunk,
                    only_embed=True,
                    precision=precision,
                )
                emb = emb.detach() if detach_embedding else emb.clone()
                chunk_embeddings.append(emb)

            donor_embedding = torch.cat(chunk_embeddings, dim=0)
            donor_attention_mask = torch.zeros(
                donor_embedding.size(0), device=donor_embedding.device
            )
            donor_ref_labels = ref_labels_tensor[donor_id].to(donor_embedding.device)

            if max_cre_length > donor_embedding.size(0):
                padding_length = max_cre_length - donor_embedding.size(0)
                padding = torch.zeros(
                    padding_length,
                    donor_embedding.size(1),
                    donor_embedding.size(2),
                    device=donor_embedding.device,
                    dtype=donor_embedding.dtype,
                )
                donor_embedding = torch.cat((donor_embedding, padding), dim=0)
                donor_attention_mask = torch.cat(
                    (
                        donor_attention_mask,
                        torch.ones(padding_length, device=donor_embedding.device),
                    )
                )
                donor_ref_labels = torch.cat(
                    (
                        donor_ref_labels,
                        torch.zeros(padding_length, device=donor_embedding.device),
                    )
                )

            assert (
                donor_embedding.size(1) == 1
            ), "donor_embedding has 1 strand because strand is picked in dataloader"
            donor_embedding = donor_embedding[:, 0, :].squeeze(1)

            embedded_sequences.append(donor_embedding)
            attention_masks_out.append(donor_attention_mask)
            reference_labels.append(donor_ref_labels)

        X = torch.stack(embedded_sequences)
        cre_attention_mask = torch.stack(attention_masks_out).to(torch.bool)
        ref_labels = torch.stack(reference_labels).to(torch.long)

        return X, cre_attention_mask, ref_labels, precision, donor_list

    def _set_state_of_encoders_train(self):
        if self.cre_tokenizer_train_mode == "train":
            self.cre_tokenizer.train()
        else:
            self.cre_tokenizer.eval()

        if self.gene_tokenizer is not None:
            if self.train_gene_tokenizer:
                if self.gene_tokenizer_train_mode == "train":
                    self.gene_tokenizer.train()
                else:
                    self.gene_tokenizer.eval()
            else:
                self.gene_tokenizer.eval()

    def _set_state_of_encoders_val(self):
        if self.cre_tokenizer_val_mode == "train":
            self.cre_tokenizer.train()
        else:
            self.cre_tokenizer.eval()
        if self.gene_tokenizer is not None:
            if self.gene_tokenizer_val_mode == "train":
                self.gene_tokenizer.train()
            else:
                self.gene_tokenizer.eval()

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        """Predict step"""
        self._set_state_of_encoders_val()  # Set the state of the encoders
        self.eval()
        if self.vep:
            return self.variant_prediction(batch)
        # Unpack batch. gene_exp_original is batch[5]
        # Unpack batch
        x = batch["cre_sequences"]
        attention_mask = batch["cre_attention_masks"]
        tissue_vector = batch["tissue_context"]
        context = batch["ref_cre_labels"]
        strand = batch["strand_val"]
        gene_embedding = batch["gene_embeddings"]
        gene_att_mask = batch["gene_attention_masks"]

        pred_gene_exp, donors, embd, _, _ = self(
            x,
            attention_mask,
            tissue_vector,
            context,
            strand,
            gene_embedding,
            gene_att_mask,
            return_embedding=True,
        )

        assert len(donors) == len(x), "Number of donors and CREs do not match"
        pred_gene_exp = (
            pred_gene_exp.detach().cpu().float().numpy()
        )  # Detach the predictions from the graph
        embd = (
            embd.detach().cpu().float().numpy()
        )  # Detach the embeddings from the graph
        pred_gene_exp_list = []
        embeddings = []
        start_idx = 0
        for sample_id in donors:
            num_tissues = len(tissue_vector[sample_id])
            end_idx = start_idx + num_tissues
            pred_gene_exp_list.append(pred_gene_exp[start_idx:end_idx])
            embeddings.append(embd[start_idx:end_idx])
            start_idx = end_idx

        D = {
            "pred_gene_exp": pred_gene_exp_list,
            "embeddings": embeddings,
            "batch_idx": batch_idx,
            "dataloader_idx": dataloader_idx,
        }
        return D

    def variant_prediction(self, batch):
        """Get all gene expression from all heads"""
        x = batch["cre_sequences"]  # CRE sequences
        num_batch = len(x)
        if num_batch == 0:
            return {
                "pred_gene_exp": [],
                "embd": [],
                "variant_type": batch["variant_type"],
                "gene_token_embedding": [],
                "cre_token_embedding": [],
            }
        attention_mask = batch[
            "cre_attention_masks"
        ]  # Attention masks for CRE sequences
        tissue_vector = batch["tissue_context"]  # Tissue identifiers
        context = batch["ref_labels"]  # CRE context labels
        strand = batch["strand"]  # Strand information
        gene_embedding = batch["gene_embeddings"]  # Gene sequences
        gene_att_mask = batch[
            "gene_attention_masks"
        ]  # Attention masks for gene sequences
        cre_token_position = batch["cre_token_position"]  # CRE token position
        gene_token_position = batch["gene_token_position"]  # Gene token position
        assert (
            len(cre_token_position) == 3
        ), "there should be 3 samples in the batch for ref, het, hom"
        assert (
            len(gene_token_position) == 3
        ), "there should be 3 samples in the batch for ref, het, hom"
        if torch.isnan(cre_token_position).any():
            cre_token_position = None
        if torch.isnan(gene_token_position).any():
            gene_token_position = None
        # Process the input in batches of tissues
        pred_gene_exp_list = []
        embd_list = []
        gene_token_embedding_list = []
        cre_token_embedding_list = []
        for i in range(num_batch):
            pred_gene_exp, _, embd, gene_token_embedding, cre_token_embedding = self(
                x[i : i + 1],
                attention_mask[i : i + 1],
                tissue_vector[i : i + 1],
                context[i : i + 1],
                strand[i : i + 1],
                gene_embedding[i : i + 1],
                gene_att_mask[i : i + 1],
                return_embedding=True,
                cre_token_position=cre_token_position[i : i + 1]
                if cre_token_position is not None
                else None,
                gene_token_position=gene_token_position[i : i + 1]
                if gene_token_position is not None
                else None,
            )
            pred_gene_exp_list.append(pred_gene_exp)
            embd_list.append(embd)
            gene_token_embedding_list.append(gene_token_embedding)
            cre_token_embedding_list.append(cre_token_embedding)
        pred_gene_exp = torch.cat(pred_gene_exp_list, dim=0)
        embd = torch.cat(embd_list, dim=0)
        gene_token_embedding = torch.cat(gene_token_embedding_list, dim=0)
        cre_token_embedding = torch.cat(cre_token_embedding_list, dim=0)
        pred_gene_exp = (
            pred_gene_exp.cpu().float().numpy()
        )  # (num_batch*num_tissues, 1)
        embd = embd.cpu().float().numpy()  # (num_batch*num_tissues, emb_dim)
        gene_token_embedding = (
            gene_token_embedding.cpu().float().numpy()
        )  # (num_batch*num_tissues, emb_dim)
        cre_token_embedding = (
            cre_token_embedding.cpu().float().numpy()
        )  # (num_batch*num_tissues, emb_dim)
        pred_gene_exp_list = []
        embd_list = []
        gene_token_embedding_list = []
        cre_token_embedding_list = []
        start_i = 0
        # Split the predictions and embeddings into lists for each batch
        for num_i in range(num_batch):
            num_batch_tissues = len(tissue_vector[num_i])
            end_i = start_i + num_batch_tissues
            pred_gene_exp_list.append(pred_gene_exp[start_i:end_i])
            embd_list.append(embd[start_i:end_i])
            gene_token_embedding_list.append(gene_token_embedding[start_i:end_i])
            cre_token_embedding_list.append(cre_token_embedding[start_i:end_i])
            start_i = end_i
        D = {
            "pred_gene_exp": pred_gene_exp_list,
            "embd": embd_list,
            "variant_type": batch["variant_type"],
            "gene_token_embedding": gene_token_embedding_list,
            "cre_token_embedding": cre_token_embedding_list,
        }
        return D
