"A PyTorch Lightning module for the Seq2GenePredictor model"

import lightning.pytorch as pl  # type: ignore
import torch  # type: ignore
import torch.nn as nn  # type: ignore
from seq2gene.modules.layers import (
    EpigeneticsModulator,
    GeneModulator,
    TissueExpressionHeads,
    MultiRegistry,
    StartToken,
    AddContext,
    ConcatTissueContext,
)
from utils import constants
from utils.functions import precision2dtype
import numpy as np  # type: ignore

# torch.set_float32_matmul_precision('medium')
MAX_WINDOW_SIZE = 300
MAX_CHUNK_SIZE = 1024


class Seq2GenePredictor(pl.LightningModule):
    "A PyTorch Lightning module for the Seq2GenePredictor model"

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
        self.concat_tissue_context_cres = kwargs.get(
            "concat_tissue_context_cres", False
        )
        self.concat_tissue_context_genes = kwargs.get(
            "concat_tissue_context_genes", False
        )
        if self.concat_tissue_context_cres:
            assert (
                not self.add_context_to_cres
            ), "add_context_to_cres and concat_tissue_context_cres cannot both be True"
            self.concat_tissue_cres = ConcatTissueContext(num_tissues, emb_dim)
        else:
            self.concat_tissue_cres = None
        if self.concat_tissue_context_genes:
            assert (
                (self.gene_pooling != "start_token")
                and (self.gene_pooling != "multi_registry")
            ), "concat_tissue_context_genes is not supported with gene_pooling set to start_token or multi_registry"
            self.concat_tissue_genes = ConcatTissueContext(num_tissues, emb_dim)
        else:
            self.concat_tissue_genes = None
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

        self.epigenetics_modulator = EpigeneticsModulator(
            emb_dim=emb_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            use_alibi=use_alibi,
            mlp_dout=mlp_dout,
            use_context=use_context,
            num_ref_cres=len(constants.REF_CREs) if use_context else None,
            flash_attn_3=flash_attn_3,
        )

        self.gene_modulator = GeneModulator(
            emb_dim=emb_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            use_alibi=use_alibi,
            mlp_dout=mlp_dout,
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
        """Forward pass"""
        # extract the embeddings of cis-regulatory elements surrounding the gene
        x, padding_mask, context, precision, _ = self.transform_with_batching(
            inp,
            attention_mask,
            tissue_vector,
            cre_context,
            strand,
            embedder=self.cre_tokenizer,
        )

        # Extract the gene sequence parsed as 200 token long windows, where x_gene.size(1) is the number of windows.
        dummy_gene_tissue_vector = [
            torch.zeros(gene_embedding[i].size(0)) for i in range(len(gene_embedding))
        ]
        dummy_cre_context = [
            torch.zeros(gene_embedding[i].size(0)) for i in range(len(gene_embedding))
        ]
        if self.gene_tokenizer is not None:
            if self.train_gene_tokenizer:
                detach = False
            else:
                detach = True
            x_gene, padding_mask_gene, _, _, donors = self.transform_with_batching(
                gene_embedding,
                gene_att_mask,
                dummy_gene_tissue_vector,
                dummy_cre_context,
                strand,
                embedder=self.gene_tokenizer,
                detach_embedding=detach,
            )
        else:
            x_gene, padding_mask_gene, _, _, donors = self.transform_with_batching(
                gene_embedding,
                gene_att_mask,
                dummy_gene_tissue_vector,
                dummy_cre_context,
                strand,
                embedder=self.cre_tokenizer,
            )

        x, padding_mask, context = (
            x[donors],
            padding_mask[donors],
            context[donors],
        )  # subset the batch to the donors from the gene embedding
        tissue_vector = [tissue_vector[i] for i in donors]

        # --- Special case: generate predictions for all tissues from a single sample ---
        if get_all:
            # This mode is only supported for single-sample inputs
            assert len(x) == 1, "get_all is only supported for one sample"

            # How many tissues we need to generate predictions for
            num_tissues = self.num_tissues

            # Repeat the CRE data for each tissue
            x = x.repeat(num_tissues, 1, 1)  # [num_tissues, num_cres, emb_dim]
            padding_mask = padding_mask.repeat(num_tissues, 1)  # [num_tissues, seq_len]
            context = context.repeat(num_tissues, 1)  # [num_tissues, num_cres]

            # Repeat the gene data for each tissue
            x_gene = x_gene.repeat(
                num_tissues, 1, 1
            )  # [num_tissues, num_windows, emb_dim]
            padding_mask_gene = padding_mask_gene.repeat(
                num_tissues, 1
            )  # [num_tissues, num_windows]

            # Create a tissue vector with one entry per tissue (0 to num_tissues-1)
            tissue_vector = (
                torch.arange(num_tissues).unsqueeze(1).to(x.device)
            )  # [num_tissues, 1]
        else:
            # --- The input embedding ---
            # Mapping the input to the embedding dimension
            if x.size(-1) != self.emb_dim:
                x = self.cre_map(x)

            # --- Map the gene embedding to the same dimension as the cis-regulatory elements ---
            x_gene = self.gene_map(x_gene)

            # --- combine multiple tissues into a single batch ---
            gene_token_position = kwargs.get("gene_token_position", None)
            cre_token_position = kwargs.get("cre_token_position", None)
            x_repeat = []
            padding_mask_repeat = []
            context_repeat = []
            x_gene_repeat = []
            padding_mask_gene_repeat = []
            tissue_vector_repeat = []
            cre_token_position_repeat = []
            gene_token_position_repeat = []
            for i, tissue in enumerate(tissue_vector):
                num_tissues = len(tissue)
                # Repeat the CRE data for each tissue
                x_repeat.append(x[i : i + 1].repeat(num_tissues, 1, 1))
                padding_mask_repeat.append(
                    padding_mask[i : i + 1].repeat(num_tissues, 1)
                )
                context_repeat.append(context[i : i + 1].repeat(num_tissues, 1))
                # Repeat the gene data for each tissue
                x_gene_repeat.append(x_gene[i : i + 1].repeat(num_tissues, 1, 1))
                padding_mask_gene_repeat.append(
                    padding_mask_gene[i : i + 1].repeat(num_tissues, 1)
                )

                # Create a tissue vector with one entry per tissue (0 to num_tissues-1)
                tissue_vector_repeat.extend([t for t in tissue])

                # Repeat the token positions for each tissue
                if cre_token_position is not None:
                    cre_token_position_repeat.append(
                        cre_token_position[i].repeat(num_tissues)
                    )
                if gene_token_position is not None:
                    gene_token_position_repeat.append(
                        gene_token_position[i].repeat(num_tissues)
                    )

            x = torch.cat(x_repeat, dim=0)
            padding_mask = torch.cat(padding_mask_repeat, dim=0)
            context = torch.cat(context_repeat, dim=0)
            x_gene = torch.cat(x_gene_repeat, dim=0)
            padding_mask_gene = torch.cat(padding_mask_gene_repeat, dim=0)
            tissue_vector = torch.tensor(
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

        # --- Add tissue specific embedding to the cis-regulatory elements ---
        mode_finetune = kwargs.get("mode_finetune", False)
        if self.add_context_to_cres:
            x = self.add_context(x, tissue_vector)

        # --- Concatenate the tissue context to the cis-regulatory elements and genes ---
        if self.concat_tissue_context_cres:
            if not mode_finetune:
                x, padding_mask_cres = self.concat_tissue_cres(
                    x, tissue_vector, padding_mask
                )  # (x.size(0), 1 + x.size(1), emb_dim), (x.size(0), 1 + x.size(1))
            else:
                concat_tissue_cres = kwargs.get("concat_tissue_cres", None)
                assert (
                    concat_tissue_cres is not None
                ), "concat_tissue_cres must be provided in finetune mode"
                x, padding_mask_cres = concat_tissue_cres(
                    x, tissue_vector, padding_mask
                )  # (x.size(0), 1 + x.size(1), emb_dim), (x.size(0), 1 + x.size(1))
        else:
            padding_mask_cres = padding_mask
        if self.concat_tissue_context_genes:
            if not mode_finetune:
                x_gene, padding_mask_gene = self.concat_tissue_genes(
                    x_gene, tissue_vector, padding_mask_gene
                )  # (x.size(0), 1 + x.size(1), emb_dim), (x.size(0), 1 + x.size(1))
            else:
                concat_tissue_genes = kwargs.get("concat_tissue_genes", None)
                assert (
                    concat_tissue_genes is not None
                ), "concat_tissue_genes must be provided in finetune mode"
                x_gene, padding_mask_gene = concat_tissue_genes(
                    x_gene, tissue_vector, padding_mask_gene
                )  # (x.size(0), 1 + x.size(1), emb_dim), (x.size(0), 1 + x.size(1))

        # --- The epigenetics modulator ---
        modulator_outputs = self.epigenetics_modulator(
            x,
            context,
            src_key_padding_mask=padding_mask_cres,
            precision=precision,
            context_padding_mask=padding_mask,
            keep_intermediates_unpadded=True,
        )

        # --- The gene modulator ---
        g_exp = x_gene.clone()

        if not mode_finetune:
            start_tkn = self.start_tkn
        else:
            start_tkn = kwargs.get("start_tkn", None)
            assert start_tkn is not None, "start_tkn must be provided in finetune mode"

        # Prepare input with start/registry tokens if needed
        g_exp, res, padding_mask_gene = self.gene_modulator.prepare_input(
            g_exp,
            self.gene_pooling,
            start_tkn=start_tkn,
            tissue_vector=tissue_vector,
            padding_mask_gene=padding_mask_gene,
        )

        # Apply gene modulation
        g_exp = self.gene_modulator(
            g_exp,
            modulator_outputs,
            res=res,  # Pass the residual connection reference
            padding_mask=padding_mask_cres,
            src_key_padding_mask=padding_mask_gene,
            precision=precision,
        )

        if gene_token_position is not None:
            gene_token_position = gene_token_position.long()
            gene_token_position = (
                gene_token_position + 1
                if self.start_tkn is not None
                else gene_token_position
            )
            # Use advanced indexing to get the correct positions for each sample
            batch_indices = torch.arange(
                gene_token_position.size(0), device=gene_token_position.device
            )
            gene_token_embedding = g_exp[batch_indices, gene_token_position, :]
            gene_token_embedding = gene_token_embedding.squeeze(1)
        else:
            gene_token_embedding = torch.zeros(
                g_exp.size(0), g_exp.size(2), device=g_exp.device
            )
        if cre_token_position is not None:
            cre_token_position = cre_token_position.long()
            # Use advanced indexing to get the correct positions for each sample
            batch_indices = torch.arange(
                cre_token_position.size(0), device=cre_token_position.device
            )
            cre_final_output = modulator_outputs["final_output"][
                batch_indices, cre_token_position, :
            ]
        else:
            cre_final_output = torch.zeros(
                g_exp.size(0), g_exp.size(2), device=g_exp.device
            )

        # Pool the outputs based on the configured method
        g_exp = self.gene_modulator.pool_outputs(
            g_exp, self.gene_pooling, padding_mask_gene=padding_mask_gene
        )

        if kwargs.get("only_embedding", False):
            D = {"embedding": g_exp, "donors": donors}
            return D

        # --- The tissue specific gene expression ---
        pred_gene_exp = self.tissue_heads(g_exp, tissue_vector)

        # --- The output ---
        if return_embedding:
            return pred_gene_exp, donors, g_exp, gene_token_embedding, cre_final_output
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

        Args:
            x: List of input sequences [batch_size, seq_len, dim]
            attention_mask: Attention masks for input sequences
            tissue_context: Tissue context for each token
            ref_labels_tensor: Reference CRE labels
            strand: Strand information
            embedder: Model to use for embedding sequences
            detach_embedding: Whether to detach embeddings (True for inference, False for training)

        Returns:
            X: Batched and embedded sequences
            cre_attention_mask: Padding mask for the sequences
            ref_labels: Reference CRE labels
            precision: Precision used for computation
            donor_list: List of donors that were processed
        """
        # --- Determine computation precision ---
        try:
            precision = precision2dtype(self.trainer.precision)
        except Exception:
            precision = torch.float32

        # Override precision for mixed precision
        if precision in [torch.float16, torch.bfloat16]:
            precision = None
        else:
            precision = torch.float32

        # --- Determine which samples to process ---
        # For very large inputs during training, we may need to limit processing to avoid OOM errors
        total_sequence_length = sum(v.shape[0] for v in x)
        exceeds_memory_limit = (
            total_sequence_length > MAX_WINDOW_SIZE
            and not detach_embedding
            and self.training
        )

        if exceeds_memory_limit:
            # Select only the largest sequence to process
            donor_windows = [v.shape[0] for v in x]
            donor_list = [np.argmax(donor_windows).item()]  # pick the larger gene
            self.count_big_genes += 1
        else:
            # Process all sequences
            donor_list = list(range(len(x)))

        # --- Process each donor in chunks ---
        max_cre_length = max(v.shape[0] for v in x)
        chunk_size = min(MAX_CHUNK_SIZE, max_cre_length)

        embedded_sequences = []
        attention_masks = []
        reference_labels = []

        for donor_id in donor_list:
            # Process the donor sequence in chunks to avoid memory issues
            chunk_embeddings = []
            for chunk_start in range(0, len(x[donor_id]), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(x[donor_id]))

                # Extract chunk data
                x_chunk = x[donor_id][chunk_start:chunk_end]
                mask_chunk = attention_mask[donor_id][chunk_start:chunk_end]
                tissue_chunk = tissue_context[donor_id][chunk_start:chunk_end]
                labels_chunk = ref_labels_tensor[donor_id][chunk_start:chunk_end]

                # Embed the chunk
                emb = embedder(
                    x_chunk,
                    mask_chunk,
                    tissue_chunk,
                    context=labels_chunk,
                    only_embed=True,
                    precision=precision,
                )

                # Detach or clone based on setting
                if detach_embedding:
                    emb = emb.detach()
                else:
                    emb = emb.clone()

                chunk_embeddings.append(emb)

            # Combine all chunks for this donor
            donor_embedding = torch.cat(chunk_embeddings, dim=0)

            # Create base attention mask (zeros = not masked)
            donor_attention_mask = torch.zeros(
                donor_embedding.size(0), device=donor_embedding.device
            )

            # Get reference labels for this donor
            donor_ref_labels = ref_labels_tensor[donor_id].to(donor_embedding.device)

            # --- Handle padding if needed ---
            if max_cre_length > donor_embedding.size(0):
                padding_length = max_cre_length - donor_embedding.size(0)

                # Create padding
                padding = torch.zeros(
                    padding_length,
                    donor_embedding.size(1),
                    donor_embedding.size(2),
                    device=donor_embedding.device,
                    dtype=donor_embedding.dtype,
                )

                # Apply padding
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

            # Collect results
            embedded_sequences.append(donor_embedding)
            attention_masks.append(donor_attention_mask)
            reference_labels.append(donor_ref_labels)

        # --- Stack and prepare final outputs ---
        X = torch.stack(embedded_sequences)
        cre_attention_mask = torch.stack(attention_masks).to(
            torch.bool
        )  # False = unmasked, True = masked
        ref_labels = torch.stack(reference_labels).to(torch.long)

        return X, cre_attention_mask, ref_labels, precision, donor_list

    def _set_state_of_encoders_train(self):
        # self.cre_tokenizer.to(self.device) # Pretrained encoder of tokens to tokenize the regulatory sequences
        if self.cre_tokenizer_train_mode == "train":
            self.cre_tokenizer.train()
        else:
            self.cre_tokenizer.eval()

        if self.gene_tokenizer is not None:
            # self.gene_tokenizer.to(self.device) # Pretrained encoder of tokens to tokenize the gene sequences
            if self.train_gene_tokenizer:
                if self.gene_tokenizer_train_mode == "train":
                    self.gene_tokenizer.train()
                else:
                    self.gene_tokenizer.eval()
            else:
                self.gene_tokenizer.eval()

    def _set_state_of_encoders_val(self):
        # self.cre_tokenizer.to(self.device)
        if self.cre_tokenizer_val_mode == "train":
            self.cre_tokenizer.train()
        else:
            self.cre_tokenizer.eval()
        if self.gene_tokenizer is not None:
            # self.gene_tokenizer.to(self.device)
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
        attention_mask = batch["cre_attention_mask"]
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
