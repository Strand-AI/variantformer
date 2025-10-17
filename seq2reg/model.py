"Implementation of the Seq2Reg model for predicting regulatory states from DNA sequences"

import torch
import torch.nn as nn
import lightning.pytorch as pl
import utils.constants as constants
from seq2reg.losses import get_loss_fn
from seq2reg.modules import FlashTransformerLayer, ContextFlashAttentionEncoderLayer
import math
import numpy as np

torch.set_float32_matmul_precision("medium")


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError(
            "Cannot use sin/cos positional encoding with "
            "odd dim (got dim={:d})".format(d_model)
        )
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp(
        (
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )
    )
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


class Seq2RegPredictor(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        num_tissues: int,
        num_classes: int,
        learning_rate: float,
        loss_fn: str,
        seq_pool: str = "mean",
        cre_type: str = "multi",
        token_length: int = None,
        use_context: bool = False,
        positional_encoding: str = "sinusoidal",
        use_flash: bool = False,
        majority_weight: float = None,
        weight_decay: float = 0.0,
        lr_scale: float = 1.0,
        strand_agg: str = "mean",
        expand_context: bool = False,
        mlp_dout: float = 0.1,
        tissues: list = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # print(token_length)

        # Embedding layer for DNA tokens
        assert use_flash, "Only Flash is supported"

        self.weight_decay = weight_decay
        self.lr_scale = lr_scale
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.majority_weight = majority_weight
        self.expand_context_type = expand_context
        self.num_classes = num_classes
        self.tissues = tissues
        self.use_dual_loss = kwargs.get("use_dual_loss", False)
        self.use_temp = kwargs.get("use_temp", False)
        self.fraction = kwargs.get("fraction", None)
        # if self.use_dual_loss:
        #    print("Using dual loss")
        self.reduction = kwargs.get("reduction", "sum")
        if self.use_dual_loss:
            if self.use_temp:
                self.logit_scale = nn.Parameter(
                    torch.ones([]) * np.log(1 / 0.07)
                ).requires_grad_(True)

        if use_context:
            self.context_embedding = nn.Embedding(
                len(constants.REF_CREs), embedding_dim
            )
            if expand_context:
                self.expand_context = nn.Linear(1, token_length)
        assert positional_encoding in [
            "sinusoidal",
            "alibi",
        ], "Position encoding must be either 'sinusoidal' or 'alibi'"
        self.pos_encoding_type = positional_encoding
        if positional_encoding == "sinusoidal":
            self.position_encoding = positionalencoding1d(embedding_dim, token_length)
            use_alibi = False
        else:
            use_alibi = True

        # Transformer Encoder
        if use_context:
            if use_flash:
                self.transformer_encoder = nn.ModuleList(
                    [
                        ContextFlashAttentionEncoderLayer(
                            d_model=embedding_dim,
                            nhead=num_heads,
                            batch_first=True,
                            use_alibi=use_alibi,
                            mlp_dout=mlp_dout,
                        )
                        for _ in range(num_layers)
                    ]
                )

        else:
            if use_flash:
                self.transformer_encoder = nn.ModuleList(
                    [
                        FlashTransformerLayer(
                            d_model=embedding_dim, nhead=num_heads, use_alibi=use_alibi
                        )
                        for _ in range(num_layers)
                    ]
                )

        self.use_context = use_context

        # Create a classification head for each tissue
        self.strand_agg = strand_agg
        if strand_agg == "concat":
            self.tissue_classifiers = nn.ModuleDict(
                {
                    str(tissue_id): nn.Linear(embedding_dim * 2, num_classes)
                    for tissue_id in range(num_tissues)
                }
            )
        else:
            self.tissue_classifiers = nn.ModuleDict(
                {
                    str(tissue_id): nn.Linear(embedding_dim, num_classes)
                    for tissue_id in range(num_tissues)
                }
            )

        # Loss function
        loss_type = loss_fn[0]
        gamma = float(loss_fn[1])
        assert seq_pool in ["mean", "max", "linear"]
        self.seq_pool = seq_pool
        if seq_pool == "linear":
            self.linear = nn.Linear(token_length, 1)

        self.validation_outputs = []  # To store validation outputs
        self.learning_rate = learning_rate
        assert cre_type in ["multi", "binary", "9class"]
        if cre_type == "multi":
            self.cre_type = constants.CREs
        elif cre_type == "9class":
            self.cre_type = constants.NINE_CLASS_CREs
        elif cre_type == "binary":
            self.cre_type = constants.BINARY_CREs
        else:
            raise ValueError("Unknown cre type")
        class_weights_tensor = None

        assert loss_type in ["cross_entropy", "focal", "weighted_cross_entropy"]

        if loss_type == "weighted_cross_entropy":
            if cre_type == "multi":
                class_weights_tensor = torch.tensor(constants.MULTI_CLASS_WEIGHTS)
            elif cre_type == "binary":
                class_weights_tensor = torch.tensor(constants.BINARY_CLASS_WEIGHTS)
            elif cre_type == "9class":
                class_weights_tensor = torch.tensor(constants.NINE_CLASS_WEIGHTS)
            else:
                raise ValueError("Unknown cre type")

        self.criterion = get_loss_fn(
            loss_type, reduction="none", gamma=gamma, class_weight=class_weights_tensor
        )

    def forward(
        self,
        x,
        padding_mask,
        tissue_vector,
        context=None,
        only_embed=False,
        precision=torch.float32,
    ):
        """
        x: Tensor of shape (batch_size, num_strands, token_length)
        padding_mask: Tensor of shape (batch_size, num_strands, token_length)
        tissue_vector: Tensor of shape (batch_size,)
        context: Tensor of shape (batch_size,)
        """
        batch_size, num_strands, token_length = x.size()

        # Reshape for embedding
        x = x.view(batch_size * num_strands, token_length)
        padding_mask = padding_mask.view(batch_size * num_strands, token_length)

        # Token embedding
        x = self.token_embedding(
            x
        )  # Shape: (batch_size*num_strands, token_length, embedding_dim)

        if self.pos_encoding_type == "sinusoidal":
            x = x + self.position_encoding.to(x.device)

        if self.use_context:
            context = self.context_embedding(context)
            context = context.unsqueeze(1).repeat(
                1, num_strands, 1
            )  # [batch_size, num_strands, embed_dim]
            if self.expand_context_type:
                context = context.unsqueeze(
                    2
                )  # [batch_size, num_strands, 1, embed_dim]
                context = self.expand_context(context.permute(0, 1, 3, 2)).permute(
                    0, 1, 3, 2
                )  # [batch_size, num_strands, token_length, embed_dim]
                context = context.view(
                    batch_size * num_strands, token_length, -1
                )  # [batch_size*num_strands, token_length, embed_dim]
            else:
                context = context.unsqueeze(2).repeat(
                    1, 1, token_length, 1
                )  # [batch_size, num_strands, token_length, embed_dim]
                context = context.view(
                    batch_size * num_strands, token_length, -1
                )  # [batch_size*num_strands, token_length, embed_dim]
            for layer in self.transformer_encoder:
                x = layer(
                    x,
                    context=context,
                    key_padding_mask=padding_mask,
                    precision=precision,
                )
        else:
            for layer in self.transformer_encoder:
                x = layer(x, src_key_padding_mask=padding_mask, precision=precision)

        # Use masking to compute the mean over valid tokens

        if self.seq_pool == "max":
            pad_mask = padding_mask.float()
            mask = pad_mask.masked_fill(pad_mask == 1, float("-inf"))
            sub_x = x + mask.unsqueeze(2)
            x = torch.max(sub_x, dim=1)[0]  # [batch_size*num_strands, embed_dim]

        elif self.seq_pool == "mean":
            lengths = (~padding_mask).sum(dim=1).unsqueeze(1)  # [batch_size, 1]
            x = (x * (~padding_mask).unsqueeze(2)).sum(
                dim=1
            ) / lengths  # [batch_size*num_strands, embed_dim]

        elif self.seq_pool == "linear":
            x = x * (~padding_mask).unsqueeze(2)
            x = x.permute(0, 2, 1)  # [batch_size*num_strands, embed_dim, token_length]
            x = self.linear(x).squeeze(2)  # [batch_size*num_strands, embed_dim]

        # Reshape back and aggregate over strands
        x = x.view(batch_size, num_strands, -1)

        x_embed = x.clone()
        if only_embed:
            return x_embed
        if self.strand_agg == "concat":
            x = x.view(batch_size, -1)
        else:
            x = x.mean(dim=1)  # Aggregate over strands

        # Prepare to collect logits from different tissue classifiers
        logits = torch.zeros(
            batch_size, self.hparams.num_classes, device=x.device, dtype=x.dtype
        )

        # Process each tissue separately
        for tissue_id in torch.unique(tissue_vector):
            tissue_id = tissue_id.item()
            tissue_indices = (tissue_vector == tissue_id).nonzero(as_tuple=True)[0]
            if tissue_indices.numel() == 0:
                continue  # Skip if no samples for this tissue

            x_tissue = x[tissue_indices]
            classifier = self.tissue_classifiers[str(tissue_id)]
            logits_tissue = classifier(x_tissue)
            logits[tissue_indices] = logits_tissue

        return logits, x_embed

    def training_step(self, batch, batch_idx):
        """Training step"""
        x, padding_mask, tissue_vector, y, context, majority_y, _ = batch
        if self.use_context:
            logits, embed = self.forward(x, padding_mask, tissue_vector, context)
        else:
            logits, embed = self.forward(x, padding_mask, tissue_vector)
        weights = torch.ones_like(y).float()
        if self.majority_weight:
            weights[y == majority_y] = self.majority_weight
        loss = self.criterion(logits, y)
        if self.reduction == "sum":
            loss = (loss * weights).sum()
        else:
            loss = (loss * weights).mean()
        if self.use_dual_loss:
            embed = embed.view(embed.size(0), -1)
            embed = torch.nn.functional.normalize(embed, p=2, dim=1)
            if self.use_temp:
                adj = (
                    torch.matmul(embed, embed.transpose(0, 1)) * self.logit_scale.exp()
                )
                tau = 1 / self.logit_scale.exp()
                self.log("temperature", tau, on_epoch=True, sync_dist=True)

            else:
                adj = torch.matmul(embed, embed.transpose(0, 1))
            logit_loss1 = torch.nn.functional.cross_entropy(
                adj, torch.arange(adj.size(0)).to(adj.device), reduction=self.reduction
            )
            logit_loss2 = torch.nn.functional.cross_entropy(
                adj.transpose(0, 1),
                torch.arange(adj.size(0)).to(adj.device),
                reduction=self.reduction,
            )
            logit_loss = (logit_loss1 + logit_loss2) / 2
            if self.fraction is not None:
                fraction = self.fraction
            else:
                fraction = (
                    loss.clone().detach().item() / logit_loss.clone().detach().item()
                )
            # Clip fraction to a sensible range, e.g. [0.1, 10]
            fraction = max(0.01, min(100.0, fraction))
            loss = loss + fraction * logit_loss
            self.log("fraction", fraction, on_epoch=True, sync_dist=True)
            self.log("logit_loss", logit_loss, on_epoch=True, sync_dist=True)

        self.log("train_loss", loss, on_epoch=True, sync_dist=True)
        return loss

    def logf1(self, preds, y, cre_mapping, prefix):
        for class_idx in range(self.num_classes):
            class_targets = y == class_idx
            if class_targets.sum() == 0:
                continue  # Skip if no samples for this class

            class_preds = preds == class_idx

            tp = (class_preds & class_targets).sum()
            fp = (class_targets[class_preds.bool()] == 0).sum()
            fn = (class_targets[~class_preds.bool()] == 1).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            self.log(
                f"{prefix}_val_f1_class_{cre_mapping[class_idx]}".replace(",", ""),
                f1_score,
                prog_bar=True,
                sync_dist=True,
                on_epoch=True,
            )
        # Store outputs for plotting
        acc = (preds == y).float().mean()
        self.log(f"{prefix}_val_acc", acc, prog_bar=True, sync_dist=True, on_epoch=True)

    def process_sub_batch(self, x, padding_mask, tissue_vector, context, y):
        chunk_size = 1000
        logits_out = []
        for i in range(0, x.size(0), chunk_size):
            x_chunk = x[i : i + chunk_size]
            padding_mask_chunk = padding_mask[i : i + chunk_size]
            tissue_vector_chunk = tissue_vector[i : i + chunk_size]
            context_chunk = context[i : i + chunk_size]
            y_chunk = y[i : i + chunk_size]

            if self.use_context:
                logits, _ = self.forward(
                    x_chunk, padding_mask_chunk, tissue_vector_chunk, context_chunk
                )
            else:
                logits, _ = self.forward(
                    x_chunk, padding_mask_chunk, tissue_vector_chunk
                )

            logits_out.append(logits)
        return torch.cat(logits_out, dim=0).to(x.device)

    def validation_step(self, batch, batch_idx):
        """Validation step"""

        x, padding_mask, tissue_vector, y, context, _, _ = batch

        logits = self.process_sub_batch(x, padding_mask, tissue_vector, context, y)

        loss = self.criterion(logits, y)
        loss = loss.sum()
        self.log("val_loss", loss, prog_bar=True, sync_dist=True, on_epoch=True)

        # Calculate F1 scores for each class
        preds = torch.argmax(logits, dim=1)
        cre_mapping = self.cre_type

        self.logf1(preds, y, cre_mapping, "")

        # Calculate F1 scores for each tissue
        for tissue_id in torch.unique(tissue_vector):
            tissue_id = tissue_id.item()
            tissue_indices = (tissue_vector == tissue_id).nonzero(as_tuple=True)[0]
            if tissue_indices.numel() == 0:
                continue  # Skip if no samples for this tissue

            preds_tissue = preds[tissue_indices]
            y_tissue = y[tissue_indices]
            self.logf1(
                preds_tissue, y_tissue, cre_mapping, f"{self.tissues[tissue_id]}_"
            )

        return loss

    def predict_step(self, batch, batch_idx):
        """Prediction step"""
        x, padding_mask, tissue_vector, y, context, _ = batch

        if self.use_context:
            logits, _ = self.forward(x, padding_mask, tissue_vector, context)
        else:
            logits, _ = self.forward(x, padding_mask, tissue_vector)

        return logits

    def configure_optimizers(self):
        print("Learning rate: ", self.hparams.learning_rate)
        print("Weight decay: ", self.hparams.weight_decay)

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn  # full param name
                if (
                    fpn.endswith("expand_context.weight")
                    or pn.endswith("bias")
                    or isinstance(m, blacklist_weight_modules)
                ):
                    no_decay.add(fpn)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn
                if fpn not in no_decay:
                    decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, (
            "parameters %s made it into both decay/no_decay sets!"
            % (str(inter_params),)
        )
        assert len(param_dict.keys() - union_params) == 0, (
            "parameters %s were not separated into either decay/no_decay set!"
            % (str(param_dict.keys() - union_params),)
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.Adam(optim_groups, lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=2,
            factor=self.lr_scale,
            min_lr=1e-7,
            verbose=True,
        )

        scheduler_config = {
            "scheduler": scheduler,
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [scheduler_config]
