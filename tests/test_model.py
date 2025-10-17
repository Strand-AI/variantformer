import unittest
from pathlib import Path
import pandas as pd
import numpy as np
import subprocess
import tempfile

from processors.vcfprocessor import VCFProcessor
from processors.variantprocessor import VariantProcessor
from processors import ad_risk

# Get the repo root directory (equivalent to /app/ in deployment)
_REPO_ROOT = Path(__file__).parent.parent.resolve()
VCF_EXAMPLE = _REPO_ROOT / "_artifacts" / "HG00096.vcf.gz"
TISSUE_MAP_GUID = "be73e19a"


class TestVariantProcessorAnVcfProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # bcftools expects an index so create one
        index_file = VCF_EXAMPLE.with_suffix(VCF_EXAMPLE.suffix + ".tbi")
        if not index_file.exists():
            subprocess.run(["bcftools", "index", "-t", str(VCF_EXAMPLE)], check=True)

    def setUp(self) -> None:
        """
        Set up the test case with a sample VariantProcessor and VCFProcessor instance and test data.
        """
        self.vcf_df = pd.read_parquet(_REPO_ROOT / "_artifacts" / "f9bbc0ba.pq")
        model_class = "D2C_AG"
        self.variant_processor = VariantProcessor(model_class=model_class)
        self.vcf_processor = VCFProcessor(model_class=model_class)

    def test_1(self) -> None:
        """
        Test the VariantProcessor and VCFProcessor prediction pipeline with sample data.
        """
        vcf_path = str(VCF_EXAMPLE)
        variant_df = {
            "chrom": ["chr19"],
            "pos": [44908684],
            "ref": ["T"],
            "alt": ["T"],
            "tissue": ["whole blood"],
            "gene_id": ["ENSG00000130203.9"],
        }
        variant_df = pd.DataFrame(variant_df)
        variant_df["tissues"] = variant_df["tissue"]
        vcf_dataset, dataloader = self.vcf_processor.create_data(vcf_path, variant_df)
        model, checkpoint_path, trainer = self.vcf_processor.load_model()
        predictions_df = self.vcf_processor.predict(
            model, checkpoint_path, trainer, dataloader, vcf_dataset
        )
        print("VCF-based predictions:")
        print(predictions_df.head(2))

        with tempfile.TemporaryDirectory() as temp_dir:
            variant_vcf_df = self.variant_processor.predict(
                variant_df,
                output_dir=temp_dir,
                vcf_path=vcf_path,
                sample_name=self.vcf_df["name"].iloc[0],
            )

        variant_based_df = variant_vcf_df[
            (variant_vcf_df["sample_name"] == self.vcf_df["name"].iloc[0])
            & (variant_vcf_df["zygosity"] == "0")
        ].reset_index(drop=True)
        print("Variant-based predictions:")
        print(variant_based_df.head(2))

        self.assertTrue(
            np.allclose(
                variant_based_df["gene_exp"].values[0],
                predictions_df["predicted_expression"].iloc[0][0],
                atol=0.1,
            ),
            f"Variant-based predictions: {variant_based_df['gene_exp']} do not match VCF-based predictions: {predictions_df['predicted_expression'].iloc[0]}",
        )

        self.assertTrue(
            np.allclose(
                variant_based_df["gene_emb"].values[0],
                predictions_df["embeddings"].iloc[0][0],
                atol=1,
            ),
            f"Variant-based predictions: {variant_based_df['gene_emb']} do not match VCF-based predictions: {predictions_df['embeddings'].iloc[0]}",
        )


class TestGeneExpressionAndEmbedding(unittest.TestCase):
    def setUp(self) -> None:
        self.vcf_processor = VCFProcessor(model_class="D2C_PCG")
        simple_query = {
            "gene_id": ["ENSG00000000457.13"] * 2,
            "tissues": ["whole blood,thyroid,artery - aorta", "brain - amygdala"],
        }
        self.query_df = pd.DataFrame(simple_query)
        self.vcf_path = str(VCF_EXAMPLE)
        self.vcf_dataset, self.dataloader = self.vcf_processor.create_data(
            self.vcf_path, self.query_df
        )
        self.model, self.checkpoint_path, self.trainer = self.vcf_processor.load_model()
        self.target_df = pd.read_parquet(
            _REPO_ROOT / "_artifacts" / "924979a7.pq"
        )  # <- Needs to be changed to new version

    def test_1(self) -> None:
        preds_df = self.vcf_processor.predict(
            self.model,
            self.checkpoint_path,
            self.trainer,
            self.dataloader,
            self.vcf_dataset,
        )
        preds_df.predicted_expression = preds_df.predicted_expression.apply(
            lambda x: x.flatten()
        )
        preds_df.embeddings = preds_df.embeddings.apply(lambda x: x.flatten())
        preds_df.tissues = preds_df.tissues.apply(lambda x: np.array(x))
        preds_df.tissue_names = preds_df.tissue_names.apply(lambda x: np.array(x))
        # pd.testing.assert_frame_equal(preds_df, self.target_df) <- Can be uncommented when target df is updated
        print("Gene expression and embedding predictions:")
        print(preds_df.head(2))


class TestADrisk(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up the test case with a sample ADrisk instance and test data.
        """
        self.model_class = "v4_pcg"
        self.gene_id = "ENSG00000130203.9"  # APOE gene
        self.tissue_id = 7
        self.adrisk = ad_risk.ADrisk(
            self.gene_id, self.tissue_id, model_class=self.model_class
        )
        self.vcf_path = str(VCF_EXAMPLE)
        self._init_vcf_processor()
        self._init_dataloader()
        self._init_vcf_processor()

    def _init_vcf_processor(self):
        model_class = "D2C_" + self.model_class.split("_")[-1].upper()
        self.vcf_processor = VCFProcessor(model_class=model_class)
        tissues_dict = self.vcf_processor.tissue_vocab
        self.tissue_map = pd.DataFrame(
            {"tissue": list(tissues_dict.keys())},
            index=pd.Index(list(tissues_dict.values()), name="tissue_id"),
        )
        self.genes_map = self.vcf_processor.get_genes()
        self.genes_map.set_index("gene_id", inplace=True)
        self.model, self.checkpoint_path, self.trainer = self.vcf_processor.load_model()

    def _init_dataloader(self):
        self.tissue_map = pd.read_parquet(
            (_REPO_ROOT / "_artifacts" / TISSUE_MAP_GUID).with_suffix(".pq")
        )
        tissue_name = self.tissue_map.loc[self.tissue_id, "tissue"]
        self.query_df = pd.DataFrame(
            {"gene_id": [self.gene_id], "tissues": [tissue_name]}
        )
        self.vcf_dataset, self.dataloader = self.vcf_processor.create_data(
            self.vcf_path, self.query_df
        )

    def test_1(self) -> None:
        """
        Test the ADrisk prediction pipeline with sample data.
        """
        preds_df = self.vcf_processor.predict(
            self.model,
            self.checkpoint_path,
            self.trainer,
            self.dataloader,
            self.vcf_dataset,
        )
        gene_tissue_embeds = preds_df["embeddings"].iloc[0]
        # gene_tissue_embeds = np.vstack([gene_tissue_embeds, gene_tissue_embeds])
        preds = self.adrisk(gene_tissue_embeds)
        self.assertAlmostEqual(preds[0], 0.31081957, places=5)  # <- Needs to be checked
        print(f"AD risk proba prediction is: {preds}")


class TestADriskFromVCF(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up the test case with a sample ADrisk instance and test data.
        """
        self.adrisk = ad_risk.ADriskFromVCF()
        self.vcf_path = str(VCF_EXAMPLE)
        self.gene_ids = ["ENSG00000000457.13"] * 2
        self.tissue_ids = [43, 47]

    def test_1(self) -> None:
        """
        Test the ADrisk prediction pipeline with sample data.
        """
        preds = self.adrisk(self.vcf_path, self.gene_ids, self.tissue_ids)
        print(preds[["gene_name", "tissue_name", "ad_risk"]])


if __name__ == "__main__":
    unittest.main()
