import unittest
import numpy as np
import pandas as pd
from pathlib import Path
from processors.variantprocessor import VariantProcessor
import tempfile
import shutil


_REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
VCF_EXAMPLE = _REPO_ROOT / "_artifacts" / "HG00096.vcf.gz"


class Test(unittest.TestCase):
    def setUp(self) -> None:
        # Load target predictions for regression testing
        self.target_predictions = np.load(_REPO_ROOT / "_artifacts" / "befd2388.npz")
        model_class = "D2C_PCG"
        # model_class = "D2C_AG"
        self.processor = VariantProcessor(model_class=model_class)
        print("Model class: ", model_class)
        # Create test data similar to the notebook
        self.test_data = {
            "chr": ["chr13"],
            "pos": [113978728],
            "ref": ["A"],
            "alt": ["G"],
            "tissue": ["whole blood"],
            "gene_id": ["ENSG00000185989.10"],
        }
        self.test_df = pd.DataFrame(self.test_data)
        self.temp_dir = tempfile.mkdtemp()
        self.vcf_df = pd.read_parquet(_REPO_ROOT / "_artifacts" / "f9bbc0ba.pq")
        self.sample_name = self.vcf_df["name"].iloc[0]

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_1(self) -> None:
        with tempfile.TemporaryDirectory() as raw_test_dir:
            raw_predictions = self.processor.predict(
                var_df=self.test_df, output_dir=raw_test_dir
            )

        with tempfile.TemporaryDirectory() as vcf_test_dir:
            vcf_predictions = self.processor.predict(
                var_df=self.test_df,
                output_dir=vcf_test_dir,
                vcf_path=str(VCF_EXAMPLE),
                sample_name=self.sample_name,
            )

        # Compare predictions
        vcf_exp = vcf_predictions[
            (vcf_predictions["sample_name"] == self.sample_name)
            & (vcf_predictions["zygosity"] == "2")
        ]["gene_exp"]
        raw_exp = raw_predictions[
            (raw_predictions["sample_name"] == self.sample_name)
            & (raw_predictions["zygosity"] == "2")
        ]["gene_exp"]
        print(
            vcf_predictions[
                (vcf_predictions["sample_name"] == self.sample_name)
                & (vcf_predictions["zygosity"] == "2")
            ]
        )
        print(
            raw_predictions[
                (raw_predictions["sample_name"] == self.sample_name)
                & (raw_predictions["zygosity"] == "2")
            ]
        )
        self.assertTrue(
            np.allclose(vcf_exp, raw_exp, atol=1),
            f"VCF predictions: {vcf_exp} do not match raw predictions: {raw_exp}",
        )

        vcf_exp = vcf_predictions[
            (vcf_predictions["sample_name"] == self.sample_name)
            & (vcf_predictions["zygosity"] == "1")
        ]["gene_exp"]
        raw_exp = raw_predictions[
            (raw_predictions["sample_name"] == self.sample_name)
            & (raw_predictions["zygosity"] == "1")
        ]["gene_exp"]
        self.assertTrue(
            np.allclose(vcf_exp, raw_exp, atol=1),
            f"VCF predictions: {vcf_exp} do not match raw predictions: {raw_exp}",
        )

    def test_2(self) -> None:
        print("checkpoint 0")
        # Initialize the processor and get raw predictions for comparison with target
        vep_dataset, dataloader, model, trainer, ckpt_path = self.processor.initialize(
            var_df=self.test_df, output_dir=self.temp_dir
        )

        print("checkpoint 1")
        # Get raw predictions for comparison with target predictions
        raw_predictions = trainer.predict(
            model=model, dataloaders=dataloader, ckpt_path=ckpt_path
        )

        print("checkpoint 2")
        for key in raw_predictions[0].keys():
            cur_pred = np.array(raw_predictions[0][key])
            target_pred = self.target_predictions[key]
            print(f"Checking predictions for key: {key}")
            np.testing.assert_allclose(cur_pred, target_pred)

        print("checkpoint 3")
        # Compile predictions to DataFrame and validate format
        predictions_df = self.processor.compile_predictions(raw_predictions)
        # Clean up resources
        self.processor.cleanup()

        print("checkpoint 4")
        # Basic assertions to verify the DataFrame format
        self.assertIsInstance(predictions_df, pd.DataFrame)
        self.assertGreater(len(predictions_df), 0)

        # Check that required columns are present
        expected_columns = [
            "chrom",
            "pos",
            "ref",
            "alt",
            "genes",
            "tissues",
            "variant_type",
            "population",
            "sample_name",
            "zygosity",
            "gene_exp",
        ]
        for col in expected_columns:
            self.assertIn(col, predictions_df.columns)

        # Verify the test data is reflected in predictions
        self.assertTrue((predictions_df["chrom"] == "chr13").any())
        self.assertTrue((predictions_df["pos"] == 113978728).any())
        self.assertTrue((predictions_df["ref"] == "A").any())
        self.assertTrue((predictions_df["alt"] == "G").any())
        self.assertTrue((predictions_df["tissues"] == "whole blood").any())

        # Verify gene expression predictions are numeric
        self.assertTrue(np.issubdtype(predictions_df["gene_exp"].dtype, np.number))

        print("✅ Regression test passed! Raw predictions match target predictions.")
        print(f"✅ Format test passed! Generated {len(predictions_df)} prediction rows")
        print(f"Predictions shape: {predictions_df.shape}")
        print(f"Sample predictions:\n{predictions_df.head()}")


if __name__ == "__main__":
    unittest.main()
