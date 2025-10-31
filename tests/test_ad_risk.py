import os
import unittest
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import stash as st
import shutil
# Add parent directory to path
CURRENT_PATH = Path(__file__).parent
sys.path.insert(0, str(CURRENT_PATH.parent))

from processors.vcfprocessor import VCFProcessor
from processors import ad_risk

_REPO_ROOT = Path(__file__).parent.parent.resolve()
ARTIFACTS_DIR = _REPO_ROOT / '_artifacts'
VCF_EXAMPLE = ARTIFACTS_DIR / "HG00096.vcf.gz"
TISSUE_MAP_PATH = ARTIFACTS_DIR / "be73e19a.pq"


class TestADrisk(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up the test case with a sample ADrisk instance and test data.
        """
        self.model_class = 'v4_pcg'
        self.gene_id = 'ENSG00000130203.9'  # APOE gene
        self.tissue_id = 7
        self.adrisk = ad_risk.ADrisk(self.gene_id, self.tissue_id, model_class=self.model_class)
        self.vcf_path = '/mnt/czi-sci-ai/intrinsic-variation-gene-ex-2/project_gene_regulation/dna2cell_training/v2_pcg_flash2/sample_vcf/HG00096.vcf.gz'
        self._init_vcf_processor()
        self._init_dataloader()
        self._init_vcf_processor()
    
    def _init_vcf_processor(self):
        model_class = 'D2C_' + self.model_class.split('_')[-1].upper()
        self.vcf_processor = VCFProcessor(model_class=model_class)
        tissues_dict = self.vcf_processor.tissue_vocab
        self.tissue_map = pd.DataFrame({'tissue': list(tissues_dict.keys())}, index=pd.Index(list(tissues_dict.values()), name='tissue_id'))
        self.genes_map = self.vcf_processor.get_genes()
        self.genes_map.set_index('gene_id', inplace=True)
        self.model, self.checkpoint_path, self.trainer = self.vcf_processor.load_model()
    
    def _init_dataloader(self):
        self.tissue_map = pd.read_parquet(TISSUE_MAP_PATH)
        tissue_name = self.tissue_map.loc[self.tissue_id, 'tissue']
        self.query_df = pd.DataFrame({'gene_id': [self.gene_id], 'tissues': [tissue_name]})
        self.vcf_dataset, self.dataloader = self.vcf_processor.create_data(self.vcf_path, self.query_df)

    def test_1(self) -> None:
        """
        Test the ADrisk prediction pipeline with sample data.
        """
        preds_df = self.vcf_processor.predict(self.model, self.checkpoint_path, self.trainer, self.dataloader, self.vcf_dataset)
        gene_tissue_embeds = preds_df['embeddings'].iloc[0]
        # gene_tissue_embeds = np.vstack([gene_tissue_embeds, gene_tissue_embeds])
        preds = self.adrisk(gene_tissue_embeds)
        import ipdb; ipdb.set_trace()
        self.assertAlmostEqual(preds[0], 0.31081957, places=5) # <- Needs to be checked
        print(f"AD risk proba prediction is: {preds}")


class TestADriskFromVCF(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up the test case with a sample ADrisk instance and test data.
        """
        self.adrisk = ad_risk.ADriskFromVCF()
        self.vcf_path = 
        self.gene_ids = ['ENSG00000000457.13'] * 2
        self.tissue_ids = [43, 47]

    def test_1(self) -> None:
        """
        Test the ADrisk prediction pipeline with sample data.
        """
        preds = self.adrisk(self.vcf_path, self.gene_ids, self.tissue_ids)
        print(preds[['gene_name', 'tissue_name', 'ad_risk']])