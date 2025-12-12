"This script processes the data for the seq2gene model"

import os
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from utils.constants import CREs
from utils.functions import (
    merge_across_dfs,
    split_dataframe_across_chromosomes,
    load_bed_regions_as_df,
)
import random


# Define functions
class GeneProcess:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.shard_data_path = f"{config['output']}/data_split"
        self.gene_shard_data_path = f"{config['output']}/gene_shard_data"
        self.train_path = f"{config['output']}/{config['train_dir']}"
        self.test_path = f"{config['output']}/{config['test_dir']}"
        self.reg_test_dir = f"{config['output']}/{config['reg_test_dir']}"
        self.reg_train_dir = f"{config['output']}/{config['reg_train_dir']}"

    def save_chrs(
        self,
        dfs: List[pd.DataFrame],
        chrs: List[str],
        subjects: List[str],
        tissues: List[str],
    ) -> None:
        "Save the dataframes"
        Path(f"{self.shard_data_path}").mkdir(parents=True, exist_ok=True)
        for i, Chr in enumerate(chrs):
            df = dfs[i].copy()
            for subject in subjects:
                if os.path.exists(f"{self.shard_data_path}/{subject}_{Chr}.pkl.gz"):
                    raise FileExistsError(
                        f"File {self.shard_data_path}/{subject}_{Chr}.pkl.gz already exists"
                    )

                relevant_colums = (
                    [
                        "chrom",
                        "start",
                        "end",
                        "cCRE",
                        f"{subject}_sequence",
                        f"{subject}_encoded_seq",
                    ]
                    + [f"{subject}_{tissue}_cCRE" for tissue in tissues]
                    + [f"{tissue}_majority" for tissue in tissues]
                )
                sub_df = df[relevant_colums].copy()
                print(f"\nSaving {subject} {Chr}\n")
                sub_df.to_pickle(f"{self.shard_data_path}/{subject}_{Chr}.pkl.gz")

    def save_seperate_chrs(
        self,
        dfs: List[pd.DataFrame],
        chrs: List[str],
        subjects: List[str],
        tissues: List[str],
    ) -> None:
        "Save the dataframes"
        Path(f"{self.shard_data_path}").mkdir(parents=True, exist_ok=True)
        for i, Chr in enumerate(chrs):
            df = dfs[i].copy()
            for subject in subjects:
                if os.path.exists(f"{self.shard_data_path}/{subject}_{Chr}.pkl.gz"):
                    print(
                        f"File {self.shard_data_path}/{subject}_{Chr}.pkl.gz already exists"
                    )
                    continue
                relevant_colums = [
                    "chrom",
                    "start",
                    "end",
                    "cCRE",
                    f"{subject}_sequence",
                    f"{subject}_encoded_seq",
                ]
                sub_df = df[relevant_colums].copy()
                print(f"\nSaving {subject} {Chr}\n")
                sub_df.to_pickle(f"{self.shard_data_path}/{subject}_{Chr}.pkl.gz")

    def shard_data(self):
        "Shard the data into chromosomes"
        config = self.config
        df = pd.read_csv(config["input"])
        subject_dfs: List[pd.DataFrame] = []
        check_heterogeneity: List[str] = []
        relevant_colums: List[str] = ["chrom", "start", "end", "cCRE"]
        for i in range(len(df)):
            subject = df.iloc[i]["Donor"]  # get the subject
            tissues = list(df.columns)[2:]  # get the tissues
            ref_cre = pd.read_pickle(
                df.iloc[i]["Encoded_Seq"]
            )  # load the reference CRE

            ref_cre = ref_cre[
                "chrom start end cCRE sequence encoded_representation".split()
            ]
            ref_cre = ref_cre.rename(
                columns={
                    "sequence": f"{subject}_sequence",
                    "encoded_representation": f"{subject}_encoded_seq",
                }
            )
            relevant_colums.append(f"{subject}_sequence")
            relevant_colums.append(f"{subject}_encoded_seq")

            dfs: List[pd.DataFrame] = [ref_cre]
            for tissue in tissues:
                if len(df.iloc[i][tissue]) == 0 or df.iloc[i][tissue] == None:
                    continue
                df_tissue = load_bed_regions_as_df(df.iloc[i][tissue])
                df_tissue = df_tissue.rename(
                    columns={
                        0: "chrom",
                        1: "start",
                        2: "end",
                        3: "name",
                        9: f"{subject}_{tissue}_cCRE",
                        10: f"{subject}_{tissue}_data_prop",
                    }
                )
                df_tissue["start"] = df_tissue["start"] - config["neighbour_hood"]
                df_tissue["end"] = df_tissue["end"] + config["neighbour_hood"]
                relevant_colums.append(f"{subject}_{tissue}_cCRE")
                relevant_colums.append(f"{subject}_{tissue}_data_prop")
                check_heterogeneity.append(f"{subject}_{tissue}_cCRE")

                df_tissue = df_tissue[
                    [
                        "chrom",
                        "start",
                        "end",
                        f"{subject}_{tissue}_cCRE",
                        f"{subject}_{tissue}_data_prop",
                    ]
                ]
                dfs.append(df_tissue)

            merged_df = merge_across_dfs(dfs)
            merged_df = merged_df.reset_index(drop=True)
            # merged_df = Sync_CRES(merged_df, tissues, [subject], check_heterogeneity)
            subject_dfs.append(merged_df)

        all_subs = merge_across_dfs(subject_dfs, on=["chrom", "start", "end", "cCRE"])
        all_subs = all_subs.reset_index(drop=True)
        all_subs = all_subs[relevant_colums]
        print(all_subs.head())
        print(f"length of all_subs: {len(all_subs)}")

        majority_class = [CREs[0]] * len(all_subs)
        for t in tissues:
            all_subs[f"{t}_majority"] = majority_class

        all_subs, all_sub_chrs = split_dataframe_across_chromosomes(all_subs)
        self.save_chrs(all_subs, all_sub_chrs, list(df["Donor"]), tissues)

    def split_to_chrs(self):
        "Shard the data into chromosomes"
        config = self.config
        df = pd.read_csv(config["input"])
        subject_dfs: List[pd.DataFrame] = []
        check_heterogeneity: List[str] = []
        relevant_colums: List[str] = ["chrom", "start", "end", "cCRE"]
        for i in range(len(df)):
            subject = df.iloc[i]["Donor"]  # get the subject
            tissues = list(df.columns)[2:]  # get the tissues
            ref_cre = pd.read_pickle(
                df.iloc[i]["Encoded_Seq"]
            )  # load the reference CRE

            ref_cre = ref_cre[
                "chrom start end cCRE sequence encoded_representation".split()
            ]
            ref_cre = ref_cre.rename(
                columns={
                    "sequence": f"{subject}_sequence",
                    "encoded_representation": f"{subject}_encoded_seq",
                }
            )
            relevant_colums.append(f"{subject}_sequence")
            relevant_colums.append(f"{subject}_encoded_seq")
            merged_df = ref_cre.copy()
            merged_df = merged_df.reset_index(drop=True)
            # merged_df = Sync_CRES(merged_df, tissues, [subject], check_heterogeneity)
            subject_dfs.append(merged_df)

        all_subs = merge_across_dfs(subject_dfs, on=["chrom", "start", "end", "cCRE"])
        all_subs = all_subs.reset_index(drop=True)
        all_subs = all_subs[relevant_colums]
        print(all_subs.head())
        print(f"length of all_subs: {len(all_subs)}")

        all_subs, all_sub_chrs = split_dataframe_across_chromosomes(all_subs)
        self.save_seperate_chrs(all_subs, all_sub_chrs, list(df["Donor"]), tissues)

    def process_chunks(self, args):
        "Process the chunks"
        df, subject, tissue, i = args
        df = df.reset_index(drop=True)
        if df.iloc[0]["chrom"] == self.config["test_chr"]:
            store_path = self.reg_test_dir
        else:
            store_path = self.reg_train_dir

        if len(df) < self.config["chunk_size"]:
            indices = list(df.index)
            new_indices = indices + random.choices(
                indices, k=self.config["chunk_size"] - len(indices)
            )
            df = df.loc[new_indices]
        df = df.sort_values(by=["start"]).reset_index(drop=True)
        df["tissue"] = [tissue] * len(df)
        df.to_pickle(
            f"{store_path}/{subject}_{tissue}_{df.iloc[0]['chrom']}_chunk_{i}.pkl.gz"
        )

    def reg_train_test_split(self) -> None:
        config = self.config
        chunk_size = config["chunk_size"]
        pruned_cres = pd.read_csv(
            config["pruned_cres"], sep="\t"
        )  # 'chromosome', 'start_cre', 'end_cre', 'cre_id', 'cre_name', 'strand_cre', 'cre_color']
        pruned_cres["start_cre"] = pruned_cres["start_cre"] - config["neighbour_hood"]
        pruned_cres["end_cre"] = pruned_cres["end_cre"] + config["neighbour_hood"]

        # pick n_test files for testing
        Path(self.reg_train_dir).mkdir(parents=True, exist_ok=True)
        Path(self.reg_test_dir).mkdir(parents=True, exist_ok=True)

        # Combine files for each chromosome and tissue
        tissues = list(pd.read_csv(config["input"]).columns)[2:]
        for subject in list(pd.read_csv(config["input"])["Donor"]):
            for chromosome in [f"chr{i}" for i in range(1, 23)]:
                file_path = f"{self.shard_data_path}/{subject}_{chromosome}.pkl"
                if not os.path.exists(file_path):
                    file_path += ".gz"
                if os.path.exists(file_path):
                    df = pd.read_pickle(
                        file_path
                    )  # 'chrom', 'start', 'end', 'cCRE', f'{subject}_sequence', f'{subject}_encoded_seq' f'{subject}_{tissue}_cCRE'  f'{tissue}_majority'
                else:
                    raise FileNotFoundError(f"File {file_path} not found")
                pruned_df = df.merge(
                    pruned_cres,
                    left_on=["chrom", "start", "end"],
                    right_on=["chromosome", "start_cre", "end_cre"],
                ).reset_index(drop=True)
                for tissue in tissues:
                    sub_df = pruned_df[
                        [
                            "chrom",
                            "start",
                            "end",
                            "cCRE",
                            f"{subject}_sequence",
                            f"{subject}_encoded_seq",
                            f"{subject}_{tissue}_cCRE",
                            f"{tissue}_majority",
                        ]
                    ]
                    sub_df = sub_df.rename(
                        columns={
                            f"{subject}_{tissue}_cCRE": "tissue_CRE",
                            f"{tissue}_majority": "tissue_majority_CRE",
                            f"{subject}_sequence": "sequence",
                            f"{subject}_encoded_seq": "encoded_seq",
                        }
                    )
                    with Pool(config["ncpus"]) as p:
                        p.map(
                            self.process_chunks,
                            [
                                (sub_df.iloc[i : i + chunk_size], subject, tissue, i)
                                for i in range(0, len(sub_df), chunk_size)
                            ],
                        )

    def process_genes(self, gene: str) -> None:
        "Process the genes"
        config = self.config
        input_df = pd.read_csv(config["data_prop_gene"])
        tissues = list(input_df.columns)[2:]
        subjects = list(input_df["Donors"])
        gene_df = pd.read_csv(f"{self.config['gene_vocab_path']}/{gene}/gene_vocab.csv")
        # columns: chromosome	start	end	gene_id	gene_name	strand	start_cre	end_cre	cre_id	score	strand_cre	att1	att2	cre_color	cre_name	embedding	start_gene	end_gene
        gene_df["start_cre"] = gene_df["start_cre"] - self.config["neighbour_hood"]
        gene_df["end_cre"] = gene_df["end_cre"] + self.config["neighbour_hood"]
        for i, subject in enumerate(subjects):
            try:
                file_path = f"{self.gene_shard_data_path}/{gene}_{subject}.pkl"
                if not os.path.exists(file_path):
                    file_path += ".gz"
                df = pd.read_pickle(file_path)
                print(f"Skipping {gene}_{subject} because it already exists")
                continue
            except:
                print(f"Processing {gene}_{subject}")
            X = []
            gene_cres = []
            file_path = (
                f"{self.shard_data_path}/{subject}_{gene_df.iloc[0]['chromosome']}.pkl"
            )
            if not os.path.exists(file_path):
                file_path += ".gz"
            if os.path.exists(file_path):
                df = pd.read_pickle(file_path)
            else:
                raise FileNotFoundError(f"File {file_path} not found")
            df = df.rename(columns={"start": "start_cre", "end": "end_cre"})
            gene_cres_df = pd.merge(
                gene_df,
                df,
                left_on=["chromosome", "start_cre", "end_cre"],
                right_on=["chrom", "start_cre", "end_cre"],
            )
            if gene_cres_df.iloc[0]["strand"] == "-":
                gene_cres_df = gene_cres_df.sort_values(by=["end_cre"], ascending=False)
            else:
                gene_cres_df = gene_cres_df.sort_values(
                    by=["start_cre"], ascending=True
                )
            gene_cres_df = gene_cres_df.reset_index(drop=True)

            D = {}
            for item in gene_cres_df.columns:
                if item.endswith("_sequence"):
                    D[item] = "sequence"
                if item.endswith("_encoded_seq"):
                    D[item] = "encoded_seq"
                if item.endswith("cre_name"):
                    D[item] = "cCRE"
            gene_cres_df = gene_cres_df.rename(columns=D)
            # cres_seq = [len(enc_seq.split(',')[0]) for enc_seq in gene_cres_df['encoded_seq'].values]
            encoded_seq_f = [
                len(gene_cres_df["encoded_seq"].iloc[it][0])
                for it in range(len(gene_cres_df["encoded_seq"]))
            ]
            encoded_seq_r = [
                len(gene_cres_df["encoded_seq"].iloc[it][1])
                for it in range(len(gene_cres_df["encoded_seq"]))
            ]
            if min(encoded_seq_f) == 0 or min(encoded_seq_r) == 0:
                print(f"Skipping {gene}_{subject} due to empty CRE sequence")
                return
            gene_cres_df = gene_cres_df[
                [
                    "chromosome",
                    "start",
                    "end",
                    "gene_id",
                    "gene_name",
                    "strand",
                    "start_cre",
                    "end_cre",
                    "cre_id",
                    "score",
                    "strand_cre",
                    "att1",
                    "att2",
                    "cre_color",
                    "cCRE",
                    "start_gene",
                    "end_gene",
                    "chrom",
                    "cCRE",
                    "sequence",
                    "encoded_seq",
                ]
            ].copy()

            """
                for tissue in tissues:
                    gene_ex_file = input_df[input_df['Donors'] == subject][tissue].values[0]
                    gene_ex = pd.read_csv(gene_ex_file, sep='\t')
                    exp_val = gene_ex[gene_ex['gene_id'] == gene][['TPM', 'FPKM']]
                    if len(exp_val) > 0:
                        gene_cres_df[f'{subject}_{tissue}_exp_TPM'] = [exp_val['TPM'].values[0]]*len(gene_cres_df)
                        gene_cres_df[f'{subject}_{tissue}_exp_FPKM'] = [exp_val['FPKM'].values[0]]*len(gene_cres_df)
                    else:
                        gene_cres_df[f'{subject}_{tissue}_exp_TPM'] = ['']*len(gene_cres_df)
                        gene_cres_df[f'{subject}_{tissue}_exp_FPKM'] = ['']*len(gene_cres_df)
                """

            gene_cres_df.to_pickle(
                f"{self.gene_shard_data_path}/{gene}_{subject}.pkl.gz"
            )

    def shard_genes(self):
        "Shard the genes"
        config = self.config
        gene_vocab_path = config["gene_vocab_path"]
        genes = os.listdir(gene_vocab_path)
        Path(f"{self.gene_shard_data_path}").mkdir(parents=True, exist_ok=True)
        with Pool(config["ncpus"]) as p:
            p.map(self.process_genes, genes)

    def process_train_test_split(self, gene_file: str) -> None:
        "Process the train test split"
        if not os.path.exists(f"{self.gene_shard_data_path}/{gene_file}"):
            print(f"Skipping {gene_file} because it does not exist")
            print(f"{self.gene_shard_data_path}/{gene_file}")
            return
        config = self.config
        gene_vocab_path = config["gene_vocab_path"]
        gene = gene_file.split("_")[0]
        file_name_without_gene = "_".join(
            gene_file.split("_")[1:]
        )  # remove the gene name from the file name
        subject = file_name_without_gene.split(".")[
            0
        ]  # get the subject name from the file name and remove the extension .pkl.gz
        gene_df = pd.read_csv(f"{gene_vocab_path}/{gene}/gene_vocab.csv")
        tissues = list(pd.read_csv(config["data_prop_gene"]).columns)[2:]
        if config["autosomes_only"]:
            if gene_df.iloc[0]["chromosome"] not in [f"chr{i}" for i in range(1, 23)]:
                print(f"Skipping {gene_df.iloc[0]['chromosome']}")
                return
        test_chr = config["test_chr"]
        if gene_df.iloc[0]["chromosome"] == test_chr:
            store_path = self.test_path

        else:
            store_path = self.train_path
        try:
            df = pd.read_pickle(f"{self.gene_shard_data_path}/{gene_file}")
        except:
            print(f"Corrupted: {self.gene_shard_data_path}/{gene_file}")

        for tissue in tissues:
            df_new = df[
                f"chrom start_cre end_cre start_gene end_gene strand cCRE embedding sequence encoded_seq \
                    {subject}_{tissue}_exp_TPM {subject}_{tissue}_exp_FPKM {subject}_{tissue}_cCRE".split()
            ]
            D = {}
            for item in df.columns:
                if item.endswith("_cCRE"):
                    D[item] = "tissue_CRE"
                if item.endswith("_exp_TPM"):
                    D[item] = "TPM"
                if item.endswith("_exp_FPKM"):
                    D[item] = "FPKM"

            df_new = df_new.rename(columns=D)
            df_new = df_new[
                [
                    "TPM",
                    "FPKM",
                    "tissue_CRE",
                    "encoded_seq",
                    "cCRE",
                    "embedding",
                    "strand",
                ]
            ]

            if (
                df_new["TPM"].isnull().values.any()
                or df_new["FPKM"].isnull().values.any()
                or df_new["TPM"].isna().values.any()
                or df_new["FPKM"].isna().values.any()
                or df_new["TPM"].values[0] == ""
                or df_new["FPKM"].values[0] == ""
            ):
                print("=" * 50)
                print(f"Skipping {gene}_{subject}_{tissue} due to missing TPM or FPKM")
                return
            if config["log1p"]:
                df_new["log1p_TPM"] = np.log1p(df_new["TPM"])
                df_new["log1p_FPKM"] = np.log1p(df_new["FPKM"])
            df_new["tissue"] = [tissue] * len(df_new)
            df_new.to_pickle(
                f"{store_path}/{gene}_{subject}_{tissue}.pkl.gz"
            )  # df do not contain sequence and encoded_seq which needs to be pulled from sharded gene data.
        return

    def train_test_split(self):
        "Split the data into train and test"
        config = self.config
        gene_vocab_path = config["gene_vocab_path"]
        genes = os.listdir(gene_vocab_path)
        gene_files = []
        input_df = pd.read_csv(config["data_prop_gene"])
        for gene in genes:
            for subject in list(input_df["Donors"]):
                gene_files.append(f"{gene}_{subject}.pkl.gz")
        Path(f"{self.test_path}").mkdir(parents=True, exist_ok=True)
        Path(f"{self.train_path}").mkdir(parents=True, exist_ok=True)
        with Pool(config["ncpus"]) as p:
            p.map(self.process_train_test_split, gene_files)
