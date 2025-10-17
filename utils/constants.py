BLACKLIST_ALLELES = {".", "*", "N", "n", "-"}
IUPAC_CODES = {
    "A": ["A"],  # Adenine
    "C": ["C"],  # Cytosine
    "G": ["G"],  # Guanine
    "T": ["T"],  # Thymine (or U for Uracil in RNA)
    "R": ["A", "G"],  # Purine (A or G)
    "Y": ["C", "T"],  # Pyrimidine (C or T)
    "S": ["G", "C"],  # Strong interaction (G or C)
    "W": ["A", "T"],  # Weak interaction (A or T)
    "K": ["G", "T"],  # Keto (G or T)
    "M": ["A", "C"],  # Amino (A or C)
    "B": ["C", "G", "T"],  # Not A (C, G, or T)
    "D": ["A", "G", "T"],  # Not C (A, G, or T)
    "H": ["A", "C", "T"],  # Not G (A, C, or T)
    "V": ["A", "C", "G"],  # Not T (A, C, or G)
}
IGNORE_CHRS = ["chrX", "chrY", "chrM"]

REF_CREs = [
    "CTCF-only,CTCF-bound",
    "DNase-H3K4me3",
    "DNase-H3K4me3,CTCF-bound",
    "PLS",
    "PLS,CTCF-bound",
    "dELS",
    "dELS,CTCF-bound",
    "pELS",
    "pELS,CTCF-bound",
]
MAP_REF_CRE_TO_IDX = {cre: idx for idx, cre in enumerate(REF_CREs)}

# Binary CREs
BINARY_CREs = ["Low-DNase", "Non-Low-DNase"]
MAP_BINARY_CRE_TO_IDX = {
    "Low-DNase": 0,
    "DNase-only": 1,
    "CTCF-only,CTCF-bound": 1,
    "DNase-H3K4me3": 1,
    "DNase-H3K4me3,CTCF-bound": 1,
    "PLS": 1,
    "PLS,CTCF-bound": 1,
    "dELS": 1,
    "dELS,CTCF-bound": 1,
    "pELS": 1,
    "pELS,CTCF-bound": 1,
}

# Nine class CREs
NINE_CLASS_CREs = [
    "Low-DNase",
    "DNase-only",
    "CTCF-only,CTCF-bound",
    "DNase-H3K4me3",
    "DNase-H3K4me3,CTCF-bound",
    "PLS",
    "PLS,CTCF-bound",
    "ELS",
    "ELS,CTCF-bound",
]
MAP_NINE_CLASS_CRE_TO_IDX = {
    "Low-DNase": 0,
    "DNase-only": 1,
    "CTCF-only,CTCF-bound": 2,
    "DNase-H3K4me3": 3,
    "DNase-H3K4me3,CTCF-bound": 4,
    "PLS": 5,
    "PLS,CTCF-bound": 6,
    "dELS": 7,
    "dELS,CTCF-bound": 8,
    "pELS": 7,
    "pELS,CTCF-bound": 8,
}

# Multi-class CREs
CREs = [
    "Low-DNase",
    "DNase-only",
    "CTCF-only,CTCF-bound",
    "DNase-H3K4me3",
    "DNase-H3K4me3,CTCF-bound",
    "PLS",
    "PLS,CTCF-bound",
    "dELS",
    "dELS,CTCF-bound",
    "pELS",
    "pELS,CTCF-bound",
]
MAP_CRE_TO_IDX = {cre: idx for idx, cre in enumerate(CREs)}

# Tissue types
# TISSUES = ['transverse_colon', 'thyroid_gland', 'adrenal_gland','upper_lobe_of_left_lung', 'stomach', 'gastrocnemius_medialis']
# MAP_TISSUE_TO_IDX = {tissue: idx for idx, tissue in enumerate(TISSUES)}

SPECIAL_TOKENS = {
    "pad_token": "<pad>",
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
}

# MULTI_CLASS_WEIGHTS = [1.3751975557238645, 23.33259446784185, 35.96540128396482, 179.2837613733136, 333.36111342611883, 26.864568991571243, 82.65658256359391, 14.981048973049093, 108.14025791451513, 17.8077739837326, 82.53719332274106]
# MULTI_CLASS_WEIGHTS =  [10.0,               50.0,              50.0,              100.0,             100.0,              50.0,               50.0,              50.0,               100,                50.0,             50.0]
MULTI_CLASS_WEIGHTS = [1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]

# sum_weights = sum(MULTI_CLASS_WEIGHTS)
# MULTI_CLASS_WEIGHTS = [weight/sum_weights for weight in MULTI_CLASS_WEIGHTS]
BINARY_CLASS_WEIGHTS = [1.3751975557238645, 3.66526256566547]
NINE_CLASS_WEIGHTS = [1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
