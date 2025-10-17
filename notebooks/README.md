# Goal
These are applications:
    vcf2exp: Takes a VCF file representing a donor's genome and returns gene expression per tissue
    vcf2risk: Takes a VCF file representing a donor's genome and returns Alzheimer's risk per gene and tissue

# How tos
## Install dependencies
Make sure you have installed all dependencies for the core library:
```shell
cd variantformer
./install-packages.rc.sh
```

Once that is done, activate your new virtual environment

```shell
export VENV_DIR=~/.vcm-venv
source $VENV_DIR/bin/activate
```

and run the notebooks

Note: the vcf2risk.py does not work yet. It has an issue with importing dependencies from the source folder. 
