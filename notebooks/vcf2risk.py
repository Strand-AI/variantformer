import marimo

__generated_with = "0.13.9"
app = marimo.App(app_title="VCF2risk")


@app.cell
def _(mo):
    mo.md(r"""# Alzheimer's diagnosis risk prediction (per gene)""")
    return


@app.cell
def _(Path):
    import marimo as mo
    import hashlib

    UPLOAD_DIR = Path("uploads")
    UPLOAD_DIR.mkdir(exist_ok=True)

    def _target_path(u) -> Path:
        # stable name using content hash to avoid duplicates
        digest = hashlib.sha1(u.contents).hexdigest()[:8]
        stem = Path(u.name).stem
        suffix = Path(u.name).suffix
        return UPLOAD_DIR / f"{stem}_{digest}{suffix}"

    def maybe_save(u):
        if not u:  # no file uploaded yet
            return None
        path = _target_path(u)
        if not path.exists():  # idempotent
            path.write_bytes(u.contents)
        return path

    return maybe_save, mo


@app.cell
def _(mo):
    file = mo.ui.file(
        multiple=False, filetypes=[".vcf", ".gz"]
    )  # single VCF or gzipped file
    mo.md(f"Upload a VCF file: {file}")
    return (file,)


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    # from processors import ad_risk

    return (Path,)


@app.cell
def _():
    import os

    print(os.environ["PYTHONPATH"])
    return


@app.cell
def _(file, maybe_save):
    if file.value:
        saved_path = maybe_save(file.value[0])
    return


@app.function
def run_analysis(gene_id, tissue_ids):
    print(f"Gene: {gene_id}")
    print(f"Tissues: {tissue_ids}")


@app.cell
def _(mo):
    gene_selector = mo.ui.dropdown(
        options={
            "ENSG00000115419.12": 1,
            "ENSG00000115419.12": 2,
            "ENSG00000115419.12": 3,
        },
        value="ENSG00000115419.12",
        label="",
        searchable=True,
    )
    mo.md(f"Choose 1 gene {gene_selector}")
    return (gene_selector,)


@app.cell
def _(mo):
    tissue_selector = mo.ui.multiselect(
        options={"adipose - subcutaneous": 1, "blood": 2},
        label="",
    )
    mo.md(f"Choose multiple tissues {tissue_selector}")
    return (tissue_selector,)


@app.cell
def _(mo):
    run_analysis_button = mo.ui.run_button(label="Generate AD risk per gene")
    mo.md(f"Analyse AD risk per gene {run_analysis_button}")
    return (run_analysis_button,)


@app.cell
def _(gene_selector, run_analysis_button, tissue_selector):
    if run_analysis_button.value:
        run_analysis(gene_selector.value, tissue_selector.value)
    return


@app.cell
def _():
    import plotly.express as px

    # Example data
    fruits = ["Apples", "Oranges", "Bananas", "Grapes", "Pears"]
    counts = [10, 15, 7, 12, 5]

    # Create bar plot
    fig = px.bar(
        x=fruits, y=counts, labels={"x": "Fruit", "y": "Count"}, title="Fruit Counts"
    )

    # Show in notebook or interactive session
    fig.show()

    return


if __name__ == "__main__":
    app.run()
