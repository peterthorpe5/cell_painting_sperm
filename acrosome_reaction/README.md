# Acrosome Reaction Analysis Pipeline

A reproducible Python workflow for analysing acrosome reaction (AR) data extracted from CellProfiler measurements. The pipeline supports both **single-dose single-replicate screens** and **dose–response plates**, and produces QC metrics, statistical testing, ranking tables, and per-compound visualisations.

---

## Installation and Dependencies

### Required Python packages

The core analysis relies on the following external packages:

| Package | Purpose |
|------|--------|
| `numpy` | Numerical operations |
| `pandas` | Table manipulation and metadata handling |
| `scipy` | Fisher’s exact tests and curve fitting |
| `matplotlib` | All static plots and QC PDFs |

All other imports (`argparse`, `logging`, `pathlib`, `typing`, etc.) are part of the Python standard library.

---

### Optional packages

These are only required for specific features:

| Package | Required for |
|------|-------------|
| `plotly` | Interactive HTML volcano plots |
| `jinja2` | HTML report templating (if enabled) |

If these are not installed, the pipeline will still run but will skip interactive outputs.

---

## Conda installation (recommended)

```bash
conda create \
    --name acrosome_reaction \
    --channel conda-forge \
    python=3.10 \
    numpy \
    pandas \
    scipy \
    matplotlib \
    plotly \
    jinja2
```

```bash
conda activate acrosome_reaction
```

or install the required packages using pip in an existing python 3 instance:

Pip installation

```bash
pip install \
    numpy \
    pandas \
    scipy \
    matplotlib \
    plotly \
    jinja2
```

## Key Features

### 1. Data Loading
- Automatically discovers and loads CellProfiler output tables:
  - `Image.csv`
  - `Acrosome.csv`
  - optional: `FilteredNuclei.csv`, `SpermCells.csv`
- Handles `.csv` and `.csv.gz`.
- Identifies plates based on file naming convention.

### 2. Metadata Harmonisation
- Reads compound metadata (library file) and standardises:
  - `Plate` / `plate_id`
  - `Well` / `well_id`
  - `Compound ID` (`cpd_id`)
  - concentrations, dosing, and optional annotation fields.
- Normalises plate identifiers (e.g. stripping date suffixes).

### 3. Per-Well Metrics
- Computes per-well AR% from object-level data.
- Merges metadata with Image-level measurements.
- Creates `df_well_qc` with all required per-well measurements.

### 4. Quality Control
- Image-level QC (e.g. focus, illumination, object counts).
- Well-level QC with a `qc_keep` flag.
- Object-level QC PDF (counts, intensity summaries, distributions).

### 5. Fisher’s Exact Tests
- Performs compound-versus-DMSO Fisher’s exact tests.
- Supports:
  - pooled-plate DMSO (default)
  - optional per-plate Fisher logic (future flag)
- Produces:
  - AR% per compound
  - delta AR% (treatment – DMSO)
  - odds ratios
  - p-values and Benjamini–Hochberg q-values

### 6. Hit Ranking
- Ranks compounds by any chosen metric:
  - default: `delta_AR_pct`
- Writes:
  - `compound_ranking.tsv`
  - optional: `top20_compounds.tsv`
- Extracts top-N compound IDs for downstream plots.

### 7. Visualisation Outputs
Produces a comprehensive set of plots including:

- plate AR% heatmap  
- volcano plot (static and interactive HTML)  
- barplots:
  - top delta-AR compounds  
  - hit classes  
  - per-compound summaries  
- **global inducer boxplot** (DMSO vs all TEST compounds)  
- **per-compound boxplots** (DMSO vs each TEST compound), saved to: results/<plate>/boxplots_per_compound/ 





### 8. HTML Report
Generates a standalone HTML summary including:
- QC metrics
- key plots
- hit tables
- links to interactive visualisations
- links to per-compound boxplots

---

# Input Requirements

## Metadata file: required columns and format

The metadata (library / plate map) file links CellProfiler wells to compounds and concentrations.  
It must be a delimiter-detected text file (TSV strongly recommended).

### Essential columns

The following logical fields **must** be present, either under the exact name or one of the recognised aliases:

| Logical field | Required | Accepted column names / notes |
|-------------|----------|--------------------------------|
| Plate ID | Yes | `plate_id`, `Plate ID`, `Plate_Metadata` |
| Well ID | Yes | `well_id`, `Well`, `Well_Metadata`, or derived from `Row` + `Col` |
| Compound ID | Yes | `cpd_id`, `DDD`, `OBJDID` |
| Concentration | Yes (TEST compounds) | `conc`, `PTODCONCVALUE`; numeric |

### Well identifier handling

- All wells are normalised internally to `A01` format.
- The script can automatically construct `well_id` if you provide:
  - `Row` (A–H) **and** `Col` (1–12), or
  - `Well` values such as `A1`, `A01`, or `A001`.

### Concentration (`conc`) rules

- Must be numeric for all non-DMSO wells.
- DMSO wells should have `conc = NA` (recommended).
- Non-numeric strings (e.g. `10 uM`) are coerced to NA and excluded from testing.

### Example minimal metadata file (TSV)

```
plate_id	well_id	cpd_id	conc
PLATE_001	A01	DMSO	
PLATE_001	A02	DMSO	
PLATE_001	A03	CMPD_001	0.1
PLATE_001	A04	CMPD_001	1.0
PLATE_001	A05	CMPD_001	10.0
PLATE_001	B01	CMPD_002	0.1
PLATE_001	B02	CMPD_002	1.0
```


Optional but recommended columns

These columns are not required for analysis but are preserved in outputs and reports:

Column	Purpose
cpd_type	One of DMSO, POS, TEST (overrides automatic inference)
Library	Compound library name
SMILES, Series, Class	Annotation only
---



## CellProfiler Image table (required columns)

The *_Image.csv table must contain:

ImageNumber

Plate identifier column (default: Plate_Metadata)

Well identifier column (default: Well_Metadata)

Total cell count column (default: Count_SpermCells)

Acrosome-reacted cell count column (default: Count_Acrosome)

These column names can be overridden via command-line arguments.



## Example Commands
Basic single-plate run

```
python analyse_acrosome_dose_response.py \
    --cp_dir raw \
    --library_metadata metadata/plate_001_metadata.tsv \
    --output_dir results/plate_001 \
    --verbosity 1
```

With explicit controls and dose–response fitting

```
python analyse_acrosome_dose_response.py \
    --cp_dir raw \
    --library_metadata metadata/plate_001_metadata.tsv \
    --controls metadata/plate_001_controls.tsv \
    --fit_dose_response \
    --min_cells_per_well 100 \
    --output_dir results/plate_001 \
    --verbosity 2
```

Including optional motility data

```
python analyse_acrosome_dose_response.py \
    --cp_dir raw \
    --library_metadata metadata/plate_001_metadata.tsv \
    --motility_csv metadata/plate_001_motility.tsv \
    --output_dir results/plate_001

```

## Output Structure

Example output folder:

results/
INPUT_FOLDER/
acrosome_fisher_per_dose.tsv
acrosome_per_compound_summary.tsv
compound_ranking.tsv
top20_compounds.tsv
plate_AR_pct_heatmap.png
fisher_volcano.png
fisher_volcano_interactive.html
top_delta_AR_pct_barplot.png
inducer_boxplot.png
boxplots_per_compound/



---

## Typical Workflow

1. Place CellProfiler tables and metadata file in a plate folder.
2. Run the script:
   ```bash
   python analyse_acrosome_dose_response.py \
       --input_dir <path_to_plate_folder> \
       --metadata_file <metadata.tsv> \
       --output_dir results/
