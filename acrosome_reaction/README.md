# Acrosome Reaction Analysis Pipeline

A reproducible Python workflow for analysing acrosome reaction (AR) data extracted from CellProfiler measurements. The pipeline supports both **single-dose single-replicate screens** and **dose–response plates**, and produces QC metrics, statistical testing, ranking tables, and per-compound visualisations.

---

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

## Input Requirements

### Mandatory Inputs
- CellProfiler output tables containing:
- `total_cells`
- `total_AR_cells`
- metadata needed to identify wells and plates
- A compound metadata file with:
- `plate_id`
- `well_id`
- `cpd_id`
- concentration

### Optional Inputs
- Annotation fields such as SMILES, chemical series, formulae.
- Object-level tables for extra QC.

---

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
