
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.feature_selection import VarianceThreshold

# Configure logging
logging.basicConfig(
    filename='feature_select.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

logger.info("Feature selection script started.")


def variance_threshold_selector(data: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    """
    Filters columns based on low variance.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing numerical features.
    threshold : float, optional
        Variance threshold for feature selection (default is 0.05).

    Returns
    -------
    pd.DataFrame
        DataFrame with features above the specified variance threshold.
    """
    logger.debug("Applying variance threshold selector.")
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    selected_data = data.iloc[:, selector.get_support(indices=True)]
    logger.debug(f"Selected {selected_data.shape[1]} features out of {data.shape[1]} after variance filtering.")
    return selected_data


def csv_parser(file: Path) -> pd.DataFrame:
    """
    Reads normalised CSV files and annotates organelle information.

    Parameters
    ----------
    file : Path
        Path to the input CSV file.

    Returns
    -------
    pd.DataFrame
        Parsed and formatted DataFrame.
    """
    organelle_map = {'acrosome': 'acrosome', 'nucleus': 'nucleus', 'mitochondria': 'mitochondria'}
    organelle = next((name for key, name in organelle_map.items() if key in file.stem.lower()), None)
    if organelle is None:
        raise ValueError(f"Unknown organelle type in filename: {file.stem}")

    logger.debug(f"Parsing file: {file}")
    df = pd.read_csv(file)
    logger.debug(f"Sample data from {file.name}:\n{df.head()}")

    df = (df.assign(fn=file.stem)
          .rename(columns=lambda x: x.replace('Cy5', 'AR'))
          .set_index(['Plate_Metadata', 'Well_Metadata'])
          .add_suffix(f"_{organelle}"))
    return df


# Load annotation data
annotation_path = Path('/uod/npsc/Lab_Book/BMGF/NHCP/SCP/STB/04022025/KVP_4Plates_04022025.csv')
logger.info("Loading annotation data.")
ddu = pd.read_csv(annotation_path)
logger.info(f"Columns in annotation file before renaming: {list(ddu.columns)}")



# Ensure the correct column names exist before renaming
if 'Plate' in ddu.columns and 'Well' in ddu.columns:
    ddu = ddu.rename(columns={'Plate': 'Plate_Metadata', 'Well': 'Well_Metadata'})
else:
    raise KeyError("Annotation file does not contain expected 'Plate' and 'Well' columns")

# Ensure 'Plate_Metadata' does not contain NaN before using .str.contains()
ddu['Plate_Metadata'] = ddu['Plate_Metadata'].astype(str).fillna('')

# Standardise Well_Metadata format (e.g., 'A01' → 'A1')
ddu['Well_Metadata'] = ddu['Well_Metadata'].astype(str)
ddu['Well_Metadata'] = ddu['Well_Metadata'].str[0] + ddu['Well_Metadata'].str[1:].astype(int).astype(str)

# Standardise plate metadata names
plate_patterns = {
    'NPSCDD000401': '20241129_NPSCDD000401_STB',
    'NPSCDD000400': '20241129_NPSCDD000400_STB',
    'NPSCDD0003971': 'NPSCDD0003971_05092024',
    'NPSCDD0003972': 'NPSCDD0003972_05092024'
}

for pattern, replacement in plate_patterns.items():
    mask = ddu['Plate_Metadata'].str.contains(pattern, na=False)
    ddu.loc[mask, 'Plate_Metadata'] = replacement

# Drop 'Unnamed: 0' if it exists
ddu.drop(columns=['Unnamed: 0'], errors='ignore', inplace=True)

# Reset index and set new index
ddu.reset_index(inplace=True, drop=True)
ddu.set_index(['Plate_Metadata', 'Well_Metadata'], inplace=True)

logger.info(f"Annotation data loaded with shape: {ddu.shape}")
logger.debug(f"Annotation data sample:\n{ddu.head()}")



# Load and parse SCP data
input_directory = Path('/uod/npsc/Lab_Book/BMGF/NHCP/SCP/STB/IXM_data/MCP09_ext_3751/Processed_data/')
input_files = list(input_directory.glob('NPSCDD000400*csv*'))
logger.info(f"Loading SCP data from {len(input_files)} files.")
parsed_dfs = [csv_parser(file) for file in input_files]

# Merge datasets
df = parsed_dfs[0].join(parsed_dfs[1:]).join(ddu, how='inner').reset_index()
df.query('~Plate_Metadata.str.contains("Plate_Metadata")', inplace=True)
df.set_index(['Source_Plate_Barcode', 'Source_well', 'name', 'Plate_Metadata', 'Well_Metadata', 'cpd_id', 'cpd_type'], inplace=True)
logger.info(f"Merged data shape: {df.shape}")

# Drop unwanted columns before correlation analysis
filter_regex = 'Notes|Seahorse_alert|Treatment|Number|Child|Paren|Location_[XYZ]|ZernikePhase|Euler|Plate|Well|Field|Center_[XYZ]|no_|fn_'
df_numeric = df.select_dtypes(include=[np.number]).copy()
df_numeric.drop(columns=df.filter(regex=filter_regex).columns, errors='ignore', inplace=True)

# Handle non-numeric data explicitly
non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
if len(non_numeric_cols) > 0:
    logger.warning(f"Non-numeric columns found: {non_numeric_cols.tolist()}")
    df2 = df.drop(columns=non_numeric_cols)
else:
    df2 = df

# Correlation-based feature selection
correlation_threshold = 0.99
logger.info("Computing correlation matrix.")
cm = df2.corr().abs()
upper_tri = cm.where(np.triu(np.ones(cm.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > correlation_threshold)]
logger.debug(f"Dropping {len(to_drop)} correlated features.")
df3 = df2.drop(columns=to_drop)

# Variance thresholding
final_df = variance_threshold_selector(df3)

logger.info(f"Final data shape after feature selection: {final_df.shape}")
logger.debug(f"Final data sample:\n{final_df.head()}")

# Save cleaned data
final_df.to_csv("STB_NPSCDD000400_05092024_normalised.csv")
logger.info("Cleaned data saved successfully.")
print("Feature selection and cleaning completed. Data saved to 'STB_NPSCDD000400_05092024_normalised.csv'.")
