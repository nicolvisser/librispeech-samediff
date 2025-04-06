from pathlib import Path

import pandas as pd
import pkg_resources


def get_subset_list():
    resource_package = __name__
    resource_path = "data"
    data_path = pkg_resources.resource_filename(resource_package, resource_path)
    csv_files = [f.stem for f in Path(data_path).glob("*.csv")]
    return csv_files


def read_data(subset):
    resource_package = __name__
    resource_path = f"data/{subset}.csv"
    data_path = pkg_resources.resource_filename(resource_package, resource_path)
    df = pd.read_csv(data_path, index_col=0)
    return df
