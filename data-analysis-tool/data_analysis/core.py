import io
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy.stats import chi2_contingency, pointbiserialr
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    MinMaxScaler,
    StandardScaler,
    RobustScaler
)

try:
    from google.colab import files
except Exception:
    files = None


class DataInspector:
    """
    DataInspector is a reusable class for uploading, cleaning,
    normalizing, encoding, and visualizing CSV datasets.
    """

    def __init__(self, dataframe=None):
        """
        Initialize DataInspector with an optional pandas DataFrame.
        """
        self.df = dataframe

    def _check_data(self):
        """
        Check whether data is available.
        """
        if self.df is None or self.df.empty:
            print("No data available. Please upload or load a dataset first.")
            return False
        return True

    def upload_data(self, file_path=None):
        """
        Upload or load a CSV file.

        If file_path is given, it loads that CSV file.
        If file_path is not given, it opens Google Colab file upload.
        """
        if file_path is not None:
            self.df = pd.read_csv(file_path)
        else:
            if files is None:
                print("Google Colab upload is not available here.")
                return None

            uploaded = files.upload()
            file_name = list(uploaded.keys())[0]
            self.df = pd.read_csv(io.BytesIO(uploaded[file_name]))

        garbage_values = ["?", "n/a", "N/A", "NULL", "null", " ", ""]
        self.df.replace(garbage_values, np.nan, inplace=True)

        for col in self.df.columns:
            converted = pd.to_numeric(self.df[col], errors="coerce")
            if not converted.isna().all():
                self.df[col] = converted

        print("Dataset loaded successfully.")
        return self.df

    def summary(self):
        """
        Display dataset shape, first 20 rows, numerical columns,
        and categorical columns.
        """
        if not self._check_data():
            return None

        print("Rows:", self.df.shape[0])
        print("Columns:", self.df.shape[1])

        print("\nFirst 20 rows:")
        display(self.df.head(20))

        numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = self.df.select_dtypes(exclude=np.number).columns.tolist()

        print("\nNumeric columns:")
        print(numeric_cols)

        print("\nCategorical columns:")
        print(categorical_cols)

    def handle_missing_values(self, strategy="mean", fill_value=None, columns=None):
        """
        Handle missing values using mean, median, mode, or constant value.
        """
        if not self._check_data():
            return None

        if columns is None:
            columns = self.df.columns

        for col in columns:
            if col not in self.df.columns:
                continue

            if self.df[col].isna().sum() == 0:
                continue

            if strategy == "mean":
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col] = self.df[col].fillna(self.df[col].mean())
                else:
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0])

            elif strategy == "median":
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                else:
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0])

            elif strategy == "mode":
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])

            elif strategy == "constant":
                self.df[col] = self.df[col].fillna(fill_value)

        print("Missing values handled successfully.")
        return self.df

    def remove_duplicates(self):
        """
        Remove exact duplicate rows.
        """
        if not self._check_data():
            return None

        before = len(self.df)
        self.df = self.df.drop_duplicates()
        after = len(self.df)

        print(f"Removed {before - after} duplicate rows.")
        return self.df

    def handle_outliers(self, column, action="flag"):
        """
        Detect or remove outliers using IQR method.

        action='flag' returns outlier rows.
        action='remove' removes outlier rows from dataset.
        """
        if not self._check_data():
            return None

        if column not in self.df.columns:
            print("Column not found.")
            return None

        if not pd.api.types.is_numeric_dtype(self.df[column]):
            print("Outlier detection works only with numeric columns.")
            return None

        q1 = self.df[column].quantile(0.25)
        q3 = self.df[column].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = self.df[(self.df[column] < lower) | (self.df[column] > upper)]

        if action == "remove":
            self.df = self.df[(self.df[column] >= lower) & (self.df[column] <= upper)]
            print(f"Removed {len(outliers)} outlier rows.")
            return self.df

        print(f"Found {len(outliers)} outlier rows.")
        return outliers

    def delete_rows(self, row_indexes):
        """
        Delete rows using comma-separated row indexes.
        Example: '1,2,5'
        """
        if not self._check_data():
            return None

        indexes = [int(i.strip()) for i in row_indexes.split(",")]
        self.df = self.df.drop(index=indexes, errors="ignore")

        print("Selected rows deleted.")
        return self.df

    def delete_columns(self, columns):
        """
        Delete columns using comma-separated column names.
        Example: 'Name,Ticket,Cabin'
        """
        if not self._check_data():
            return None

        col_list = [c.strip() for c in columns.split(",")]
        self.df = self.df.drop(columns=col_list, errors="ignore")

        print("Selected columns deleted.")
        return self.df

    def extract_normalized_numeric_data(self, method="standard"):
        """
        Normalize numeric columns using minmax, standard, or robust scaling.
        """
        if not self._check_data():
            return None

        numeric_df = self.df.select_dtypes(include=np.number)

        if numeric_df.empty:
            print("No numeric columns found.")
            return pd.DataFrame()

        numeric_df = numeric_df.fillna(numeric_df.median())

        if method == "minmax":
            scaler = MinMaxScaler()
        elif method == "robust":
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()

        scaled_data = scaler.fit_transform(numeric_df)

        return pd.DataFrame(
            scaled_data,
            columns=numeric_df.columns,
            index=self.df.index
        )

    def extract_normalized_categorical_data(self, method="onehot"):
        """
        Encode categorical columns using onehot, ordinal, or uniform encoding.
        """
        if not self._check_data():
            return None

        categorical_df = self.df.select_dtypes(exclude=np.number)

        if categorical_df.empty:
            print("No categorical columns found.")
            return pd.DataFrame()

        categorical_df = categorical_df.fillna("Missing")

        if method == "onehot":
            try:
                encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            except TypeError:
                encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")

            encoded = encoder.fit_transform(categorical_df)
            columns = encoder.get_feature_names_out(categorical_df.columns)

            return pd.DataFrame(encoded, columns=columns, index=self.df.index)

        elif method == "ordinal":
            encoder = OrdinalEncoder()
            encoded = encoder.fit_transform(categorical_df)

            return pd.DataFrame(
                encoded,
                columns=categorical_df.columns,
                index=self.df.index
            )

        elif method == "uniform":
            encoder = OrdinalEncoder()
            encoded = encoder.fit_transform(categorical_df)

            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(encoded)

            return pd.DataFrame(
                scaled,
                columns=categorical_df.columns,
                index=self.df.index
            )

    def merge_normalized_data(self, numeric_method="standard", categorical_method="onehot"):
        """
        Merge normalized numeric data with encoded categorical data.
        """
        numeric_data = self.extract_normalized_numeric_data(method=numeric_method)
        categorical_data = self.extract_normalized_categorical_data(method=categorical_method)

        merged = pd.concat([numeric_data, categorical_data], axis=1)

        print("Normalized numeric and categorical data merged successfully.")
        return merged

    def plot_numeric_distribution(self, column):
        """
        Create a 3-panel numeric plot:
        box plot, scatter plot, and histogram.
        """
        if not self._check_data():
            return None

        if column not in self.df.columns:
            print("Column not found.")
            return None

        if not pd.api.types.is_numeric_dtype(self.df[column]):
            print("This plot works only for numeric columns.")
            return None

        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=["Box Plot", "Index vs Value", "Histogram"]
        )

        fig.add_trace(go.Box(x=self.df[column], name="Box"), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df[column], mode="markers", name="Scatter"), row=1, col=2)
        fig.add_trace(go.Histogram(x=self.df[column], name="Histogram"), row=1, col=3)

        fig.update_layout(title=f"Distribution Analysis of {column}")
        fig.show()

    def plot_relationship(self, col1, col2):
        """
        Automatically plot relationship based on column data types.

        Numeric-Numeric: scatter plot with trendline.
        Categorical-Numeric: box plot.
        Categorical-Categorical: grouped bar chart.
        """
        if not self._check_data():
            return None

        if col1 not in self.df.columns or col2 not in self.df.columns:
            print("One or both columns not found.")
            return None

        col1_numeric = pd.api.types.is_numeric_dtype(self.df[col1])
        col2_numeric = pd.api.types.is_numeric_dtype(self.df[col2])

        if col1_numeric and col2_numeric:
            fig = px.scatter(self.df, x=col1, y=col2, trendline="ols",
                             title=f"{col1} vs {col2}")

        elif not col1_numeric and col2_numeric:
            fig = px.box(self.df, x=col1, y=col2, points="all",
                         title=f"{col2} by {col1}")

        elif col1_numeric and not col2_numeric:
            fig = px.box(self.df, x=col2, y=col1, points="all",
                         title=f"{col1} by {col2}")

        else:
            grouped = self.df.groupby([col1, col2]).size().reset_index(name="count")
            fig = px.bar(grouped, x=col1, y="count", color=col2, barmode="group",
                         title=f"{col1} vs {col2}")

        fig.show()

    def plot_categorical_frequency(self, column):
        """
        Plot categorical frequency with count and percentage labels.
        """
        if not self._check_data():
            return None

        if column not in self.df.columns:
            print("Column not found.")
            return None

        counts = self.df[column].value_counts(dropna=False).reset_index()
        counts.columns = [column, "count"]
        counts["percentage"] = round((counts["count"] / counts["count"].sum()) * 100, 2)
        counts["label"] = counts["count"].astype(str) + " (" + counts["percentage"].astype(str) + "%)"

        fig = px.bar(counts, x=column, y="count", text="label",
                     title=f"Frequency of {column}")
        fig.show()

    def _cramers_v(self, x, y):
        """
        Calculate Cramér's V for categorical-categorical association.
        """
        confusion_matrix = pd.crosstab(x, y)
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()

        if n == 0:
            return 0

        r, k = confusion_matrix.shape
        return np.sqrt(chi2 / (n * (min(k - 1, r - 1))))

    def _eta_squared(self, categories, values):
        """
        Calculate eta correlation ratio for categorical-numeric association.
        """
        data = pd.DataFrame({"cat": categories, "num": values}).dropna()

        if data.empty:
            return 0

        groups = [group["num"].values for name, group in data.groupby("cat")]

        overall_mean = data["num"].mean()
        between_group = sum(len(group) * (group.mean() - overall_mean) ** 2 for group in groups)
        total = sum((data["num"] - overall_mean) ** 2)

        if total == 0:
            return 0

        return np.sqrt(between_group / total)

    def plot_all_associations_heatmap(self):
        """
        Plot a unified association heatmap for all columns.

        Numeric-Numeric: Pearson correlation.
        Categorical-Categorical: Cramér's V.
        Mixed Numeric-Categorical: Point-Biserial or Eta.
        """
        if not self._check_data():
            return None

        columns = self.df.columns
        assoc = pd.DataFrame(index=columns, columns=columns, dtype=float)

        for col1 in columns:
            for col2 in columns:

                if col1 == col2:
                    assoc.loc[col1, col2] = 1.0
                    continue

                col1_numeric = pd.api.types.is_numeric_dtype(self.df[col1])
                col2_numeric = pd.api.types.is_numeric_dtype(self.df[col2])

                try:
                    if col1_numeric and col2_numeric:
                        assoc.loc[col1, col2] = self.df[[col1, col2]].corr().iloc[0, 1]

                    elif not col1_numeric and not col2_numeric:
                        assoc.loc[col1, col2] = self._cramers_v(self.df[col1], self.df[col2])

                    else:
                        if col1_numeric:
                            num_col = col1
                            cat_col = col2
                        else:
                            num_col = col2
                            cat_col = col1

                        unique_count = self.df[cat_col].nunique()

                        if unique_count == 2:
                            temp = self.df[[cat_col, num_col]].dropna()
                            codes = pd.factorize(temp[cat_col])[0]
                            corr, _ = pointbiserialr(codes, temp[num_col])
                            assoc.loc[col1, col2] = abs(corr)
                        else:
                            assoc.loc[col1, col2] = self._eta_squared(self.df[cat_col], self.df[num_col])

                except Exception:
                    assoc.loc[col1, col2] = 0

        assoc = assoc.fillna(0)

        fig = px.imshow(
            assoc,
            text_auto=True,
            title="Unified Association Heatmap"
        )
        fig.show()


class PlottingMethods:
    """
    Separate plotting class for reusable Plotly chart methods.
    """

    @staticmethod
    def bar_chart(df, column):
        """
        Create a bar chart and return it as HTML.
        """
        if df is None or df.empty:
            return "<p>No data available.</p>"

        fig = px.bar(df[column].value_counts().reset_index(),
                     x="index",
                     y=column,
                     title=f"Bar Chart of {column}")

        return fig.to_html()

    @staticmethod
    def pie_chart(df, column):
        """
        Create a pie chart and return it as HTML.
        """
        if df is None or df.empty:
            return "<p>No data available.</p>"

        fig = px.pie(df, names=column, title=f"Pie Chart of {column}")
        return fig.to_html()

    @staticmethod
    def histogram(df, column):
        """
        Create a histogram and return it as HTML.
        """
        if df is None or df.empty:
            return "<p>No data available.</p>"

        fig = px.histogram(df, x=column, title=f"Histogram of {column}")
        return fig.to_html()
