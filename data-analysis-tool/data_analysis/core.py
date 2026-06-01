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

try:
    from IPython.display import display
except Exception:
    display = print


class DataInspector:
    """
    DataInspector is a reusable class for CSV data ingestion,
    data cleaning, normalization, encoding, visualization,
    and statistical association analysis.
    """

    def __init__(self, dataframe=None):
        """
        Initialize the DataInspector object.

        Parameters
        ----------
        dataframe : pandas.DataFrame, optional
            Existing DataFrame to use.
        """
        self.df = dataframe

    def _check_data(self):
        """
        Check whether a DataFrame is available.

        Returns
        -------
        bool
            True if data exists, False otherwise.
        """
        if self.df is None or self.df.empty:
            print("No data available. Please upload or load a dataset first.")
            return False
        return True

    def upload_data(self, file_path=None):
        """
        Upload or load a CSV file.

        If file_path is provided, the CSV file is loaded from the given path.
        If file_path is not provided, Google Colab file upload is used.

        This method also:
        - Replaces garbage values with NaN
        - Converts possible numeric columns into numeric type

        Parameters
        ----------
        file_path : str, optional
            CSV file path.

        Returns
        -------
        pandas.DataFrame
            Loaded and sanitized DataFrame.
        """
        if file_path is not None:
            self.df = pd.read_csv(file_path)
        else:
            if files is None:
                print("Google Colab upload is not available here.")
                return None

            uploaded = files.upload()

            if len(uploaded) == 0:
                print("No file uploaded.")
                return None

            file_name = list(uploaded.keys())[0]
            self.df = pd.read_csv(io.BytesIO(uploaded[file_name]))

        garbage_values = [
            "?",
            "n/a",
            "N/A",
            "NA",
            "NULL",
            "null",
            "None",
            "none",
            "",
            " "
        ]

        self.df.replace(garbage_values, np.nan, inplace=True)
        self.df.replace(r"^\s*$", np.nan, regex=True, inplace=True)

        for col in self.df.columns:
            converted = pd.to_numeric(self.df[col], errors="coerce")

            if not converted.isna().all():
                self.df[col] = converted

        print("Dataset loaded successfully.")
        return self.df

    def summary(self):
        """
        Display dataset summary.

        Shows:
        - Number of rows
        - Number of columns
        - First 20 rows
        - Numeric columns
        - Categorical columns
        - Missing value count
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

        print("\nMissing values:")
        display(self.df.isna().sum())

    def handle_missing_values(self, strategy="mean", fill_value=None, columns=None):
        """
        Handle missing values using selected strategy.

        Supported strategies:
        - mean
        - median
        - mode
        - constant

        Parameters
        ----------
        strategy : str
            Missing value handling method.
        fill_value : any, optional
            Value used when strategy is 'constant'.
        columns : list, optional
            Specific columns to handle. If None, all columns are handled.

        Returns
        -------
        pandas.DataFrame
            DataFrame after missing value handling.
        """
        if not self._check_data():
            return None

        strategy = strategy.lower()

        if columns is None:
            columns = self.df.columns

        for col in columns:
            if col not in self.df.columns:
                print(f"Column not found: {col}")
                continue

            if self.df[col].isna().sum() == 0:
                continue

            if strategy == "mean":
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col] = self.df[col].fillna(self.df[col].mean())
                else:
                    mode_value = self.df[col].mode(dropna=True)
                    self.df[col] = self.df[col].fillna(
                        mode_value[0] if not mode_value.empty else "Missing"
                    )

            elif strategy == "median":
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col] = self.df[col].fillna(self.df[col].median())
                else:
                    mode_value = self.df[col].mode(dropna=True)
                    self.df[col] = self.df[col].fillna(
                        mode_value[0] if not mode_value.empty else "Missing"
                    )

            elif strategy == "mode":
                mode_value = self.df[col].mode(dropna=True)
                self.df[col] = self.df[col].fillna(
                    mode_value[0] if not mode_value.empty else "Missing"
                )

            elif strategy == "constant":
                self.df[col] = self.df[col].fillna(fill_value)

            else:
                print("Invalid strategy. Use mean, median, mode, or constant.")
                return self.df

        print("Missing values handled successfully.")
        return self.df

    def remove_duplicates(self):
        """
        Remove exact duplicate rows.

        Returns
        -------
        pandas.DataFrame
            DataFrame after duplicate removal.
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
        Detect or remove outliers using the IQR method.

        Parameters
        ----------
        column : str
            Numeric column name.
        action : str
            'flag' returns outlier rows.
            'remove' removes outlier rows.

        Returns
        -------
        pandas.DataFrame
            Outlier rows or cleaned DataFrame.
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

        if iqr == 0:
            print("IQR is zero. No outliers detected using IQR method.")
            return pd.DataFrame()

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = self.df[
            (self.df[column] < lower) | (self.df[column] > upper)
        ]

        if action == "remove":
            self.df = self.df[
                (self.df[column] >= lower) & (self.df[column] <= upper)
            ]
            print(f"Removed {len(outliers)} outlier rows.")
            return self.df

        print(f"Found {len(outliers)} outlier rows.")
        return outliers

    def delete_rows(self, row_indexes):
        """
        Delete selected rows.

        Parameters
        ----------
        row_indexes : str
            Comma-separated row indexes.
            Example: '1,2,5'

        Returns
        -------
        pandas.DataFrame
            DataFrame after row deletion.
        """
        if not self._check_data():
            return None

        try:
            indexes = [int(i.strip()) for i in row_indexes.split(",")]
            self.df = self.df.drop(index=indexes, errors="ignore")
            print("Selected rows deleted.")
        except Exception as e:
            print("Invalid row indexes:", e)

        return self.df

    def delete_columns(self, columns):
        """
        Delete selected columns.

        Parameters
        ----------
        columns : str
            Comma-separated column names.
            Example: 'Name,Ticket,Cabin'

        Returns
        -------
        pandas.DataFrame
            DataFrame after column deletion.
        """
        if not self._check_data():
            return None

        col_list = [c.strip() for c in columns.split(",")]
        self.df = self.df.drop(columns=col_list, errors="ignore")

        print("Selected columns deleted.")
        return self.df

    def extract_normalized_numeric_data(self, method="standard"):
        """
        Normalize numeric columns.

        Supported methods:
        - minmax
        - standard
        - robust

        Parameters
        ----------
        method : str
            Scaling method.

        Returns
        -------
        pandas.DataFrame
            Normalized numeric DataFrame.
        """
        if not self._check_data():
            return None

        method = method.lower()
        numeric_df = self.df.select_dtypes(include=np.number)

        if numeric_df.empty:
            print("No numeric columns found.")
            return pd.DataFrame(index=self.df.index)

        numeric_df = numeric_df.fillna(numeric_df.median())

        if method == "minmax":
            scaler = MinMaxScaler()
        elif method == "robust":
            scaler = RobustScaler()
        elif method == "standard":
            scaler = StandardScaler()
        else:
            print("Invalid method. Use minmax, standard, or robust.")
            return numeric_df

        scaled_data = scaler.fit_transform(numeric_df)

        normalized_df = pd.DataFrame(
            scaled_data,
            columns=numeric_df.columns,
            index=self.df.index
        )

        return normalized_df

    def extract_normalized_categorical_data(self, method="onehot"):
        """
        Encode categorical columns.

        Supported methods:
        - onehot
        - ordinal
        - uniform

        Parameters
        ----------
        method : str
            Encoding method.

        Returns
        -------
        pandas.DataFrame
            Encoded categorical DataFrame.
        """
        if not self._check_data():
            return None

        method = method.lower()
        categorical_df = self.df.select_dtypes(exclude=np.number)

        if categorical_df.empty:
            print("No categorical columns found.")
            return pd.DataFrame(index=self.df.index)

        categorical_df = categorical_df.fillna("Missing")

        if method == "onehot":
            try:
                encoder = OneHotEncoder(
                    sparse_output=False,
                    handle_unknown="ignore"
                )
            except TypeError:
                encoder = OneHotEncoder(
                    sparse=False,
                    handle_unknown="ignore"
                )

            encoded = encoder.fit_transform(categorical_df)
            columns = encoder.get_feature_names_out(categorical_df.columns)

            return pd.DataFrame(
                encoded,
                columns=columns,
                index=self.df.index
            )

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

        else:
            print("Invalid method. Use onehot, ordinal, or uniform.")
            return pd.DataFrame(index=self.df.index)

    def merge_normalized_data(self, numeric_method="standard", categorical_method="onehot"):
        """
        Merge normalized numeric data with encoded categorical data.

        Parameters
        ----------
        numeric_method : str
            Numeric scaling method.
        categorical_method : str
            Categorical encoding method.

        Returns
        -------
        pandas.DataFrame
            Merged processed DataFrame.
        """
        if not self._check_data():
            return None

        numeric_data = self.extract_normalized_numeric_data(method=numeric_method)
        categorical_data = self.extract_normalized_categorical_data(method=categorical_method)

        merged = pd.concat([numeric_data, categorical_data], axis=1)

        print("Normalized numeric and categorical data merged successfully.")
        return merged

    def plot_numeric_distribution(self, column):
        """
        Create a 3-panel numeric visualization.

        Panels:
        - Horizontal violin and box plot
        - Scatter plot of index vs value
        - Histogram

        Parameters
        ----------
        column : str
            Numeric column name.
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
            subplot_titles=[
                "Horizontal Violin / Box",
                "Index vs Value",
                "Histogram"
            ]
        )

        fig.add_trace(
            go.Violin(
                x=self.df[column],
                box_visible=True,
                meanline_visible=True,
                name=column
            ),
            row=1,
            col=1
        )

        fig.add_trace(
            go.Scatter(
                x=self.df.index,
                y=self.df[column],
                mode="markers",
                name="Index vs Value"
            ),
            row=1,
            col=2
        )

        fig.add_trace(
            go.Histogram(
                x=self.df[column],
                name="Histogram"
            ),
            row=1,
            col=3
        )

        fig.update_layout(
            title=f"Distribution Analysis of {column}",
            showlegend=False
        )

        fig.show()

    def plot_relationship(self, col1, col2):
        """
        Plot relationship between two columns based on their data types.

        Numeric-Numeric:
            Scatter plot with OLS trendline.

        Categorical-Numeric:
            Box plot with all data points.

        Categorical-Categorical:
            Grouped bar chart.

        Parameters
        ----------
        col1 : str
            First column name.
        col2 : str
            Second column name.
        """
        if not self._check_data():
            return None

        if col1 not in self.df.columns or col2 not in self.df.columns:
            print("One or both columns not found.")
            return None

        col1_numeric = pd.api.types.is_numeric_dtype(self.df[col1])
        col2_numeric = pd.api.types.is_numeric_dtype(self.df[col2])

        if col1_numeric and col2_numeric:
            fig = px.scatter(
                self.df,
                x=col1,
                y=col2,
                trendline="ols",
                title=f"{col1} vs {col2}"
            )

        elif not col1_numeric and col2_numeric:
            fig = px.box(
                self.df,
                x=col1,
                y=col2,
                points="all",
                title=f"{col2} by {col1}"
            )

        elif col1_numeric and not col2_numeric:
            fig = px.box(
                self.df,
                x=col2,
                y=col1,
                points="all",
                title=f"{col1} by {col2}"
            )

        else:
            grouped = (
                self.df
                .groupby([col1, col2])
                .size()
                .reset_index(name="count")
            )

            fig = px.bar(
                grouped,
                x=col1,
                y="count",
                color=col2,
                barmode="group",
                title=f"{col1} vs {col2}"
            )

        fig.show()

    def plot_categorical_frequency(self, column):
        """
        Plot frequency of a categorical column using counts and percentages.

        Parameters
        ----------
        column : str
            Categorical column name.
        """
        if not self._check_data():
            return None

        if column not in self.df.columns:
            print("Column not found.")
            return None

        counts = self.df[column].value_counts(dropna=False).reset_index()
        counts.columns = [column, "count"]

        counts["percentage"] = round(
            (counts["count"] / counts["count"].sum()) * 100,
            2
        )

        counts["label"] = (
            counts["count"].astype(str)
            + " ("
            + counts["percentage"].astype(str)
            + "%)"
        )

        fig = px.bar(
            counts,
            x=column,
            y="count",
            text="label",
            title=f"Frequency of {column}"
        )

        fig.update_traces(textposition="outside")
        fig.show()

    def _cramers_v(self, x, y):
        """
        Calculate Cramér's V for categorical-categorical association.

        Parameters
        ----------
        x : pandas.Series
            First categorical column.
        y : pandas.Series
            Second categorical column.

        Returns
        -------
        float
            Cramér's V value.
        """
        confusion_matrix = pd.crosstab(x, y)

        if confusion_matrix.empty:
            return 0

        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()

        r, k = confusion_matrix.shape
        denominator = n * (min(k - 1, r - 1))

        if denominator == 0:
            return 0

        return np.sqrt(chi2 / denominator)

    def _eta_squared(self, categories, values):
        """
        Calculate eta correlation ratio for categorical-numeric association.

        Parameters
        ----------
        categories : pandas.Series
            Categorical variable.
        values : pandas.Series
            Numeric variable.

        Returns
        -------
        float
            Eta value.
        """
        data = pd.DataFrame(
            {
                "cat": categories,
                "num": values
            }
        ).dropna()

        if data.empty:
            return 0

        groups = [
            group["num"].values
            for _, group in data.groupby("cat")
        ]

        if len(groups) <= 1:
            return 0

        overall_mean = data["num"].mean()

        between_group = sum(
            len(group) * (group.mean() - overall_mean) ** 2
            for group in groups
        )

        total = sum((data["num"] - overall_mean) ** 2)

        if total == 0:
            return 0

        return np.sqrt(between_group / total)

    def plot_all_associations_heatmap(self):
        """
        Plot a unified association heatmap for all columns.

        Association methods:
        - Numeric-Numeric: Pearson correlation
        - Categorical-Categorical: Cramér's V
        - Numeric-Categorical with two categories: Point-Biserial correlation
        - Numeric-Categorical with more than two categories: Eta correlation
        """
        if not self._check_data():
            return None

        columns = self.df.columns
        assoc = pd.DataFrame(
            index=columns,
            columns=columns,
            dtype=float
        )

        for col1 in columns:
            for col2 in columns:

                if col1 == col2:
                    assoc.loc[col1, col2] = 1.0
                    continue

                col1_numeric = pd.api.types.is_numeric_dtype(self.df[col1])
                col2_numeric = pd.api.types.is_numeric_dtype(self.df[col2])

                try:
                    if col1_numeric and col2_numeric:
                        corr_value = self.df[[col1, col2]].corr().iloc[0, 1]
                        assoc.loc[col1, col2] = abs(corr_value)

                    elif not col1_numeric and not col2_numeric:
                        assoc.loc[col1, col2] = self._cramers_v(
                            self.df[col1],
                            self.df[col2]
                        )

                    else:
                        if col1_numeric:
                            num_col = col1
                            cat_col = col2
                        else:
                            num_col = col2
                            cat_col = col1

                        temp = self.df[[cat_col, num_col]].dropna()
                        unique_count = temp[cat_col].nunique()

                        if unique_count == 2:
                            codes = pd.factorize(temp[cat_col])[0]
                            corr, _ = pointbiserialr(codes, temp[num_col])

                            if np.isnan(corr):
                                assoc.loc[col1, col2] = 0
                            else:
                                assoc.loc[col1, col2] = abs(corr)
                        else:
                            assoc.loc[col1, col2] = self._eta_squared(
                                self.df[cat_col],
                                self.df[num_col]
                            )

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

    Methods return HTML-wrapped Plotly figures for flexible embedding.
    """

    @staticmethod
    def bar_chart(df, column):
        """
        Create a bar chart and return it as HTML.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataset.
        column : str
            Column name.

        Returns
        -------
        str
            HTML representation of the Plotly figure.
        """
        if df is None or df.empty:
            return "<p>No data available.</p>"

        if column not in df.columns:
            return "<p>Column not found.</p>"

        counts = df[column].value_counts(dropna=False).reset_index()
        counts.columns = [column, "count"]

        fig = px.bar(
            counts,
            x=column,
            y="count",
            title=f"Bar Chart of {column}"
        )

        return fig.to_html()

    @staticmethod
    def pie_chart(df, column):
        """
        Create a pie chart and return it as HTML.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataset.
        column : str
            Column name.

        Returns
        -------
        str
            HTML representation of the Plotly figure.
        """
        if df is None or df.empty:
            return "<p>No data available.</p>"

        if column not in df.columns:
            return "<p>Column not found.</p>"

        fig = px.pie(
            df,
            names=column,
            title=f"Pie Chart of {column}"
        )

        return fig.to_html()

    @staticmethod
    def histogram(df, column):
        """
        Create a histogram and return it as HTML.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataset.
        column : str
            Column name.

        Returns
        -------
        str
            HTML representation of the Plotly figure.
        """
        if df is None or df.empty:
            return "<p>No data available.</p>"

        if column not in df.columns:
            return "<p>Column not found.</p>"

        fig = px.histogram(
            df,
            x=column,
            title=f"Histogram of {column}"
        )

        return fig.to_html()
