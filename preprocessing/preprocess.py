
import pandas as pd
from io import BytesIO
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns

class Preprocess():

    def __init__(self, file: BytesIO):

        self.opened_file = self._load_file(file)

    def _load_file(self, file_path: str) -> BytesIO:

        """
        Load the file based on its extension.
        :param file_path: Path to the file to be loaded
        :return: DataFrame if successfully loaded, otherwise None
        """
        try:
            if file_path.endswith(".xlsx"):
                return pd.read_excel(file_path, engine="openpyxl")
            elif file_path.endswith(".xls"):
                return pd.read_excel(file_path, engine="xlrd")
            elif file_path.endswith(".csv"):
                return pd.read_csv(file_path)
            else:
                print(f"Unsupported file type for {file_path}.")
                return None
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return None

    def get_dummies(self, dataframe: pd.DataFrame) -> pd.DataFrame:

        """
        Converts categorical columns in a DataFrame into dummy/indicator variables.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The input DataFrame containing categorical columns to be transformed.

        Returns
        -------
        pd.DataFrame
            A DataFrame with categorical columns converted to dummy variables. Boolean columns
            are also converted to integers (0 or 1).
        """

        df: pd.DataFrame = dataframe.copy()
        categorical_cols = df.select_dtypes(include=["object"]).columns

        df_dummies : pd.DataFrame = pd.get_dummies(df, categorical_cols, drop_first=True)

        for col in df_dummies.select_dtypes(include='bool'):
            df_dummies[col] = df_dummies[col].astype(int)

        return df_dummies

    def normalize(self, dataframe: pd.DataFrame, exclude_columns: list = None) -> pd.DataFrame:


        """
        Normaliza las columnas numéricas de un DataFrame, excluyendo las columnas especificadas.

        Parameters:
        - dataframe: pd.DataFrame - DataFrame a normalizar.
        - exclude_columns: list - Lista de columnas numéricas a excluir de la normalización.

        Returns:
        - pd.DataFrame: DataFrame con las columnas numéricas normalizadas (excepto las excluidas).
        """

        dataframe = dataframe.copy()

        numeric_columns = dataframe.select_dtypes(include=['float64', 'int64']).columns

        if exclude_columns is not None:
            numeric_columns = numeric_columns.difference(exclude_columns)

        # Normalizar solo las columnas seleccionadas
        dataframe[numeric_columns] = (dataframe[numeric_columns] - dataframe[numeric_columns].mean()) / (
            dataframe[numeric_columns].std()
        )

        return dataframe

    def distribution(self) -> Figure:

        """
        Plots the distribution of numerical columns in the dataset.

        This method selects all numerical columns from the `personal` DataFrame and creates
        histograms with kernel density estimation (KDE) for each column to visualize their
        distributions. Subplots are arranged in a grid with three columns per row. Any
        unused subplot axes are removed for a clean layout.

        Returns:
            Figure: A matplotlib Figure object displaying the distribution plots of each numerical column.
        """

        numeric_columns = self.personal.select_dtypes(include="number")

        num_columns = 3
        num_rows = (len(numeric_columns.columns) + num_columns - 1) // num_columns  # Calculate the needed rows
        fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, num_rows * 5))  # Adjust figsize as needed
        axes = axes.flatten()  # Flatten the axes array for easy indexing

        for idx, column in enumerate(numeric_columns):
            sns.histplot(numeric_columns[column], kde=True, ax=axes[idx])
            axes[idx].set_title(f'Distribution of {column}')

        for i in range(len(numeric_columns.columns), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout(h_pad=5)
        plt.show()

    def distribution_with_changes(self, old_dataframe: pd.DataFrame, new_dataframe: pd.DataFrame) -> Figure:

        """
        Plots the distribution comparison of numerical columns between two DataFrames.

        This method takes an original DataFrame and a modified DataFrame, then creates
        kernel density plots (KDE) for each numerical column to visually compare their
        distributions. The original DataFrame's distribution is shown in blue, and the
        modified DataFrame's distribution is shown in red. Subplots are arranged in a
        grid with three columns per row, and unused axes are removed for a clean layout.

        Parameters:
            old_dataframe (pd.DataFrame): The original DataFrame to compare.
            new_dataframe (pd.DataFrame): The modified DataFrame to compare.

        Returns:
            Figure: A matplotlib Figure object displaying KDE plots comparing each numerical column's distribution.
        """

        numeric_columns = old_dataframe.select_dtypes(include="number").columns
        num_columns = 3
        num_rows = (len(numeric_columns) + num_columns - 1) // num_columns

        fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, num_rows * 5))
        axes = axes.flatten()

        for idx, column in enumerate(numeric_columns):
            sns.kdeplot(old_dataframe[column], ax=axes[idx], color="blue", label="Original", fill=True)
            sns.kdeplot(new_dataframe[column], ax=axes[idx], color="red", label="Modified", fill=True)

            axes[idx].set_title(f'Distribution of {column}')
            axes[idx].legend()

        for i in range(len(numeric_columns), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout(h_pad=5)
        plt.show()
        return fig


pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 20)




if __name__ == "__main__":
    route: str = "../data/Loan.csv"
    data = Preprocess(route)
    file = data.opened_file
    excluded_columns = ['PreviousLoanDefaults', 'PaymentHistory', 'LoanApproved']
    normal = data.normalize(file, exclude_columns=excluded_columns)
    print(normal)
    dummies = data.get_dummies(normal)
    print(dummies)


