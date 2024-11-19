
import pandas as pd
from io import BytesIO
from matplotlib.figure import Figure
from sklearn.model_selection import train_test_split
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
        if file_path.endswith(".xlsx"):
            try:
                return pd.read_excel(file_path, engine='openpyxl')
            except Exception as e:
                print(f"Error loading .xlsx file {file_path}: {e}")

        elif file_path.endswith(".xls"):
            try:
                return pd.read_excel(file_path, engine='xlrd')
            except Exception as e:
                print(f"Error loading .xls file {file_path}: {e}")

        elif file_path.endswith(".csv"):
            try:
                return pd.read_csv(file_path)
            except Exception as e:
                print(f"Error loading CSV file {file_path}: {e}")
        else:
            print(f"Unsupported file type for {file_path}. Skipping.")
            return None


    def get_dummies(self) -> pd.DataFrame:

        df: pd.DataFrame = self.opened_file
        categorical_cols = df.select_dtypes(include=["object"]).columns

        df_dummies : pd.DataFrame = pd.get_dummies(df, categorical_cols, drop_first=True)

        for col in df_dummies.select_dtypes(include='bool'):
            df_dummies[col] = df_dummies[col].astype(int)

        return df_dummies


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
    # dummies = data.get_dummies()
    # print(dummies)
    train, test = train_test_split(file, test_size=0.2, random_state=42)
    print(train)
    print(test)
    print(train.columns)
