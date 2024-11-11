import numpy as np
import pandas as pd
from io import BytesIO
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class Preprocess():

    def __init__(self, file: BytesIO):

        self.opened_file = self._load_file(file)
        self.personal = self._get_info()

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

    def _get_info(self) -> pd.DataFrame:

        """
        Filters the dataset for rows where the loan intent is 'PERSONAL'.

        This method accesses the opened file (a DataFrame stored as an attribute),
        filters it to include only rows where the 'loan_intent' column has the
        value 'PERSONAL', and returns the filtered DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing only rows where 'loan_intent' is 'PERSONAL'.
        """


        df = self.opened_file
        df = df[df["loan_intent"] == "PERSONAL"]

        return df

    def null_proportion(self) -> pd.DataFrame:

        """
        Calculates the proportion of null values for each column in the dataset.

        This method computes the proportion of missing (null) values for each column
        in the opened file (a DataFrame stored as an attribute). It also calculates
        the overall proportion of null values across the entire DataFrame, adding it
        as a 'Total' row. The results are returned as a DataFrame.

        Returns:
            pd.DataFrame: A DataFrame with two columns - 'Column' and 'Null Proportion',
                          showing the proportion of null values per column and overall.
        """

        null_proportion = dict()

        for column in self.opened_file.columns:
            null_proportion[column] = round(
                len(self.opened_file[self.opened_file[column].isnull()]) / len(self.opened_file), 3)

        null_proportion["Total"] = round(self.opened_file.isnull().sum().sum() / self.opened_file.size, 3)
        null_df = pd.DataFrame(list(null_proportion.items()), columns=['Column', 'Null Proportion'])

        return null_df

    def _change_outliers(self, dataframe: pd.DataFrame) -> pd.DataFrame:

        """
        Replaces outliers in specific columns with the median value.

        This function creates a copy of the provided DataFrame and replaces values
        greater than 100 in the specified columns ('person_age' and 'person_emp_length')
        with the median of that column. The purpose is to handle outliers in these
        columns by capping them at a more reasonable value.

        Parameters:
            dataframe (pd.DataFrame): The input DataFrame containing the data to be processed.

        Returns:
            pd.DataFrame: A new DataFrame with outliers replaced by the median in the specified columns.
        """

        no_outliers_df = dataframe.copy()

        columns = ['person_age', 'person_emp_length']

        for column in columns:
            median = no_outliers_df[column].median()
            no_outliers_df.loc[:, column] = no_outliers_df.loc[:, column].apply(lambda x: median if x > 100 else x)

        return no_outliers_df

    def _fill_null_values(self, dataframe: pd.DataFrame) -> pd.DataFrame:

        """
        Fills null values in numerical columns with the median of each column.

        This function creates a copy of the provided DataFrame and iterates over
        its columns. For numerical columns, it replaces null values with the median
        value of that column. String (object) columns are skipped.

        Parameters:
            dataframe (pd.DataFrame): The input DataFrame with potential null values.

        Returns:
            pd.DataFrame: A new DataFrame with null values in numerical columns replaced by the median.
        """

        for column in dataframe.columns:

            if dataframe.loc[:, column].dtype == 'object':
                continue

            else:
                median = dataframe.loc[:, column].median()
                dataframe.loc[:, column] = dataframe.loc[:, column].apply(lambda x: median if pd.isna(x) else x)

        return dataframe


    def clean_df(self) -> pd.DataFrame:

        """
        Cleans the dataset by handling outliers and filling null values.

        This method applies a series of data cleaning steps on the `personal` DataFrame:
        it first removes outliers by replacing them with the median in specified columns,
        then fills null values in numerical columns with their median values. The cleaned
        DataFrame is returned.

        Returns:
            pd.DataFrame: A cleaned DataFrame with outliers handled and null values filled.
        """

        df: pd.DataFrame = self.personal
        df = self._change_outliers(df)
        df = self._fill_null_values(df)

        return df

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


    def get_dummies(self, dataframe: pd.DataFrame):

        """
        Converts specific columns in the DataFrame to categorical and one-hot encoded columns.

        This method processes a copy of the input DataFrame by transforming numerical columns
        into categorical bins based on predefined conditions. It categorizes 'person_age' by age groups,
        'person_income' by income levels, and 'loan_percent_income' by debt-to-income (DTI) levels.
        It then applies one-hot encoding to the specified categorical columns and converts any boolean
        columns to integers.

        Parameters:
            dataframe (pd.DataFrame): The input DataFrame containing columns to be categorized and encoded.

        Returns:
            pd.DataFrame: A new DataFrame with categorized and one-hot encoded columns.
        """

        df = dataframe.copy()

        def categorias(edad):
            if edad < 18:
                return 'menor_de_edad'
            elif 18 <= edad <= 25:
                return 'jovenes'
            elif 26 <= edad <= 35:
                return 'adulto_joven'
            elif 36 <= edad <= 45:
                return 'adulto'
            elif 46 <= edad <= 55:
                return 'adulto_mayor'
            else:
                return 'tercera_edad'

        def cat_ingresos(income):
            if income <= 5000:
                return 'low_income'
            elif 5001 <= income <= 10000:
                return "medium_low_income"
            elif 10001 <= income <= 20000:
                return "medium_income"
            elif 20001 <= income <= 50000:
                return "medium_high_income"
            elif 50001 <= income <= 100000:
                return "high_income"
            else:
                return "super_high_income"

        def cat_dti(percentage):
            if percentage <= .20:
                return 'bajo_endeudamiento'
            elif .21 <= percentage <= .35:
                return 'moderado_endeudamiento'
            elif .36 <= percentage <= .50:
                return 'alto_endeudamiento'
            elif .51 <= percentage <= .70:
                return 'endeudamiento_critico'
            else:
                return 'sobreendeudado'

        df = df.drop(columns="loan_intent")
        df['person_age'] = df['person_age'].apply(categorias)
        df['person_income'] = df['person_income'].apply(cat_ingresos)
        df['loan_percent_income'] = df['loan_percent_income'].apply(cat_dti)
        # 1 cayo en impago, 0 no cayo en impago
        df['cb_person_default_on_file'] = (df['cb_person_default_on_file'] != 'N').astype(int)
        dummy_columns: list = ['person_age', "person_income", 'person_home_ownership', 'loan_grade',
                               'loan_percent_income']
        df = pd.get_dummies(df, columns=dummy_columns)
        for col in df.select_dtypes(include='bool'):
            df[col] = df[col].astype(int)

        return df


pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 20)

if __name__ == "__main__":
    route: str = "../data/data.csv"
    data = Preprocess(route)
    old = data.personal
    new = data.clean_df()
    print(old)
    print(new)
    print(data.distribution_with_changes(old_dataframe=old, new_dataframe=new))
