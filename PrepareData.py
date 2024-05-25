import pandas as pd

class DataPreparer:
    """
    A class used to load, prepare, and optionally save data for training and testing.

    Attributes
    ----------
    path : str
        The path to the data directory.
    mode : str
        The mode of operation, either 'train' or 'test'.
    save_to_excel : bool
        A flag indicating whether to save the prepared data to an Excel file.
    colstonotconsider : list
        A list of columns to be excluded from the data.
    data_home_team : DataFrame
        The home team statistics data.
    data_home_players : DataFrame
        The home players statistics data.
    data_away_team : DataFrame
        The away team statistics data.
    data_away_players : DataFrame
        The away players statistics data.
    data_match : DataFrame
        The match results data.
    data : DataFrame
        The prepared data.

    Methods
    -------
    load_data():
        Loads the data from CSV files depending on the mode attribute.
    remove_columns():
        Removes columns that are not considered from the data.
    rename_columns(data, prefix='', suffix=''):
        Renames columns of the data with a specified prefix or suffix.
    prepare_player_data(data, prefix):
        Prepares player data by calculating sum, max, min, mean, and median values.
    prepare_data():
        Prepares the data by loading it, removing unwanted columns, renaming columns, and merging datasets.
    save_data():
        Saves the prepared data to parquet and optionally to an Excel file.
    """

    def __init__(self, path, train=True, save_to_excel=True, colstonotconsider=None):
        """
        Constructs all the necessary attributes for the DataPreparer object.

        Parameters
        ----------
        path : str
            The path to the data directory.
        mode : str, optional
            The mode of operation, either 'train' or 'test' (default is "train").
        save_to_excel : bool, optional
            A flag indicating whether to save the prepared data to an Excel file (default is True).
        colstonotconsider : list, optional
            A list of columns to be excluded from the data (default is ['TEAM_NAME', 'LEAGUE', 'PLAYER_NAME', 'POSITION']).
        """
        if colstonotconsider is None:
            colstonotconsider = ['TEAM_NAME', 'LEAGUE', 'PLAYER_NAME', 'POSITION']
        self.train = train
        self.path = path
        self.save_to_excel = save_to_excel
        self.colstonotconsider = colstonotconsider
        self.data_home_team = None
        self.data_home_players = None
        self.data_away_team = None
        self.data_away_players = None
        self.data_match = None
        self.data = None
        self.load_data()

    def load_data(self):
        """
        Loads the data from CSV files depending on the mode attribute.
        """
        if self.train:
            data_home_team_path = self.path + "Train_Data/train_home_team_statistics_df.csv"
            data_home_players_path = self.path + "Train_Data/train_home_player_statistics_df.csv"
            data_away_team_path = self.path + "Train_Data/train_away_team_statistics_df.csv"
            data_away_players_path = self.path + "Train_Data/train_away_player_statistics_df.csv"
            data_match_path = self.path + "y_train.csv"
            self.data_match = pd.read_csv(data_match_path)
        else:
            data_home_team_path = self.path + "Test_Data/test_home_team_statistics_df.csv"
            data_home_players_path = self.path + "Test_Data/test_home_player_statistics_df.csv"
            data_away_team_path = self.path + "Test_Data/test_away_team_statistics_df.csv"
            data_away_players_path = self.path + "Test_Data/test_away_player_statistics_df.csv"

        self.data_home_team = pd.read_csv(data_home_team_path)
        self.data_home_players = pd.read_csv(data_home_players_path)
        self.data_away_team = pd.read_csv(data_away_team_path)
        self.data_away_players = pd.read_csv(data_away_players_path)

    def remove_columns(self):
        """
        Removes columns that are not considered from the data.
        """
        for dataframe in [self.data_home_team, self.data_home_players, self.data_away_team, self.data_away_players]:
            for col in self.colstonotconsider:
                if col in dataframe.columns:
                    dataframe.drop(col, axis=1, inplace=True)

    def rename_columns(self, data, prefix='', suffix=''):
        """
        Renames columns of the data with a specified prefix or suffix.

        Parameters
        ----------
        data : DataFrame
            The data whose columns need to be renamed.
        prefix : str, optional
            The prefix to add to the columns (default is '').
        suffix : str, optional
            The suffix to add to the columns (default is '').

        Returns
        -------
        DataFrame
            The data with renamed columns.
        """
        if prefix:
            data.rename(columns={col: prefix + col for col in data.columns if col != 'ID'}, inplace=True)
        if suffix:
            data.rename(columns={col: col + suffix for col in data.columns if col != 'ID'}, inplace=True)
        return data

    def prepare_player_data(self, data, prefix):
        """
        Prepares player data by calculating sum, max, min, mean, and median values.

        Parameters
        ----------
        data : DataFrame
            The player data to be prepared.
        prefix : str
            The prefix to add to the columns.

        Returns
        -------
        DataFrame
            The prepared player data.
        """
        data_sum = data.groupby("ID").sum()
        data_sum = self.rename_columns(data_sum, suffix='_SUM')
        data_max = data.groupby("ID").max()
        data_max = self.rename_columns(data_max, suffix='_MAX')
        data_min = data.groupby("ID").min()
        data_min = self.rename_columns(data_min, suffix='_MIN')
        data_mean = data.groupby("ID").mean()
        data_mean = self.rename_columns(data_mean, suffix='_MEAN')
        data_median = data.groupby("ID").median()
        data_median = self.rename_columns(data_median, suffix='_MEDIAN')
        return pd.concat([data_sum, data_max, data_min, data_mean, data_median], axis=1)

    def prepare_data(self):
        """
        Prepares the data by loading it, removing unwanted columns, renaming columns, and merging datasets.
        """
        self.remove_columns()

        if self.mode == "train":
            self.data_match['results'] = self.data_match.apply(lambda x: 0 if x['HOME_WINS'] > 0 else 1 if x['DRAW'] else 2, axis=1)
            self.data_match = self.data_match.drop(['HOME_WINS', 'DRAW', 'AWAY_WINS'], axis=1)

        self.data_home_team = self.rename_columns(self.data_home_team, prefix='HOME_')
        self.data_away_team = self.rename_columns(self.data_away_team, prefix='AWAY_')
        self.data_home_players = self.rename_columns(self.data_home_players, prefix='HOME_PLAYERS_')
        self.data_away_players = self.rename_columns(self.data_away_players, prefix='AWAY_PLAYERS_')

        data_home_players_prepared = self.prepare_player_data(self.data_home_players, 'HOME_PLAYERS_')
        data_away_players_prepared = self.prepare_player_data(self.data_away_players, 'AWAY_PLAYERS_')

        if self.mode == "train":
            self.data = pd.merge(self.data_match, self.data_home_team, on='ID', how='left')
        else:
            self.data = self.data_home_team

        self.data = pd.merge(self.data, self.data_away_team, on='ID', how='left')
        self.data = pd.merge(self.data, data_home_players_prepared, on='ID', how='left')
        self.data = pd.merge(self.data, data_away_players_prepared, on='ID', how='left')

    def save_data(self):
        """
        Saves the prepared data to parquet and optionally to an Excel file.
        """
        output_prefix = "train" if self.train else "test"
        self.data.to_parquet(f"data/prepared_data_{output_prefix}.parquet")
        print(f"Saving the prepared data to data/prepared_data_{output_prefix}.parquet")

        if self.save_to_excel:
            self.data.to_excel(f"data/prepared_data_{output_prefix}.xlsx")
            print(f"Saving the prepared data to data/prepared_data_{output_prefix}.xlsx")

        print('Data prepared and saved!')

