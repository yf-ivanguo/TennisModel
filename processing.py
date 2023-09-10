import re
import pandas as pd
import numpy as np
import swifter
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from glicko import Glicko

class Processing():
    def __init__(self, tournament_loc_df, iso_df):
        self.tournament_loc_df = tournament_loc_df
        self.iso_df = iso_df
        self.days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        self.tour_level = ['A', 'G', 'D', 'M', 'F']
        self.challenger_level = ['C']
        self.futures_level = ['S', 15, 25, '15', '25']
        self.AVG_DAYS_IN_YEAR = 365.25
        self.AVG_DAYS_IN_MONTH = 30.437
        self.AVG_TENNIS_PLAYER_AGE = 23.6
        self.AVG_TENNIS_PLAYER_IOC = 'CHN'
        self.MOST_FREQ_SURFACE = 'Grass'
        self.REQ_COLS = ['tourney_name', 'surface', 'tourney_level', 'tourney_date', 'winner_id', 'winner_name', 'winner_ioc', 'winner_age', \
            'loser_id', 'loser_name', 'loser_ioc', 'loser_age', 'best_of']
        self.glicko = Glicko()

    """ 
    Public Functions
    """
    # Check if a tournament is not in the csv file
    def check_tourney_missing(self, df):
        new_col_df = df[df['new_col'] == 1]
        missing_tourneys = set()

        for _, row in new_col_df.iterrows():
            if not self.tournament_loc_df['tournament'].isin([row['tourney_name']]).any():
                missing_tourneys.add(row['tourney_name'])
        
        return missing_tourneys
    
    # Read a csv default format
    def read_csv(self, file):
        level = file.split('_')[3]
        file_origin = level if level == 'futures' or level == 'qual' else 'tour' 
        data = pd.read_csv(file, usecols=self.REQ_COLS, dtype={'tourney_level': str})
        data['file_origin'] = file_origin
        return data
    
    # Read multiple csvs in a directory
    def read_multiple_csv(self, files):
        matches = pd.DataFrame()
        for csv_file in files:
                level = csv_file.split('_')[3]
                file_origin = level if level == 'futures' or level == 'qual' else 'tour' 

                temp_df = pd.read_csv(csv_file, usecols=self.REQ_COLS, dtype={'tourney_level': str})
                temp_df['file_origin'] = file_origin
                matches = pd.concat([matches, temp_df], ignore_index=True)
        return matches

    # Cleans new matches
    def clean_matches(self, df, single=False, include_progress_bar=False):
        df = self.__add_datetime_col(df)
        df = self.__add_date_int_col(df)
        if not single:
            df = self.__rename_initial_cols(df)
            df = self.__impute_ages(df)
            df = self.__impute_iocs(df)
            df = self.__impute_surfaces(df)
            df = self.__create_target_helpers(df)
            df = self.__create_target(df)
        df = self.__create_iso_maps(df)
        df = self.__create_encoded_tourneys(df, include_progress_bar)
        df = self.__create_time_feats(df, include_progress_bar)
        df = self.__create_home_adv_feats(df, include_progress_bar)
        df = self.__create_surface_env_feats(df)
        return df

    # Appends matches onto a full-dataset
    def append(self, df, full_df):
        self.__check_na(df)
        full_df['new_col'] = 0
        df['new_col'] = 1
        df = df[df['tourney_date'] >= full_df.tail(1)['tourney_date'].values[0]]

        for idx, row in df.iterrows():
            mask_1 = (full_df['p1_id'] == row['p1_id']) & (full_df['p2_id'] == row['p2_id'])
            mask_2 = (full_df['p2_id'] == row['p1_id']) & (full_df['p1_id'] == row['p2_id'])
            if ((mask_1 | mask_2) & (full_df['tourney_date'] == row['tourney_date'])).any():
                df.drop(idx, inplace=True)
        
        df = self.sort_dataframe(df)
        df = pd.concat([full_df, df], ignore_index=True)
        df.index = range(df.shape[0])
        return df
    
    # Creates rolling features 
    def prepare_rolling_features(self, df, compute_all=False, include_progress_bar=False):
        df = self.glicko.compute_df_glicko(df, compute_all)
        df = self.create_win_loss(df, compute_all, include_progress_bar)
        df = self.create_surface_feats(df, compute_all, include_progress_bar)
        df = self.create_h2h_features(df, compute_all, include_progress_bar)
        df = self.create_tourney_wl_features(df, compute_all, include_progress_bar)
        df = self.create_l2w_features(df, compute_all, include_progress_bar)
        df = self.create_inactive_features(df, compute_all, include_progress_bar)
        df = self.get_player_consecutive(df, compute_all)
        return df
    
    def validate_dataset(self, df):
        return df

    """
    Non-rolling Feature Creation Functions
    """
    # Creates time features
    def __create_time_feats(self, df, include_progress_bar=False):
        try:
            self.__check_missing(df, 'tourney_datetime')
            # Year feature
            df['year'] = df['tourney_datetime'].dt.year

            # Helpers
            df['month'] = df['tourney_datetime'].dt.month
            df['day'] = df['tourney_datetime'].dt.day
            df['day_of_year'] = (df['month']).map(lambda x: np.sum(self.days_in_month[:x-1])) + df['day'] - 1
            angle = 2 * np.pi * df['day_of_year'] / 365

            # Day features
            df['day_cos'] = np.cos(angle)
            df['day_sin'] = np.sin(angle)

            # More helpers
            df['2weeks_ago'] = df.swifter.progress_bar(include_progress_bar).apply(lambda row: self.__get_earlier_date(row['tourney_datetime'], '2 weeks'), axis=1).astype('int32')
            df['semester_ago'] = df.swifter.progress_bar(include_progress_bar).apply(lambda row: self.__get_earlier_date(row['tourney_datetime'], 'semester'), axis=1).astype('int32')
            df['year_ago'] = df.swifter.progress_bar(include_progress_bar).apply(lambda row: self.__get_earlier_date(row['tourney_datetime'], 'year'), axis=1).astype('int32')

            df = df.drop(['month', 'day'], axis=1)
        except Exception as e:
            print(f"Error encountered while creating time feats: {str(e)}")
        finally:
            return df
        
    # Creates home advantage features
    def __create_home_adv_feats(self, df, include_progress_bar=False):
        try:
            cols = ['p1_iso', 'p2_iso', 'tourney_name', 'year']
            self.__check_missing(df, cols)

            df[['p1_home_adv', 'p2_home_adv']] = df.swifter.progress_bar(include_progress_bar).apply(lambda row: self.__get_home_adv(row['p1_iso'], row['p2_iso'], row['tourney_name'], row['year']), axis=1)
        except Exception as e:
            print(f"Error encountered while creating home_adv feats: {str(e)}")
        finally:
            return df

    def __create_surface_env_feats(self, df):
        try:
            self.__check_missing(df, 'surface')

            df['Clay'] = np.where((df['surface'] == 'Clay'), 1, 0)
            df['Hard'] = np.where((df['surface'] == 'Hard'), 1, 0)
            df['Carpet'] = np.where((df['surface'] == 'Carpet'), 1, 0)
            df['Grass'] = np.where((df['surface'] == 'Grass'), 1, 0)
            df = df.drop(['surface'], axis=1)
        except Exception as e:
            print(f"Error encountered while creating surface_env feats: {str(e)}")
        finally:
            return df

    """
    Rolling Feature Creation Functions
    """
    # Creates W/L Features
    def create_win_loss(self, df, compute_all, include_progress_bar=False):
        try:
            cols = ['p1_id', 'p2_id', 'semester_ago', 'year_ago']
            self.__check_missing(df, cols)

            wl_col_names = ['p1_tour_wins_last_sem','p1_tour_losses_last_sem','p1_tour_wins_last_year','p1_tour_losses_last_year','p1_tour_wins_alltime','p1_tour_losses_alltime',
                            'p1_qual/chal_wins_last_sem','p1_qual/chal_losses_last_sem','p1_qual/chal_wins_last_year','p1_qual/chal_losses_last_year','p1_qual/chal_wins_alltime','p1_qual/chal_losses_alltime',
                            'p1_futures_wins_last_sem','p1_futures_losses_last_sem','p1_futures_wins_last_year','p1_futures_losses_last_year','p1_futures_wins_alltime','p1_futures_losses_alltime',
                            'p2_tour_wins_last_sem','p2_tour_losses_last_sem','p2_tour_wins_last_year','p2_tour_losses_last_year','p2_tour_wins_alltime','p2_tour_losses_alltime',
                            'p2_qual/chal_wins_last_sem','p2_qual/chal_losses_last_sem','p2_qual/chal_wins_last_year','p2_qual/chal_losses_last_year','p2_qual/chal_wins_alltime','p2_qual/chal_losses_alltime',
                            'p2_futures_wins_last_sem','p2_futures_losses_last_sem','p2_futures_wins_last_year','p2_futures_losses_last_year','p2_futures_wins_alltime','p2_futures_losses_alltime']
            
            target_df = df[df['new_col'] == 1] if not compute_all else df

            for i in range(0, len(wl_col_names), 2):
                column_split = wl_col_names[i].split('_')
                player = column_split[0]
                level = column_split[1]
                time = column_split[3] if len(column_split) == 4 else column_split[4]

                p_id = 'p1_id' if player == 'p1' else 'p2_id'
                tourney_level = 2 if level == 'tour' else 1 if level == 'qual/chal' else 0
                period = 'semester_ago' if time == 'sem' else 'year_ago' if time == 'year' else 'alltime'

                wl_col = [wl_col_names[i], wl_col_names[i+1]]
                if period == 'alltime':
                    target_df[wl_col] = target_df.swifter.progress_bar(include_progress_bar).apply(lambda row: self.__get_wins_and_losses(df, row[p_id], row.name, tourney_level, True), axis=1)
                else:
                    target_df[wl_col] = target_df.swifter.progress_bar(include_progress_bar).apply(lambda row: self.__get_wins_and_losses(df, row[p_id], row.name, tourney_level, False, row[period]), axis=1)
                    
                if not compute_all:
                    df.loc[target_df.index, wl_col] = target_df[wl_col]

        except Exception as e:
            print(f"Error encountered while creating w/l feats: {str(e)}")
        finally:
            return df
        
    # Create Surface W/L Features
    def create_surface_feats(self, df, compute_all, include_progress_bar=False):
        try:
            cols = ['p1_id', 'p2_id', 'Clay', 'Hard', 'Grass', 'Carpet']
            self.__check_missing(df, cols)

            surface_col_names = ['p1_wins_clay','p1_losses_clay','p1_wins_hard','p1_losses_hard',
                                 'p1_wins_grass','p1_losses_grass','p1_wins_carpet','p1_losses_carpet',
                                 'p2_wins_clay','p2_losses_clay','p2_wins_hard','p2_losses_hard',
                                 'p2_wins_grass','p2_losses_grass','p2_wins_carpet','p2_losses_carpet']
            
            target_df = df[df['new_col'] == 1] if not compute_all else df

            for i in range(0, len(surface_col_names), 2):
                column_split = surface_col_names[i].split('_')
                player = column_split[0]
                surface = column_split[2]

                p_id = 'p1_id' if player == 'p1' else 'p2_id'
                surface_game = surface.capitalize()

                surface_col = [surface_col_names[i], surface_col_names[i+1]]

                target_df[surface_col] = target_df.swifter.progress_bar(include_progress_bar).apply(lambda row: self.__get_surface_games(df, row[p_id], surface_game, row.name), axis=1)
                    
                if not compute_all:
                    df.loc[target_df.index, surface_col] = target_df[surface_col]
        except Exception as e:
            print(f"Error encountered while creating surface w/l feats: {str(e)}")
        finally:
            return df
        
    # Create head-2-head features
    def create_h2h_features(self, df, compute_all, include_progress_bar=False):
        try:
            cols = ['p1_id', 'p2_id']
            self.__check_missing(df, cols)

            target_df = df[df['new_col'] == 1] if not compute_all else df

            target_df[['p1_h2h_wins', 'p2_h2h_wins']] = target_df.swifter.progress_bar(include_progress_bar).apply(lambda row: self.__get_h2h_wins(df, row['p1_id'], row['p2_id'], row.name), axis=1)
            
            if not compute_all:
                df.loc[target_df.index, ['p1_h2h_wins', 'p2_h2h_wins']] = target_df[['p1_h2h_wins', 'p2_h2h_wins']]
        except Exception as e:
            print(f"Error encountered while creating h2h feats: {str(e)}")
        finally:
            return df
        
    # Create tournament W/L features
    def create_tourney_wl_features(self, df, compute_all, include_progress_bar=False):
        try:
            cols = ['p1_id', 'p2_id']
            self.__check_missing(df, cols)
            
            target_df = df[df['new_col'] == 1] if not compute_all else df

            target_df['tourney_name_mod'] = target_df['tourney_name'].swifter.progress_bar(include_progress_bar).apply(self.__parse_tourney)
            target_df[['p1_tourney_wins', 'p1_tourney_losses']] = target_df.swifter.progress_bar(include_progress_bar).apply(lambda row: self.__get_tourney_games(df, row['p1_id'], row['tourney_name_mod'], row.name), axis=1)
            target_df[['p2_tourney_wins', 'p2_tourney_losses']] = target_df.swifter.progress_bar(include_progress_bar).apply(lambda row: self.__get_tourney_games(df, row['p2_id'], row['tourney_name_mod'], row.name), axis=1)

            if not compute_all:
                df.loc[target_df.index, ['p1_tourney_wins', 'p1_tourney_losses']] = target_df[['p1_tourney_wins', 'p1_tourney_losses']]
                df.loc[target_df.index, ['p2_tourney_wins', 'p2_tourney_losses']] = target_df[['p2_tourney_wins', 'p2_tourney_losses']]
        except Exception as e:
            print(f"Error encountered while creating tourney_w/l feats: {str(e)}")
        finally:
            return df
        
    # Create number of games in the last 2 weeks features
    def create_l2w_features(self, df, compute_all, include_progress_bar=False):
        try:
            cols = ['p1_id', 'p2_id', '2weeks_ago']
            self.__check_missing(df, cols)

            target_df = df[df['new_col'] == 1] if not compute_all else df

            target_df['p1_last2w_games'] = target_df.swifter.progress_bar(include_progress_bar).apply(lambda row: self.__get_player_last2w_count(df, row['p1_id'], row['2weeks_ago'], row.name), axis=1)
            target_df['p2_last2w_games'] = target_df.swifter.progress_bar(include_progress_bar).apply(lambda row: self.__get_player_last2w_count(df, row['p2_id'], row['2weeks_ago'], row.name), axis=1)

            if not compute_all:
                df.loc[target_df.index, ['p1_last2w_games']] = target_df[['p1_last2w_games']]
                df.loc[target_df.index, ['p2_last2w_games']] = target_df[['p2_last2w_games']]
        except Exception as e:
            print(f"Error encountered while creating l2w feats: {str(e)}")
        finally:
            return df
        
    # Create number of weeks inactive features
    def create_inactive_features(self, df, compute_all, include_progress_bar=False):
        try:
            cols = ['p1_id', 'p2_id', 'day_of_year', 'year']
            self.__check_missing(df, cols)

            target_df = df[df['new_col'] == 1] if not compute_all else df

            target_df['p1_weeks_inactive'] = target_df.swifter.progress_bar(include_progress_bar).apply(lambda row: self.__get_player_inactivity(df, row['p1_id'], row['day_of_year'], row['year'], row.name), axis=1)
            target_df['p2_weeks_inactive'] = target_df.swifter.progress_bar(include_progress_bar).apply(lambda row: self.__get_player_inactivity(df, row['p2_id'], row['day_of_year'], row['year'], row.name), axis=1)

            if not compute_all:
                df.loc[target_df.index, ['p1_weeks_inactive']] = target_df[['p1_weeks_inactive']]
                df.loc[target_df.index, ['p2_weeks_inactive']] = target_df[['p2_weeks_inactive']]
        except Exception as e:
            print(f"Error encountered while creating inactive feats: {str(e)}")
        finally:
            return df

    """
    Target Creation Functions
    """
    # Creates target helpers
    def __create_target_helpers(self, df):
        try:
            cols = ['p1_id', 'p1_name']
            self.__check_missing(df, cols)
            df['winner_id'] = df['p1_id']
            df['winner_name'] = df['p1_name']
        except Exception as e:
            print(f"Error encountered while creating target feat helpers: {str(e)}")
        finally:
            return df
    
    # Creates target
    def __create_target(self, df):
        try:
            cols = ['p1_name', 'p2_name', 'p1_ioc', 'p2_ioc', 'p1_age', 'p2_age', 'p1_id', 'p2_id']
            self.__check_missing(df, cols)
            random_values = np.random.choice([0, 1], size=df.shape[0])
            df['winner'] = random_values
            df.loc[df['winner'] == 1, ['p1_name', 'p2_name']] = df.loc[df['winner'] == 1, ['p2_name', 'p1_name']].values
            df.loc[df['winner'] == 1, ['p1_ioc', 'p2_ioc']] = df.loc[df['winner'] == 1, ['p2_ioc', 'p1_ioc']].values
            df.loc[df['winner'] == 1, ['p1_age', 'p2_age']] = df.loc[df['winner'] == 1, ['p2_age', 'p1_age']].values
            df.loc[df['winner'] == 1, ['p1_id', 'p2_id']] = df.loc[df['winner'] == 1, ['p2_id', 'p1_id']].values
        except Exception as e:
            print(f"Error encountered while creating target feat: {str(e)}")
        finally:
            return df
        
    """
    Helper Functions
    """
    # Checks to see if a required column is missing
    def __check_missing(self, df, cols):
        if type(cols) != list:
            if cols not in df:
                raise Exception(f"Error: {cols} column does not exist")
        else:
            for col in cols:
                if col not in df:
                    raise Exception(f"Error: {col} column does not exist")
                
    def __check_na(self, df, col=False):
        na_columns = df.isna().any()
        columns_with_na = na_columns[na_columns].index.tolist()
        if columns_with_na:
            if col:
                if col in columns_with_na:
                    raise Exception(f'Error: {col} has NA values.')
            else:    
                raise Exception(f'Error: NA values detected inside column(s) {columns_with_na}')
                
    """
    Feature Helper Functions
    """
    # For use with a cnn
    def create_symmetric_matches(self, df):
        swapped_df = pd.DataFrame(columns=df.columns)
        
        swapped_df = df.copy()
        p1_columns = [col for col in swapped_df.columns if col.startswith("p1")]
        p2_columns = [col for col in swapped_df.columns if col.startswith("p2")]

        for i in range(len(p1_columns)):
            swapped_df[p1_columns[i]], swapped_df[p2_columns[i]] = df[p2_columns[i]], df[p1_columns[i]]

        if 'winner' in df.columns:
            swapped_df['winner'] = np.where((df['winner'] == 0), 1, 0)
        
        return swapped_df

    # Adds a datetime column 
    def __add_datetime_col(self, df):
        try:
            self.__check_missing(df, 'tourney_date')
            # Adds in datetime format (used in time feature creation)
            df['tourney_datetime'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d')
        except Exception as e:
            print(f"Error encountered while searching for columns: {str(e)}")
        finally:
            return df
        
    # Adds a date column as int
    def __add_date_int_col(self, df):
        try:
            self.__check_missing(df, 'tourney_date')
            df['tourney_date_int'] = df['tourney_date'].astype('int32')
        except Exception as e:
            print(f"Error encountered while searching for columns: {str(e)}")
        finally:
            return df

    def __parse_tourney(self, input_string):
        # Remove substrings starting with a letter followed by a number, including additional case
        parsed_string = re.sub(r'\b(?!\w\')\w\d+.*?(?=\s|$)', '', input_string)

        # Remove substrings with only one number
        parsed_string = re.sub(r'\b(?<!\w\')\d\b(?=\s|$)', '', parsed_string)

        # Remove substrings that start and end with parentheses
        parsed_string = re.sub(r'\([^()]+\)', '', parsed_string)
        
        # Check if "Davis Cup" is present in the parsed string
        if "Davis Cup" in parsed_string:
            # Remove every substring after encountering "Davis Cup"
            parsed_string = re.sub(r'\bDavis Cup\b.*', 'Davis Cup', parsed_string)
        else:
            # If "Davis Cup" is not present, keep the parsed string as is
            parsed_string = parsed_string.strip()
        
        return parsed_string
    
    def sort_dataframe(self, df):
        df_arr = []
        dates = sorted(list(df['tourney_date'].unique()))

        for date in dates:
            date_df = df[df['tourney_date'] == date]
            sorted_date_df = []

            # Futures level
            future_level_df = date_df[date_df['tourney_level_cond'] == 0]
            if not future_level_df.empty:
                future_level_tourneys = list(future_level_df['tourney_name'].unique())
                future_df_arr = []
                for tourney in future_level_tourneys:
                    future_level_tourney = future_level_df[future_level_df['tourney_name'] == tourney]
                    future_level_tourney_sorted = [future_level_tourney[future_level_tourney['file_origin'] == 'futures'].loc[::-1],
                                                future_level_tourney[future_level_tourney['file_origin'] == 'qual'].loc[::-1],
                                                future_level_tourney[future_level_tourney['file_origin'] == 'tour'].loc[::-1]]
                    future_df_arr.append(pd.concat([df for df in future_level_tourney_sorted], ignore_index=True))
                future_df = pd.concat([df for df in future_df_arr], ignore_index=True)
                if len(future_df) != len(future_level_df):
                    raise Exception(f'future_df is not the same size as future_level_df at date {date}')
                sorted_date_df.append(future_df)

            # Challenger level
            chal_level_df = date_df[date_df['tourney_level_cond'] == 1]
            if not chal_level_df.empty:
                chal_level_tourneys = list(chal_level_df['tourney_name'].unique())
                chal_df_arr = []
                for tourney in chal_level_tourneys:
                    chal_level_tourney = chal_level_df[chal_level_df['tourney_name'] == tourney]
                    chal_level_tourney_sorted = [chal_level_tourney[chal_level_tourney['file_origin'] == 'futures'].loc[::-1],
                                                chal_level_tourney[chal_level_tourney['file_origin'] == 'qual'].loc[::-1],
                                                chal_level_tourney[chal_level_tourney['file_origin'] == 'tour'].loc[::-1]]
                    chal_df_arr.append(pd.concat([df for df in chal_level_tourney_sorted], ignore_index=True))
                chal_df = pd.concat([df for df in chal_df_arr], ignore_index=True)
                if len(chal_df) != len(chal_level_df):
                    raise Exception(f'chal_df is not the same size as chal_level_df at date {date}')
                sorted_date_df.append(chal_df)

            # Tour level
            tour_level_df = date_df[date_df['tourney_level_cond'] == 2]
            if not tour_level_df.empty:
                tour_level_tourneys = list(tour_level_df['tourney_name'].unique())
                tour_df_arr = []
                for tourney in tour_level_tourneys:
                    tour_level_tourney = tour_level_df[tour_level_df['tourney_name'] == tourney]
                    tour_level_tourney_sorted = [tour_level_tourney[tour_level_tourney['file_origin'] == 'futures'].loc[::-1],
                                                tour_level_tourney[tour_level_tourney['file_origin'] == 'qual'].loc[::-1],
                                                tour_level_tourney[tour_level_tourney['file_origin'] == 'tour'].loc[::-1]]
                    tour_df_arr.append(pd.concat([df for df in tour_level_tourney_sorted], ignore_index=True))
                tour_df = pd.concat([df for df in tour_df_arr], ignore_index=True)
                if len(tour_df) != len(tour_level_df):
                    raise Exception(f'tour_df is not the same size as tour_level_df at date {date}')
                sorted_date_df.append(tour_df)

            df_arr.append(pd.concat([df for df in sorted_date_df], ignore_index=True))

        df = pd.concat([df for df in df_arr], ignore_index=True)
        return df

    # Renames initial columns 
    def __rename_initial_cols(self, df):
        try:
            # Renames all columns from winner and loser to p1 and p2
            cols = ['winner_name', 'loser_name', 'winner_ioc', 'loser_ioc', 'winner_age', 'loser_age', 'winner_id', 'loser_id']
            self.__check_missing(df, cols)
            df = df.rename(columns = {'winner_name' : 'p1_name', 'loser_name' : 'p2_name'})
            df = df.rename(columns = {'winner_ioc' : 'p1_ioc', 'loser_ioc' : 'p2_ioc'})
            df = df.rename(columns = {'winner_age' : 'p1_age', 'loser_age' : 'p2_age'})
            df = df.rename(columns = {'winner_id' : 'p1_id', 'loser_id' : 'p2_id'})
        except Exception as e:
            print(f"Error encountered while searching for columns: {str(e)}")
        finally:
            return df
        
     # Imputes missing ages
    def __impute_ages(self, df):
        try:
            cols = ['p1_age', 'p2_age']
            self.__check_missing(df, cols)

            # Impute with the average tennis player age
            df['p1_age'] = df['p1_age'].fillna(self.AVG_TENNIS_PLAYER_AGE)
            df['p2_age'] = df['p2_age'].fillna(self.AVG_TENNIS_PLAYER_AGE)
        except Exception as e:
            print(f"Error encountered while searching for columns: {str(e)}")
        finally:
            return df
        
    # Imputes missing IOCs
    def __impute_iocs(self, df):
        try:
            cols = ['p1_ioc', 'p2_ioc']
            self.__check_missing(df, cols)

            # Impute with the average tennis player iso
            df['p1_ioc'] = df['p1_ioc'].fillna(self.AVG_TENNIS_PLAYER_IOC)
            df['p2_ioc'] = df['p2_ioc'].fillna(self.AVG_TENNIS_PLAYER_IOC)
        except Exception as e:
            print(f"Error encountered while searching for columns: {str(e)}")
        finally:
            return df
        
    def __impute_surfaces(self, df):
        try:
            self.__check_missing(df, 'surface')

            # Impute with the average tennis player iso
            df['surface'] = df['surface'].fillna(self.MOST_FREQ_SURFACE)
        except Exception as e:
            print(f"Error encountered while searching for columns: {str(e)}")
        finally:
            return df
        
    # Converts IOC to ISO
    def __create_iso_maps(self, df):
        try:
            cols = ['p1_ioc', 'p2_ioc']
            self.__check_missing(df, cols)

            ioc_iso_mapping = {
                k: v
                for k, v in self.iso_df.set_index('ioc')['alpha2'].items()
                if pd.notna(k) and pd.notna(v)
            }

            alpha3_mapping = {
                k: v
                for k, v in self.iso_df.set_index('alpha3')['alpha2'].items()
                if pd.notna(k) and pd.notna(v)
            }

            # Merges mapping for alpha3 and ioc to alpha2
            merged_mapping = {**ioc_iso_mapping, **alpha3_mapping}
            df['p1_iso'] = df['p1_ioc'].map(merged_mapping).fillna(df['p1_ioc'])
            df['p2_iso'] = df['p2_ioc'].map(merged_mapping).fillna(df['p2_ioc'])
            df = df.drop(['p1_ioc', 'p2_ioc'], axis=1)
        except Exception as e:
            print(f"Error encountered while searching for columns: {str(e)}")
        finally:
            return df
        
    # Creates numerically encoded tourney levels
    def __create_encoded_tourneys(self, df, include_progress_bar=False):
        try:
            self.__check_missing(df, 'tourney_level')
            # Gives 0 to futures level, 1 to challenger level, 2 to tour level matches
            df['tourney_level_cond'] = df['tourney_level'].swifter.progress_bar(include_progress_bar).apply(lambda x: 2 if x in self.tour_level else (1 if x in self.challenger_level else 0)).astype('int16')
        except Exception as e:
            print(f"Error encountered while searching for columns: {str(e)}")
        finally:
            return df

    # Get the location of a tournament in ISO format
    def __get_tournament_loc(self, tournament, year):
        if tournament == "Laver Cup" or tournament == "Tour Finals":
            tournament = tournament + " " + str(year)
        tourneys = self.tournament_loc_df[self.tournament_loc_df['tournament'] == tournament]
        if not tourneys.empty:
            return tourneys.iloc[0]['iso_code']

    def __get_earlier_date(self, date, period):
        if period == '2 weeks':
            return int((date - timedelta(weeks=2)).strftime('%Y%m%d'))
        elif period == 'semester':
            return int((date - relativedelta(months=6)).strftime('%Y%m%d'))
        elif period == 'year':
            return int((date - relativedelta(years=1)).strftime('%Y%m%d'))
    
    def __get_wins_and_losses(self, df, p_id, index, tourney_level, is_all_time, period_start=0):
        all_prev_games = df.iloc[:index]
        if not all_prev_games.empty:
            p1_id_vals = all_prev_games.p1_id.values
            p2_id_vals = all_prev_games.p2_id.values
            tourney_level_vals = all_prev_games.tourney_level_cond.values
            tourney_date_vals = all_prev_games.tourney_date_int.values
            if is_all_time:
                prev_games = all_prev_games[((p1_id_vals == p_id) | (p2_id_vals == p_id)) & (tourney_level_vals == tourney_level)]
            else:
                prev_games = all_prev_games[((p1_id_vals == p_id) | (p2_id_vals == p_id)) & (tourney_level_vals == tourney_level) & (tourney_date_vals >= period_start)]
            if not prev_games.empty:
                prev_games_rows = len(prev_games)
                p_wins = (prev_games['winner_id'] == p_id).sum()
                p_losses = prev_games_rows - p_wins
                return pd.Series([p_wins, p_losses])
            return pd.Series([0, 0])
        return pd.Series([0, 0])

    def __get_surface_games(self, df, p_id, surface, index):
        all_prev_games = df.iloc[:index]
        if not all_prev_games.empty:
            p1_id_vals = all_prev_games.p1_id.values
            p2_id_vals = all_prev_games.p2_id.values

            if surface == 'Clay':
                surface_id_vals = all_prev_games.Clay.values 
            elif surface == 'Hard':
                surface_id_vals = all_prev_games.Hard.values 
            elif surface == 'Grass':
                surface_id_vals = all_prev_games.Grass.values 
            else:
                surface_id_vals = all_prev_games.Carpet.values 

            prev_games = all_prev_games[((p1_id_vals == p_id) | (p2_id_vals == p_id)) & (surface_id_vals == 1)]
            if not prev_games.empty:
                prev_games_rows = len(prev_games)
                p_wins = (prev_games['winner_id'] == p_id).sum()
                p_losses = prev_games_rows - p_wins
                return pd.Series([p_wins, p_losses])
            return pd.Series([0, 0])
        return pd.Series([0, 0])

    def __get_home_adv(self, p1_iso, p2_iso, tourney_name, year):
        p1_at_home = int(p1_iso == self.__get_tournament_loc(tourney_name, year))
        p2_at_home = int(p2_iso == self.__get_tournament_loc(tourney_name, year))
        return pd.Series([p1_at_home, p2_at_home])

    def __get_h2h_wins(self, df, p1_id, p2_id, index):
        all_prev_games = df.iloc[:index]
        if not all_prev_games.empty:
            p1_id_vals = all_prev_games.p1_id.values
            p2_id_vals = all_prev_games.p2_id.values

            prev_games = all_prev_games[((p1_id_vals == p1_id) | (p2_id_vals == p1_id)) & ((p1_id_vals == p2_id) | (p2_id_vals == p2_id))]
            if not prev_games.empty:
                prev_games_rows = len(prev_games)
                p1_h2h_wins = (prev_games['winner_id'] == p1_id).sum()
                p2_h2h_wins = prev_games_rows - p1_h2h_wins
                return pd.Series([p1_h2h_wins, p2_h2h_wins])
            return pd.Series([0, 0])
        return pd.Series([0, 0])

    def __get_tourney_games(self, df, p_id, tourney_name, index):
        all_prev_games = df.iloc[:index]
        if not all_prev_games.empty:
            p1_id_vals = all_prev_games.p1_id.values
            p2_id_vals = all_prev_games.p2_id.values
            tourney_name_vals = all_prev_games.tourney_name_mod.values

            prev_games = all_prev_games[((p1_id_vals == p_id) | (p2_id_vals == p_id)) & (tourney_name_vals == tourney_name)]
            if not prev_games.empty:
                prev_games_rows = len(prev_games)
                p_tourney_wins = (prev_games['winner_id'] == p_id).sum()
                p_tourney_losses = prev_games_rows - p_tourney_wins
                return pd.Series([p_tourney_wins, p_tourney_losses])
            return pd.Series([0, 0])
        return pd.Series([0, 0])
        
    def __get_player_last2w_count(self, df, p_id, date, index):
        all_prev_games = df.iloc[:index]
        if not all_prev_games.empty:
            p1_id_vals = all_prev_games.p1_id.values
            p2_id_vals = all_prev_games.p2_id.values
            tourney_date_vals = all_prev_games.tourney_date_int.values
            
            prev_games = all_prev_games[((p1_id_vals == p_id) | (p2_id_vals == p_id)) & (tourney_date_vals >= date)]
            if not prev_games.empty:
                return prev_games.shape[0]
            else:
                return 0
        return 0
        
    def __get_player_inactivity(self, df, p_id, day_of_year, year, index):
        all_prev_games = df.iloc[:index]
        if not all_prev_games.empty:
            p1_id_vals = all_prev_games.p1_id.values
            p2_id_vals = all_prev_games.p2_id.values
            
            prev_games = all_prev_games[(p1_id_vals == p_id) | (p2_id_vals == p_id)]
            if not prev_games.empty:
                prev_row = prev_games.iloc[-1]
                # Since tourney_date is the start of a tournament and games that follow may be in the following days,
                # Weeks inactive should be positive when a tournament has not occurred in the previous week
                return (((year - prev_row['year']) * 52) + ((day_of_year - prev_row['day_of_year']) // 8))
            return 0
        return 0
    
    # Compute consecutive W/L
    def get_player_consecutive(self, df, compute_all):
        target_df = df if compute_all else df[df['new_col'] == 1]
        target_df['p1_cwins'] = 0
        target_df['p2_cwins'] = 0
        target_df['p1_closses'] = 0
        target_df['p2_closses'] = 0

        for index, row in target_df.iterrows():
            all_prev_games = df.iloc[:index]
            if not all_prev_games.empty:
                p1_id_vals = all_prev_games.p1_id.values
                p2_id_vals = all_prev_games.p2_id.values
                
                p1_prev_games = all_prev_games[(p1_id_vals == row['p1_id']) | (p2_id_vals == row['p1_id'])]
                if not p1_prev_games.empty:
                    prev_row = p1_prev_games.iloc[-1]
                    if row['p1_id'] == prev_row['winner_id']:
                        df.at[index, 'p1_cwins'] = prev_row['p1_cwins'] + 1 if prev_row['p1_id'] == row['p1_id'] else prev_row['p2_cwins'] + 1
                        df.at[index, 'p1_closses'] = 0
                    else:
                        df.at[index, 'p1_cwins'] = 0
                        df.at[index, 'p1_closses'] = prev_row['p1_closses'] + 1 if prev_row['p1_id'] == row['p1_id'] else prev_row['p2_closses'] + 1
                else:
                    df.at[index, 'p1_cwins'] = 0
                    df.at[index, 'p1_closses'] = 0

                p2_prev_games = all_prev_games[(p1_id_vals == row['p2_id']) | (p2_id_vals == row['p2_id'])]
                if not p2_prev_games.empty:
                    prev_row = p2_prev_games.iloc[-1]
                    if row['p2_id'] == prev_row['winner_id']:
                        df.at[index, 'p2_cwins'] = prev_row['p2_cwins'] + 1 if prev_row['p2_id'] == row['p2_id'] else prev_row['p1_cwins'] + 1
                        df.at[index, 'p2_closses'] = 0
                    else:
                        df.at[index, 'p2_cwins'] = 0
                        df.at[index, 'p2_closses'] = prev_row['p2_closses'] + 1 if prev_row['p2_id'] == row['p2_id'] else prev_row['p1_closses'] + 1
                else:
                    df.at[index, 'p2_cwins'] = 0
                    df.at[index, 'p2_closses'] = 0
            else:
                df.at[index, 'p1_cwins'] = 0
                df.at[index, 'p2_cwins'] = 0
                df.at[index, 'p1_closses'] = 0
                df.at[index, 'p2_closses'] = 0
        return df
    
    # def __validate_WL(self, df):
    #     cols = ['p1_tour_wins_last_sem','p1_tour_losses_last_sem','p1_tour_wins_last_year','p1_tour_losses_last_year','p1_tour_wins_alltime','p1_tour_losses_alltime',
    #             'p1_qual/chal_wins_last_sem','p1_qual/chal_losses_last_sem','p1_qual/chal_wins_last_year','p1_qual/chal_losses_last_year','p1_qual/chal_wins_alltime','p1_qual/chal_losses_alltime',
    #             'p1_futures_wins_last_sem','p1_futures_losses_last_sem','p1_futures_wins_last_year','p1_futures_losses_last_year','p1_futures_wins_alltime','p1_futures_losses_alltime',
    #             'p2_tour_wins_last_sem','p2_tour_losses_last_sem','p2_tour_wins_last_year','p2_tour_losses_last_year','p2_tour_wins_alltime','p2_tour_losses_alltime',
    #             'p2_qual/chal_wins_last_sem','p2_qual/chal_losses_last_sem','p2_qual/chal_wins_last_year','p2_qual/chal_losses_last_year','p2_qual/chal_wins_alltime','p2_qual/chal_losses_alltime',
    #             'p2_futures_wins_last_sem','p2_futures_losses_last_sem','p2_futures_wins_last_year','p2_futures_losses_last_year','p2_futures_wins_alltime','p2_futures_losses_alltime']
        
    #     self.__check_missing(df, cols)

    #     # Key should be player_id, value should have 18 values
    #     player_dict = {}

    #     def check_wl(p1_id, p2_id, winner, is_all_time, period_start=0):
    #         if p1_id in player_dict:

    #         else:
    #             time = {'sem' : 0, 'year' : 0, 'alltime' : 0}
    #             result = {'wins' : time, 'losses' : time}
    #             player_dict[p1_id] = {
    #                 'tour' : result,
    #                 'qual' : result,
    #                 'futures' : result
    #             }