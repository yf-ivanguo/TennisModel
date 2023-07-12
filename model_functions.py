import math, re, csv
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import numexpr as ne
from tqdm import tqdm
import statistics
from glicko import Glicko
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score

class Processing():
    def __init__(self, df, tournament_loc_df, iso_df):
        self.df = df
        self.tournament_loc_df = tournament_loc_df
        self.iso_df = iso_df
        self.days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        self.tour_level = ['A', 'G', 'D', 'M', 'F']
        self.challenger_level = ['C']
        self.futures_level = ['S', 15, 25, '15', '25']
        self.AVG_DAYS_IN_YEAR = 365.25
        self.AVG_DAYS_IN_MONTH = 30.437
        self.AVG_TENNIS_PLAYER_AGE = 23.6
        self.glicko = Glicko()

    # Updates the dataset
    def update_df(self, df):
        self.df = df
        return self.df

    # Checks to see if a required column is missing
    def check_missing(self, cols):
        if type(cols) != list:
            if cols not in self.df:
                raise Exception(f"Error: {cols} column does not exist")
        else:
            for col in cols:
                if col not in self.df:
                    raise Exception(f"Error: {col} column does not exist")

    # Add datetime column
    def add_datetime_col(self):
        try:
            self.check_missing('tourney_date')
            self.df['tourney_datetime'] = pd.to_datetime(self.df['tourney_date'], format='%Y%m%d')
        except Exception as e:
            print(f"Error encountered while searching for columns: {str(e)}, moving on...")
        finally:
            return self.df
    
    def add_date_int_col(self):
        try:
            self.check_missing('tourney_date')
            self.df['tourney_date_int'] = self.df['tourney_date'].astype('int32')
        except Exception as e:
            print(f"Error encountered while searching for columns: {str(e)}, moving on...")
        finally:
            return self.df
    
    def rename_initial_cols(self):
        try:
            cols = ['winner_name', 'loser_name', 'winner_ioc', 'loser_ioc', 'winner_age', 'loser_age', 'winner_id', 'loser_id']
            self.check_missing(cols)
            self.df = self.df.rename(columns = {'winner_name' : 'p1_name', 'loser_name' : 'p2_name'})
            self.df = self.df.rename(columns = {'winner_ioc' : 'p1_ioc', 'loser_ioc' : 'p2_ioc'})
            self.df = self.df.rename(columns = {'winner_age' : 'p1_age', 'loser_age' : 'p2_age'})
            self.df = self.df.rename(columns = {'winner_id' : 'p1_id', 'loser_id' : 'p2_id'})
        except Exception as e:
            print(f"Error encountered while searching for columns: {str(e)}, moving on...")
        finally:
            return self.df
    
    def impute_ages(self):
        try:
            cols = ['p1_age', 'p2_age']
            self.check_missing(cols)
            # mean_age = statistics.mean(self.df[cols].mean())
            self.df['p1_age'] = self.df['p1_age'].fillna(self.AVG_TENNIS_PLAYER_AGE)
            self.df['p2_age'] = self.df['p2_age'].fillna(self.AVG_TENNIS_PLAYER_AGE)
        except Exception as e:
            print(f"Error encountered while searching for columns: {str(e)}, moving on...")
        finally:
            return self.df
    
    def create_target_helpers(self):
        try:
            cols = ['p1_id', 'p1_name']
            self.check_missing(cols)
            self.df['winner_id'] = self.df['p1_id']
            self.df['winner_name'] = self.df['p1_name']
        except Exception as e:
            print(f"Error encountered while searching for columns: {str(e)}, moving on...")
        finally:
            return self.df
    
    def create_target(self):
        try:
            cols = ['p1_name', 'p2_name', 'p1_ioc', 'p2_ioc', 'p1_age', 'p2_age', 'p1_id', 'p2_id']
            self.check_missing(cols)
            random_values = np.random.choice([0, 1], size=self.df.shape[0])
            self.df['winner'] = random_values
            self.df.loc[self.df['winner'] == 1, ['p1_name', 'p2_name']] = self.df.loc[self.df['winner'] == 1, ['p2_name', 'p1_name']].values
            self.df.loc[self.df['winner'] == 1, ['p1_ioc', 'p2_ioc']] = self.df.loc[self.df['winner'] == 1, ['p2_ioc', 'p1_ioc']].values
            self.df.loc[self.df['winner'] == 1, ['p1_age', 'p2_age']] = self.df.loc[self.df['winner'] == 1, ['p2_age', 'p1_age']].values
            self.df.loc[self.df['winner'] == 1, ['p1_id', 'p2_id']] = self.df.loc[self.df['winner'] == 1, ['p2_id', 'p1_id']].values
        except Exception as e:
            print(f"Error encountered while searching for columns: {str(e)}, moving on...")
        finally:
            return self.df

    def create_iso_maps(self):
        try:
            cols = ['p1_ioc', 'p2_ioc']
            self.check_missing(cols)

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

            merged_mapping = {**ioc_iso_mapping, **alpha3_mapping}
            self.df['p1_iso'] = self.df['p1_ioc'].map(merged_mapping).fillna(self.df['p1_ioc'])
            self.df['p2_iso'] = self.df['p2_ioc'].map(merged_mapping).fillna(self.df['p2_ioc'])
            self.df = self.df.drop(['p1_ioc', 'p2_ioc'], axis=1)
        except Exception as e:
            print(f"Error encountered while searching for columns: {str(e)}, moving on...")
        finally:
            return self.df
    
    def create_encoded_tourneys(self):
        try:
            self.check_missing('tourney_level')
            self.df['tourney_level_cond'] = self.df['tourney_level'].apply(lambda x: 2 if x in self.tour_level else (1 if x in self.challenger_level else 0)).astype('int16')
        except Exception as e:
            print(f"Error encountered while searching for columns: {str(e)}, moving on...")
        finally:
            return self.df
        
    def create_time_features(self):
        try:
            self.check_missing('tourney_datetime')
            # Year feature
            self.df['year'] = self.df['tourney_datetime'].dt.year

            # Helpers
            self.df['month'] = self.df['tourney_datetime'].dt.month
            self.df['day'] = self.df['tourney_datetime'].dt.day
            self.df['day_of_year'] = (self.df['month']).map(lambda x: np.sum(self.days_in_month[:x-1])) + self.df['day'] - 1
            angle = 2 * np.pi * self.df['day_of_year'] / 365

            # Day features
            self.df['day_cos'] = np.cos(angle)
            self.df['day_sin'] = np.sin(angle)

            self.df['2weeks_ago'] = self.df.apply(lambda row: self.get_earlier_date(row['tourney_datetime'], '2 weeks'), axis=1).astype('int32')
            self.df['semester_ago'] = self.df.apply(lambda row: self.get_earlier_date(row['tourney_datetime'], 'semester'), axis=1).astype('int32')
            self.df['year_ago'] = self.df.apply(lambda row: self.get_earlier_date(row['tourney_datetime'], 'year'), axis=1).astype('int32')

            self.df = self.df.drop(['month', 'day'], axis=1)
        except Exception as e:
            print(f"Error encountered while searching for columns: {str(e)}, moving on...")
        finally:
            return self.df
        
    def clean_new_match(self):
        self.add_datetime_col()
        self.add_date_int_col()
        self.create_iso_maps()
        self.create_encoded_tourneys()
        self.create_time_features()

        # OHE surface features
        self.df['Clay'] = np.where((self.df['surface'] == 'Clay'), 1, 0)
        self.df['Hard'] = np.where((self.df['surface'] == 'Hard'), 1, 0)
        self.df['Carpet'] = np.where((self.df['surface'] == 'Carpet'), 1, 0)
        self.df['Grass'] = np.where((self.df['surface'] == 'Grass'), 1, 0)
        self.df = self.df.drop(['surface'], axis=1)

        return self.df
        
    def prepare_append(self):
        # Add a datetime column
        self.add_datetime_col()

        # Add a date column represented by integers
        self.add_date_int_col()

        # Rename initial columns to remove winner and loser labels
        self.rename_initial_cols()

        # Impute mean age values
        self.impute_ages()

        # Create target helpers
        self.create_target_helpers()

        # Create target
        self.create_target()

        # Convert ioc to iso 
        self.create_iso_maps()

        # Create encoded tourneys
        self.create_encoded_tourneys()

        # Create time features
        self.create_time_features()

        # OHE surface features
        self.df['Clay'] = np.where((self.df['surface'] == 'Clay'), 1, 0)
        self.df['Hard'] = np.where((self.df['surface'] == 'Hard'), 1, 0)
        self.df['Carpet'] = np.where((self.df['surface'] == 'Carpet'), 1, 0)
        self.df['Grass'] = np.where((self.df['surface'] == 'Grass'), 1, 0)
        self.df = self.df.drop(['surface'], axis=1)

        return self.df

    def append_onto(self, full_df):
        full_df['new_col'] = 0
        self.df['new_col'] = 1

        for idx, row in self.df.iterrows():
            mask_1 = (full_df['p1_id'] == row['p1_id']) & (full_df['p2_id'] == row['p2_id'])
            mask_2 = (full_df['p2_id'] == row['p1_id']) & (full_df['p1_id'] == row['p2_id'])
            if ((mask_1 | mask_2) & (full_df['tourney_date'] == row['tourney_date'])).any():
                self.df.drop(idx, inplace=True)
        
        self.df = pd.concat([full_df, self.df], ignore_index=True)
        self.df.index = range(self.df.shape[0])

        # Sort dataset
        self.sort_dataframe()
        return self.df
    
    def create_win_loss(self):
        try:
            cols = ['p1_id', 'p2_id', 'semester_ago', 'year_ago', 'new_col']
            self.check_missing(cols)

            wl_col_names = ['p1_tour_wins_last_sem','p1_tour_losses_last_sem','p1_tour_wins_last_year','p1_tour_losses_last_year','p1_tour_wins_alltime','p1_tour_losses_alltime',
                            'p1_qual/chal_wins_last_sem','p1_qual/chal_losses_last_sem','p1_qual/chal_wins_last_year','p1_qual/chal_losses_last_year','p1_qual/chal_wins_alltime','p1_qual/chal_losses_alltime',
                            'p1_futures_wins_last_sem','p1_futures_losses_last_sem','p1_futures_wins_last_year','p1_futures_losses_last_year','p1_futures_wins_alltime','p1_futures_losses_alltime',
                            'p2_tour_wins_last_sem','p2_tour_losses_last_sem','p2_tour_wins_last_year','p2_tour_losses_last_year','p2_tour_wins_alltime','p2_tour_losses_alltime',
                            'p2_qual/chal_wins_last_sem','p2_qual/chal_losses_last_sem','p2_qual/chal_wins_last_year','p2_qual/chal_losses_last_year','p2_qual/chal_wins_alltime','p2_qual/chal_losses_alltime',
                            'p2_futures_wins_last_sem','p2_futures_losses_last_sem','p2_futures_wins_last_year','p2_futures_losses_last_year','p2_futures_wins_alltime','p2_futures_losses_alltime']
            
            new_col_df = self.df[self.df['new_col'] == 1]

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
                    new_col_df[wl_col] = new_col_df.apply(lambda row: self.get_wins_and_losses(row[p_id], row.name, tourney_level, True), axis=1)
                else:
                    new_col_df[wl_col] = new_col_df.apply(lambda row: self.get_wins_and_losses(row[p_id], row.name, tourney_level, False, row[period]), axis=1)
                    
                self.df.loc[new_col_df.index, wl_col] = new_col_df[wl_col]
        except Exception as e:
            print(f"Error encountered while searching for columns: {str(e)}, moving on...")
        finally:
            return self.df
    
    def create_surface_feats(self):
        try:
            cols = ['p1_id', 'p2_id', 'Clay', 'Hard', 'Grass', 'Carpet']
            self.check_missing(cols)

            surface_col_names = ['p1_wins_clay','p1_losses_clay','p1_wins_hard','p1_losses_hard',
                                 'p1_wins_grass','p1_losses_grass','p1_wins_carpet','p1_losses_carpet',
                                 'p2_wins_clay','p2_losses_clay','p2_wins_hard','p2_losses_hard',
                                 'p2_wins_grass','p2_losses_grass','p2_wins_carpet','p2_losses_carpet']
            
            new_col_df = self.df[self.df['new_col'] == 1]

            for i in range(0, len(surface_col_names), 2):
                column_split = surface_col_names[i].split('_')
                player = column_split[0]
                surface = column_split[2]

                p_id = 'p1_id' if player == 'p1' else 'p2_id'
                surface_game = surface.capitalize()

                surface_col = [surface_col_names[i], surface_col_names[i+1]]

                new_col_df[surface_col] = new_col_df.apply(lambda row: self.get_surface_games(row[p_id], surface_game, row.name), axis=1)
                    
                self.df.loc[new_col_df.index, surface_col] = new_col_df[surface_col]
        except Exception as e:
            print(f"Error encountered while searching for columns: {str(e)}, moving on...")
        finally:
            return self.df
        
    def create_home_adv_features(self):
        try:
            cols = ['p1_iso', 'p2_iso', 'tourney_name', 'year']
            self.check_missing(cols)
            new_col_df = self.df[self.df['new_col'] == 1]

            new_col_df[['p1_home_adv', 'p2_home_adv']] = new_col_df.apply(lambda row: self.get_home_adv(row['p1_iso'], row['p2_iso'], row['tourney_name'], row['year']), axis=1)
            self.df.loc[new_col_df.index, ['p1_home_adv', 'p2_home_adv']] = new_col_df[['p1_home_adv', 'p2_home_adv']]
        except Exception as e:
            print(f"Error encountered while searching for columns: {str(e)}, moving on...")
        finally:
            return self.df
    
    def create_h2h_features(self):
        try:
            cols = ['p1_id', 'p2_id']
            self.check_missing(cols)
            new_col_df = self.df[self.df['new_col'] == 1]

            new_col_df[['p1_h2h_wins', 'p2_h2h_wins']] = new_col_df.apply(lambda row: self.get_h2h_wins(row['p1_id'], row['p2_id'], row.name), axis=1)
            self.df.loc[new_col_df.index, ['p1_h2h_wins', 'p2_h2h_wins']] = new_col_df[['p1_h2h_wins', 'p2_h2h_wins']]
        except Exception as e:
            print(f"Error encountered while searching for columns: {str(e)}, moving on...")
        finally:
            return self.df
        
    def create_tourney_wl_features(self):
        try:
            cols = ['p1_id', 'p2_id', 'tourney_name_mod']
            self.check_missing(cols)

            new_col_df = self.df[self.df['new_col'] == 1]
            new_col_df['tourney_name_mod'] = new_col_df['tourney_name'].apply(self.parse_tourney)

            new_col_df[['p1_tourney_wins', 'p1_tourney_losses']] = new_col_df.apply(lambda row: self.get_tourney_games(row['p1_id'], row['tourney_name_mod'], row.name), axis=1)
            new_col_df[['p2_tourney_wins', 'p2_tourney_losses']] = new_col_df.apply(lambda row: self.get_tourney_games(row['p2_id'], row['tourney_name_mod'], row.name), axis=1)

            self.df.loc[new_col_df.index, ['p1_tourney_wins', 'p1_tourney_losses']] = new_col_df[['p1_tourney_wins', 'p1_tourney_losses']]
            self.df.loc[new_col_df.index, ['p2_tourney_wins', 'p2_tourney_losses']] = new_col_df[['p2_tourney_wins', 'p2_tourney_losses']]
        except Exception as e:
            print(f"Error encountered while searching for columns: {str(e)}, moving on...")
        finally:
            return self.df
        
    def create_l2w_features(self):
        try:
            cols = ['p1_id', 'p2_id', '2weeks_ago']
            self.check_missing(cols)

            new_col_df = self.df[self.df['new_col'] == 1]

            new_col_df['p1_last2w_games'] = new_col_df.apply(lambda row: self.get_player_last2w_count(row['p1_id'], row['2weeks_ago'], row.name), axis=1)
            new_col_df['p2_last2w_games'] = new_col_df.apply(lambda row: self.get_player_last2w_count(row['p2_id'], row['2weeks_ago'], row.name), axis=1)

            self.df.loc[new_col_df.index, ['p1_last2w_games']] = new_col_df[['p1_last2w_games']]
            self.df.loc[new_col_df.index, ['p2_last2w_games']] = new_col_df[['p2_last2w_games']]
        except Exception as e:
            print(f"Error encountered while searching for columns: {str(e)}, moving on...")
        finally:
            return self.df
        
    def create_inactive_features(self):
        try:
            cols = ['p1_id', 'p2_id', 'day_of_year', 'year']
            self.check_missing(cols)

            new_col_df = self.df[self.df['new_col'] == 1]

            new_col_df['p1_weeks_inactive'] = new_col_df.apply(lambda row: self.get_player_inactivity(row['p1_id'], row['day_of_year'], row['year'], row.name), axis=1)
            new_col_df['p2_weeks_inactive'] = new_col_df.apply(lambda row: self.get_player_inactivity(row['p2_id'], row['day_of_year'], row['year'], row.name), axis=1)

            self.df.loc[new_col_df.index, ['p1_weeks_inactive']] = new_col_df[['p1_weeks_inactive']]
            self.df.loc[new_col_df.index, ['p2_weeks_inactive']] = new_col_df[['p2_weeks_inactive']]
        except Exception as e:
            print(f"Error encountered while searching for columns: {str(e)}, moving on...")
        finally:
            return self.df
        
    def remove_non_required_cols(self):
        cols = ['p1_age', 'p2_age', 'winner', 'year', 'day_cos', 'day_sin',
                'p1_glicko2_rating', 'p1_glicko2_rd', 'p1_glicko2_vol',
                'p2_glicko2_rating', 'p2_glicko2_rd', 'p2_glicko2_vol',
                'p1_tour_wins_last_sem', 'p1_tour_losses_last_sem',
                'p1_tour_wins_last_year', 'p1_tour_losses_last_year',
                'p1_tour_wins_alltime', 'p1_tour_losses_alltime',
                'p1_qual/chal_wins_last_sem', 'p1_qual/chal_losses_last_sem',
                'p1_qual/chal_wins_last_year', 'p1_qual/chal_losses_last_year',
                'p1_qual/chal_wins_alltime', 'p1_qual/chal_losses_alltime',
                'p1_futures_wins_last_sem', 'p1_futures_losses_last_sem',
                'p1_futures_wins_last_year', 'p1_futures_losses_last_year',
                'p1_futures_wins_alltime', 'p1_futures_losses_alltime',
                'p2_tour_wins_last_sem', 'p2_tour_losses_last_sem',
                'p2_tour_wins_last_year', 'p2_tour_losses_last_year',
                'p2_tour_wins_alltime', 'p2_tour_losses_alltime',
                'p2_qual/chal_wins_last_sem', 'p2_qual/chal_losses_last_sem',
                'p2_qual/chal_wins_last_year', 'p2_qual/chal_losses_last_year',
                'p2_qual/chal_wins_alltime', 'p2_qual/chal_losses_alltime',
                'p2_futures_wins_last_sem', 'p2_futures_losses_last_sem',
                'p2_futures_wins_last_year', 'p2_futures_losses_last_year',
                'p2_futures_wins_alltime', 'p2_futures_losses_alltime', 'Carpet',
                'Clay', 'Grass', 'Hard', 'p1_wins_clay', 'p1_losses_clay',
                'p1_wins_hard', 'p1_losses_hard', 'p1_wins_grass', 'p1_losses_grass',
                'p1_wins_carpet', 'p1_losses_carpet', 'p2_wins_clay', 'p2_losses_clay',
                'p2_wins_hard', 'p2_losses_hard', 'p2_wins_grass', 'p2_losses_grass',
                'p2_wins_carpet', 'p2_losses_carpet', 'p1_home_adv', 'p2_home_adv',
                'p1_h2h_wins', 'p2_h2h_wins', 'p1_last2w_games', 'p2_last2w_games',
                'p1_weeks_inactive', 'p2_weeks_inactive', 'p1_tourney_wins',
                'p1_tourney_losses', 'p2_tourney_wins', 'p2_tourney_losses', 'p1_cwins',
                'p2_cwins', 'p1_closses', 'p2_closses', 'best_of']
        
        self.df = self.df.loc[:, cols]
        return self.df
    
    def remove_env_feats(self):
        cols = ['p1_age', 'p2_age', 'winner',
                'p1_glicko2_rating', 'p1_glicko2_rd', 'p1_glicko2_vol',
                'p2_glicko2_rating', 'p2_glicko2_rd', 'p2_glicko2_vol',
                'p1_tour_wins_last_sem', 'p1_tour_losses_last_sem',
                'p1_tour_wins_last_year', 'p1_tour_losses_last_year',
                'p1_tour_wins_alltime', 'p1_tour_losses_alltime',
                'p1_qual/chal_wins_last_sem', 'p1_qual/chal_losses_last_sem',
                'p1_qual/chal_wins_last_year', 'p1_qual/chal_losses_last_year',
                'p1_qual/chal_wins_alltime', 'p1_qual/chal_losses_alltime',
                'p1_futures_wins_last_sem', 'p1_futures_losses_last_sem',
                'p1_futures_wins_last_year', 'p1_futures_losses_last_year',
                'p1_futures_wins_alltime', 'p1_futures_losses_alltime',
                'p2_tour_wins_last_sem', 'p2_tour_losses_last_sem',
                'p2_tour_wins_last_year', 'p2_tour_losses_last_year',
                'p2_tour_wins_alltime', 'p2_tour_losses_alltime',
                'p2_qual/chal_wins_last_sem', 'p2_qual/chal_losses_last_sem',
                'p2_qual/chal_wins_last_year', 'p2_qual/chal_losses_last_year',
                'p2_qual/chal_wins_alltime', 'p2_qual/chal_losses_alltime',
                'p2_futures_wins_last_sem', 'p2_futures_losses_last_sem',
                'p2_futures_wins_last_year', 'p2_futures_losses_last_year',
                'p2_futures_wins_alltime', 'p2_futures_losses_alltime', 
                'p1_wins_clay', 'p1_losses_clay', 'p1_wins_hard', 'p1_losses_hard', 
                'p1_wins_grass', 'p1_losses_grass',
                'p1_wins_carpet', 'p1_losses_carpet', 'p2_wins_clay', 'p2_losses_clay',
                'p2_wins_hard', 'p2_losses_hard', 'p2_wins_grass', 'p2_losses_grass',
                'p2_wins_carpet', 'p2_losses_carpet', 'p1_home_adv', 'p2_home_adv',
                'p1_h2h_wins', 'p2_h2h_wins', 'p1_last2w_games', 'p2_last2w_games',
                'p1_weeks_inactive', 'p2_weeks_inactive', 'p1_tourney_wins',
                'p1_tourney_losses', 'p2_tourney_wins', 'p2_tourney_losses', 'p1_cwins',
                'p2_cwins', 'p1_closses', 'p2_closses']
        
        self.df = self.df.loc[:, cols]
        return self.df

    def check_tourney_missing(self):
        new_col_df = self.df[self.df['new_col'] == 1]
        missing_tourneys = set()

        for _, row in new_col_df.iterrows():
            if not self.tournament_loc_df['tournament'].isin([row['tourney_name']]).any():
                missing_tourneys.add(row['tourney_name'])
        
        return missing_tourneys
        
    def prepare_rolling_features(self):
        # Compute Glicko scores
        self.glicko.compute_df_glicko(self.df, False)
        
        # Compute W/L
        self.create_win_loss()

        # Compute surface W/L
        self.create_surface_feats()

        # Compute Home Adv features
        self.create_home_adv_features()

        # Compute h2h features
        self.create_h2h_features()

        # Compute Tourney W/L features
        self.create_tourney_wl_features()

        # Compute l2w games
        self.create_l2w_features()

        # Compute inactive weeks
        self.create_inactive_features()

        # Compute consecutive W/L
        self.get_player_consecutive(False)

        # Remove non-required columns
        # self.remove_non_required_cols()

        return self.df

    # Helper Functions
    def get_tournament_loc(self, tournament, year):
        if tournament == "Laver Cup" or tournament == "Tour Finals":
            tournament = tournament + " " + str(year)
        tourneys = self.tournament_loc_df[self.tournament_loc_df['tournament'] == tournament]
        if not tourneys.empty:
            return tourneys.iloc[0]['iso_code']
        
    def get_day_of_year(self, month, day):
        return sum(self.days_in_month[:month-1]) + day - 1

    def parse_tourney(self, input_string):
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

    def get_earlier_date(self, date, period):
        if period == '2 weeks':
            return int((date - timedelta(weeks=2)).strftime('%Y%m%d'))
        elif period == 'semester':
            return int((date - relativedelta(months=6)).strftime('%Y%m%d'))
        elif period == 'year':
            return int((date - relativedelta(years=1)).strftime('%Y%m%d'))
        
    # Feature Transformation Functions
    def sort_dataframe(self):
        df_arr = []
        dates = self.df['tourney_date'].unique()

        for date in dates:
            df_copy = self.df[self.df['tourney_date'] == date].copy()
            df_arr.append(df_copy)

        # Sort the array based on tourney_date and tourney_level
        df_arr.sort(key=lambda x: (x.iloc[0]['tourney_date'], 
                                0 if x.iloc[0]['tourney_level'] in (self.futures_level) else
                                1 if x.iloc[0]['tourney_level'] in (self.challenger_level) else
                                2 if x.iloc[0]['tourney_level'] in (self.tour_level) else 3))

        self.df = pd.concat([sub_df for sub_df in df_arr], ignore_index=True)
        return self.df

    def get_wins_and_losses(self, p_id, index, tourney_level, is_all_time, period_start=0):
        all_prev_games = self.df.iloc[:index]
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

    def get_surface_games(self, p_id, surface, index):
        all_prev_games = self.df.iloc[:index]
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

    def get_home_adv(self, p1_iso, p2_iso, tourney_name, year):
        p1_at_home = int(p1_iso == self.get_tournament_loc(tourney_name, year))
        p2_at_home = int(p2_iso == self.get_tournament_loc(tourney_name, year))
        return pd.Series([p1_at_home, p2_at_home])

    def get_h2h_wins(self, p1_id, p2_id, index):
        all_prev_games = self.df.iloc[:index]
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

    def get_tourney_games(self, p_id, tourney_name, index):
        all_prev_games = self.df.iloc[:index]
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
        
    def get_player_last2w_count(self, p_id, date, index):
        all_prev_games = self.df.iloc[:index]
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
        
    def get_player_inactivity(self, p_id, day_of_year, year, index):
        all_prev_games = self.df.iloc[:index]
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
    def get_player_consecutive(self, compute_all):
        target_df = self.df if compute_all else self.df[self.df['new_col'] == 1]
        target_df['p1_cwins'] = 0
        target_df['p2_cwins'] = 0
        target_df['p1_closses'] = 0
        target_df['p2_closses'] = 0
        # Create a progress bar with the total number of iterations
        progress_bar = tqdm(total=len(target_df), desc='Processing')

        for index, row in target_df.iterrows():
            all_prev_games = self.df.iloc[:index]
            if not all_prev_games.empty:
                p1_id_vals = all_prev_games.p1_id.values
                p2_id_vals = all_prev_games.p2_id.values
                
                p1_prev_games = all_prev_games[(p1_id_vals == row['p1_id']) | (p2_id_vals == row['p1_id'])]
                if not p1_prev_games.empty:
                    prev_row = p1_prev_games.iloc[-1]
                    if row['p1_id'] == prev_row['winner_id']:
                        self.df.at[index, 'p1_cwins'] = prev_row['p1_cwins'] + 1 if prev_row['p1_id'] == row['p1_id'] else prev_row['p2_cwins'] + 1
                        self.df.at[index, 'p1_closses'] = 0
                    else:
                        self.df.at[index, 'p1_cwins'] = 0
                        self.df.at[index, 'p1_closses'] = prev_row['p1_closses'] + 1 if prev_row['p1_id'] == row['p1_id'] else prev_row['p2_closses'] + 1

                p2_prev_games = all_prev_games[(p1_id_vals == row['p2_id']) | (p2_id_vals == row['p2_id'])]
                if not p2_prev_games.empty:
                    prev_row = p2_prev_games.iloc[-1]
                    if row['p2_id'] == prev_row['winner_id']:
                        self.df.at[index, 'p2_cwins'] = prev_row['p2_cwins'] + 1 if prev_row['p2_id'] == row['p2_id'] else prev_row['p1_cwins'] + 1
                        self.df.at[index, 'p2_closses'] = 0
                    else:
                        self.df.at[index, 'p2_cwins'] = 0
                        self.df.at[index, 'p2_closses'] = prev_row['p2_closses'] + 1 if prev_row['p2_id'] == row['p2_id'] else prev_row['p1_closses'] + 1
            else:
                self.df.at[index, 'p1_cwins'] = 0
                self.df.at[index, 'p2_cwins'] = 0
                self.df.at[index, 'p1_closses'] = 0
                self.df.at[index, 'p2_closses'] = 0
            progress_bar.update(1)
        progress_bar.close()

    def get_day_of_year_converted(self, tourney_date):
        month = tourney_date.month
        day = tourney_date.day
        day_of_year = self.get_day_of_year(month, day)
        angle = 2 * math.pi * day_of_year / 365
        day_cos = np.cos(angle)
        day_sin = np.sin(angle)
        return pd.Series([day_cos, day_sin])
    
class TennisModel():
    def __init__(self, df, df_full, tournament_loc_df, iso_df, player_df):
        self.df = pd.read_csv(df, keep_default_na=False)
        self.df.index = range(self.df.shape[0])

        self.df_full = pd.read_csv(df_full, keep_default_na=False)
        self.tournament_loc_df = pd.read_csv(tournament_loc_df)
        self.iso_df = pd.read_csv(iso_df, keep_default_na=False)
        self.player_df = pd.read_csv(player_df, keep_default_na=False)
        self.player_df['full_name'] = self.player_df['name_first'] + ' ' + self.player_df['name_last']

        # Variables
        self.AVG_DAYS_IN_YEAR = 365.25
        self.AVG_DAYS_IN_MONTH = 30.437
        self.AVG_TENNIS_PLAYER_AGE = 23.6
        self.MOST_COMMON_IOC = 'CHN'
        self.BET_VAR = 0.1
        self.REQ_COLS = ['tourney_name', 'surface', 'tourney_level', 'tourney_date', 'winner_id', 'winner_name', 'winner_ioc', 'winner_age', \
            'loser_id', 'loser_name', 'loser_ioc', 'loser_age', 'best_of']

        self.df = self.remove_env_feats(self.df)
        self.optimize_model()
        self.train_model()
    
    def optimize_model(self):
        n_train = math.ceil(self.df.shape[0] * 0.8)
        train_df = self.df[:n_train]
        test_df = self.df[n_train:]
        X_train = train_df.drop(['winner'], axis=1)
        y_train = train_df['winner']
        X_test = test_df.drop(['winner'], axis=1)
        y_test = test_df['winner']

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Hyperparameter Tuning of SGDCClassifier
        sgd = SGDClassifier(loss='log_loss', max_iter=10000)

        sgd_grid = {
            'alpha': np.logspace(-4, 0, num=10),
            'l1_ratio': np.linspace(0, 1, num=10),
        }

        sgd_search = RandomizedSearchCV(sgd, sgd_grid, n_iter=5, cv=10, scoring='accuracy', n_jobs=-1, random_state=1)
        sgd_optimized = sgd_search.fit(X_train_scaled, y_train)

        print(f'Model accuracy: {accuracy_score(sgd_optimized.predict(X_test_scaled), y_test)}')
        self.l1_ratio = sgd_optimized.best_params_['l1_ratio']
        self.alpha = sgd_optimized.best_params_['alpha']
        
    def train_model(self):
        self.model = SGDClassifier(loss='log_loss', max_iter=10000, l1_ratio=self.l1_ratio, alpha=self.alpha)
        X = self.df.drop(['winner'], axis=1)
        y = self.df['winner']
        
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X.values)
        self.model.fit(X_scaled, y)

    def add_matches(self, matches):
        matches = pd.read_csv(matches, usecols=self.REQ_COLS, dtype={'tourney_level': str})
        matches.index = range(matches.shape[0])

        processing = Processing(matches, self.tournament_loc_df, self.iso_df)
        matches = processing.prepare_append()
        matches = processing.append_onto(self.df_full)
        missing_tourneys = processing.check_tourney_missing()
        if len(missing_tourneys) == 0:
            matches = processing.prepare_rolling_features()
            self.df_full = matches
            self.df = self.remove_env_feats(matches)
            self.optimize_model()
            self.train_model()
            return matches
        else:
            print('Missing tourneys detected - check return.')
            return missing_tourneys
    
    def predict(self, vals):
        match = pd.DataFrame([vals])
        match = self.impute_values(match)

        processing = Processing(match, self.tournament_loc_df, self.iso_df)
        combined_df = processing.clean_new_match()
        combined_df = processing.append_onto(self.df_full)
        combined_df = processing.prepare_rolling_features()
        combined_df = self.remove_env_feats(combined_df)

        latest_match = combined_df.tail(1).drop(['winner'], axis=1)
        latest_match_scaled = self.scaler.transform(latest_match.values)

        soft_preds = self.model.predict_proba(latest_match_scaled)

        return soft_preds
    
    def get_bet(self, vals, p1_odds, p2_odds):
        preds = self.predict(vals)

        p1_bet = self.calculate_bet_sizing(preds[0][0], self.get_implied_probability(p1_odds, 'Decimal'))
        p2_bet = self.calculate_bet_sizing(preds[0][1], self.get_implied_probability(p2_odds, 'Decimal'))

        if p1_bet:
            return (p1_bet, vals['p1_name'])
        elif p2_bet:
            return (p2_bet, vals['p2_name'])
        else:
            return None

    def impute_values(self, match):
        p1_name = match['p1_name'][0]
        p2_name = match['p2_name'][0]

        p1 = self.player_df[self.player_df['full_name'] == p1_name]
        p2 = self.player_df[self.player_df['full_name'] == p2_name]
        
        if len(p1) == 0 or len(p2) == 0:
            raise Exception(f'Player does not exist in database.')
        
        p1_id = p1['player_id'].values[0]
        p2_id = p2['player_id'].values[0]
        p1_ioc = p1['ioc'].values[0] if p1['ioc'].values[0] else self.MOST_COMMON_IOC
        p2_ioc = p2['ioc'].values[0] if p2['ioc'].values[0] else self.MOST_COMMON_IOC
        p1_age = self.get_age(p1['dob'].values[0], str(match['tourney_date'].values[0]))
        p2_age = self.get_age(p2['dob'].values[0], str(match['tourney_date'].values[0]))

        match['p1_id'] = [p1_id]
        match['p2_id'] = [p2_id]
        match['p1_ioc'] = [p1_ioc]
        match['p2_ioc'] = [p2_ioc]
        match['p1_age'] = [p1_age]
        match['p2_age'] = [p2_age]

        return match

    def get_age(self, birth_date, current_date):
        if not birth_date:
            return self.AVG_TENNIS_PLAYER_AGE
        birth_year = birth_date[:4]
        birth_month = birth_date[4:5]
        birth_day = birth_date[5:]
        current_year = current_date[:4]
        current_month = current_date[4:5]
        current_day = current_date[5:]

        year_diff_in_days = (int(current_year) - int(birth_year)) * self.AVG_DAYS_IN_YEAR
        month_diff_in_days = (int(current_month) - int(birth_month)) * self.AVG_DAYS_IN_MONTH
        day_diff = (int(current_day) - int(birth_day))
        return (year_diff_in_days + month_diff_in_days + day_diff) / self.AVG_DAYS_IN_YEAR
    
    def remove_env_feats(self, df):
        cols = ['p1_age', 'p2_age', 'winner',
                'p1_glicko2_rating', 'p1_glicko2_rd', 'p1_glicko2_vol',
                'p2_glicko2_rating', 'p2_glicko2_rd', 'p2_glicko2_vol',
                'p1_tour_wins_last_sem', 'p1_tour_losses_last_sem',
                'p1_tour_wins_last_year', 'p1_tour_losses_last_year',
                'p1_tour_wins_alltime', 'p1_tour_losses_alltime',
                'p1_qual/chal_wins_last_sem', 'p1_qual/chal_losses_last_sem',
                'p1_qual/chal_wins_last_year', 'p1_qual/chal_losses_last_year',
                'p1_qual/chal_wins_alltime', 'p1_qual/chal_losses_alltime',
                'p1_futures_wins_last_sem', 'p1_futures_losses_last_sem',
                'p1_futures_wins_last_year', 'p1_futures_losses_last_year',
                'p1_futures_wins_alltime', 'p1_futures_losses_alltime',
                'p2_tour_wins_last_sem', 'p2_tour_losses_last_sem',
                'p2_tour_wins_last_year', 'p2_tour_losses_last_year',
                'p2_tour_wins_alltime', 'p2_tour_losses_alltime',
                'p2_qual/chal_wins_last_sem', 'p2_qual/chal_losses_last_sem',
                'p2_qual/chal_wins_last_year', 'p2_qual/chal_losses_last_year',
                'p2_qual/chal_wins_alltime', 'p2_qual/chal_losses_alltime',
                'p2_futures_wins_last_sem', 'p2_futures_losses_last_sem',
                'p2_futures_wins_last_year', 'p2_futures_losses_last_year',
                'p2_futures_wins_alltime', 'p2_futures_losses_alltime', 
                'p1_wins_clay', 'p1_losses_clay', 'p1_wins_hard', 'p1_losses_hard', 
                'p1_wins_grass', 'p1_losses_grass',
                'p1_wins_carpet', 'p1_losses_carpet', 'p2_wins_clay', 'p2_losses_clay',
                'p2_wins_hard', 'p2_losses_hard', 'p2_wins_grass', 'p2_losses_grass',
                'p2_wins_carpet', 'p2_losses_carpet', 'p1_home_adv', 'p2_home_adv',
                'p1_h2h_wins', 'p2_h2h_wins', 'p1_last2w_games', 'p2_last2w_games',
                'p1_weeks_inactive', 'p2_weeks_inactive', 'p1_tourney_wins',
                'p1_tourney_losses', 'p2_tourney_wins', 'p2_tourney_losses', 'p1_cwins',
                'p2_cwins', 'p1_closses', 'p2_closses']
        
        df = df.loc[:, cols]
        return df

    def calculate_bet_sizing(self, model_prob, bookie_prob):
        sizing = self.calculate_kelly(model_prob, bookie_prob)
        return round(sizing * 10) if sizing > 0 else None

    def calculate_kelly(self, model_prob, bookie_prob):
        if model_prob > 1 or bookie_prob > 1:
            raise Exception('Probability cannot be greater than 1')
        odds = 1 / bookie_prob
        top = (model_prob * odds - 1) ** 3
        bot = (odds - 1) * (((model_prob * odds - 1) ** 2) + ((odds ** 2) * (self.BET_VAR ** 2)))

        return top / bot

    def get_implied_probability(self, odds, format, is_minus=True):
        if format == 'American':
            return odds / (odds + 100) if is_minus else 100 / (odds + 100)
        elif format == 'Decimal':
            return 1 / odds
        else:
            return None