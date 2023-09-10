import math
import pandas as pd
import keras
from processing import Processing
from sklearn.preprocessing import StandardScaler

'''
TennisModel Object.

Initializes the model object as well as 3 datasets:
1. The full dataset, which is a readable version which includes various details
2. The regular dataset, which only has the numerical features that the model depends on
3. The symmetrical regular dataset, which is used by the CNN

I want the model to be initialized with a model file and a dataset.
This model can be refitted and fine-tuned - IDK if I should be saving the new
data or not - I think I should since a tournament is going to be going on multiple days -
I can always just add it to the latest_matches

So I want to provide the model with the baseline dataset and model, then be able to 
update it periodically with ongoing tourney data, then also have functions for saving the datasets -
this should be for when the official dataset is updated.
'''
class TennisModel():
    def __init__(self, ds, ds_full, tournament_loc_ds, iso_ds, player_ds, model_file):
        # Helper datasets
        self.tournament_loc_df = pd.read_csv(tournament_loc_ds)
        self.iso_df = pd.read_csv(iso_ds, keep_default_na=False)
        self.player_df = pd.read_csv(player_ds, keep_default_na=False)
        self.player_df['full_name'] = self.player_df['name_first'] + ' ' + self.player_df['name_last']

        # Processing functions
        self.processing = Processing(self.tournament_loc_df, self.iso_df)
        
        # Dataset formats
        self.df = pd.read_csv(ds, keep_default_na=False)
        self.df_full = pd.read_csv(ds_full, keep_default_na=False)
        self.df_symm = self.create_symmetric_matches(self.df)

        # Variables
        self.AVG_DAYS_IN_YEAR = 365.25
        self.AVG_DAYS_IN_MONTH = 30.437
        self.AVG_TENNIS_PLAYER_AGE = 23.6
        self.MOST_COMMON_IOC = 'CHN'
        self.BET_VAR = math.sqrt(0.34)
        self.REQ_COLS = ['tourney_name', 'surface', 'tourney_level', 'tourney_date', 'winner_id', 'winner_name', 'winner_ioc', 'winner_age', \
            'loser_id', 'loser_name', 'loser_ioc', 'loser_age', 'best_of']

        self.df_order = list(self.df.columns)
        self.df_full_order = list(self.df_full.columns)

        # Model variables
        self.cnn_model = keras.models.load_model(model_file)
        self.scaler = StandardScaler()
        self.scaler.fit_transform(self.df_symm.drop(['winner'], axis=1).values)

    def train_model(self, X, y, batch_size, epochs, save_file=None):
        self.cnn_model.fit(X, y, batch_size=batch_size, epochs=epochs)
        if save_file:
            self.cnn_model.save(save_file)

    # Fix this to return the actual stats
    def get_player_stats(self, player):
        player_id = self.player_df[self.player_df['full_name'] == player].values['player_id']
        most_recent_game = self.df_full[(self.df_full['p1_id'] == player_id) | (self.df_full['p2_id'] == player_id)].iloc[-1]
        return most_recent_game

    def add_matches(self, matches_data):
        if type(matches_data == list):
            matches = self.processing.read_multiple_csv(matches_data)
        else:
            matches = self.processing.read_csv(matches, usecols=self.REQ_COLS, dtype={'tourney_level': str})

        matches.index = range(matches.shape[0])

        matches = self.processing.clean_matches(matches)
        matches = self.processing.append(matches, self.df_full)
        missing_tourneys = self.processing.check_tourney_missing(matches)
        if len(missing_tourneys) == 0:
            matches = self.processing.prepare_rolling_features(matches)
            self.df_full = self.rearrange_cols(matches, full=True)
            self.df = self.rearrange_cols(matches, full=False)
            self.df_symm = self.create_symmetric_matches(self.df)
            return matches
        else:
            print('Missing tourneys detected - check return.')
            return missing_tourneys
    
    # To add matches that are not part of official dataset
    def add_temporary_matches(self, matches_data):
        match = pd.Dataframe(matches_data)
        match = self.impute_values(match)

        matches = self.processing.clean_matches(match, True)
        matches = self.processing.append(matches, self.df_full)
        matches = self.processing.prepare_rolling_features(matches)

        self.df_full = self.rearrange_cols(matches, full=True)
        self.df = self.rearrange_cols(matches, full=False)
        self.df_symm = self.create_symmetric_matches(self.df)
        return matches
    
    def predict_match(self, vals):
        match = pd.DataFrame([vals])
        match = self.impute_values(match)

        matches = self.processing.clean_matches(match, True)
        matches = self.processing.append(matches, self.df_full)
        matches = self.processing.prepare_rolling_features(matches)
        matches = self.rearrange_cols(matches)

        latest_match = matches.tail(1).drop(['winner'], axis=1)
        latest_match_flip = self.processing.create_symmetric_matches(latest_match)

        latest_match_scaled = self.scaler.transform(latest_match.values)
        latest_match_flip_scaled = self.scaler.transform(latest_match_flip.values)

        cnn_pred = self.cnn_model(latest_match_scaled, training=False)[0][0].numpy().item()
        cnn_pred_flip = self.cnn_model(latest_match_flip_scaled, training=False)[0][0].numpy().item()

        preds = {
                'p1_win_prob' : (cnn_pred_flip + (1 - cnn_pred)) / 2,
                'p2_win_prob' : (cnn_pred + (1 - cnn_pred_flip)) / 2
            }

        return preds
    
    def get_bet(self, vals, p1_odds, p2_odds, model):
        preds = self.predict_match(vals, model)

        p1_bet = self.calculate_bet_sizing(preds['p1_win_prob'], self.get_implied_probability(p1_odds, 'Decimal'))
        p2_bet = self.calculate_bet_sizing(preds['p2_win_prob'], self.get_implied_probability(p2_odds, 'Decimal'))

        return p1_bet, p2_bet

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
    
    def rearrange_cols(self, df, full=False):
        if full:
            df = df.reindex(columns=self.df_full_order)
        else:
            df = df.reindex(columns=self.df_order)
        return df
    
    def create_symmetric_matches(self, df):
        return pd.concat([df, self.processing.create_symmetric_matches(df)]).sort_index(kind='merge').reset_index(drop=True)
    
    # BET
    def calculate_bet_sizing(self, model_prob, bookie_prob):
        sizing = self.calculate_kelly(model_prob, bookie_prob) * 10
        return round(sizing * 2) / 2 if sizing > 0 else 0

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