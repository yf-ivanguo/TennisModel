import pandas as pd
import re, csv

class DataChange():
    # Dataset Manipulation Functions
    def add_davis_cup_tourney(self, input_string, output_csv):
        pattern = r': (\w+) vs (\w+)'
        match = re.search(pattern, input_string)
        
        if match:
            country_code = match.group(1)
            updated_country_code = self.convert_ioc_or_iso3_to_iso2(country_code)
            with open(output_csv, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([input_string, updated_country_code])
            return True
        return False

    def add_tourney(self, input_string, output_csv):
        output_df = pd.read_csv(output_csv)
        if input_string in output_df['tournament'].unique():
            print(f'{input_string} already located inside csv.')
            return
        
        is_davis_cup = self.add_davis_cup_tourney(input_string, output_csv)
        output_df = pd.read_csv(output_csv)
        if not is_davis_cup:
            with open(output_csv, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([input_string])
                print(f'Added {input_string} to tournament csv')