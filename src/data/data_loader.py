import pandas as pd
import numpy as np
from typing import List, Tuple
from pathlib import Path

class DataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
    def load_excel_files(self, file_pattern: str = "*.xlsx") -> pd.DataFrame:
        """Load and concatenate all Excel files from the data directory."""
        all_files = list(self.data_dir.glob(file_pattern))
        dfs = []
        
        for file in all_files:
            df = pd.read_excel(file)
            dfs.append(df)
            
        combined_df = pd.concat(dfs, axis=0)
        combined_df = combined_df.sort_values(by=['timestamp'])
        combined_df = combined_df.drop_duplicates(subset=['timestamp'])
        
        return combined_df
    
    def preprocess_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess timestamps in the dataframe."""
        df['timestamp'] = pd.to_datetime(df['timestamp'].str.replace('@', ''))
        df['diff_seconds'] = df['timestamp'].diff(1).dt.total_seconds()
        df = df[df['diff_seconds'] <= 5].sort_values('timestamp').reset_index(drop=True)
        return df
    
    def process_rssi_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process RSSI data from the dataframe."""
        # Extract RSSI values and instance IDs
        instance_1 = df['nearest'].str.slice(16,24,1)
        instance_2 = df['nearest'].str.slice(53,61,1)
        instance_3 = df['nearest'].str.slice(90,98,1)
        rssi_1 = df['nearest'].str.slice(33,36,1)
        rssi_2 = df['nearest'].str.slice(70,73,1)
        rssi_3 = df['nearest'].str.slice(107,110,1)
        
        # Create processed dataset
        processed_df = pd.DataFrame({
            'timestamp': df['timestamp'],
            'instance_1': instance_1,
            'instance_2': instance_2,
            'instance_3': instance_3,
            'rssi_1': rssi_1,
            'rssi_2': rssi_2,
            'rssi_3': rssi_3,
            'Puck': df['instanceId']
        })
        
        # Clean and convert RSSI values
        processed_df = processed_df[processed_df != ''].dropna()
        processed_df['rssi_1'] = processed_df['rssi_1'].str.extract(r'(\d+)').astype(int) * -1
        processed_df['rssi_2'] = processed_df['rssi_2'].str.extract(r'(\d+)').astype(int) * -1
        processed_df['rssi_3'] = processed_df['rssi_3'].str.extract(r'(\d+)').astype(int) * -1
        
        return processed_df
    
    def process_acceleration_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process acceleration data from the dataframe."""
        acc_data = df['acceleration'].str.split(",", n=2, expand=True)
        acc_data.columns = ['acc_x', 'acc_y', 'acc_z']
        
        # Clean acceleration values
        acc_data['acc_x'] = acc_data['acc_x'].str.replace("[", '').str.replace("'", '').astype(float)
        acc_data['acc_y'] = acc_data['acc_y'].str.replace("'", '').astype(float)
        acc_data['acc_z'] = acc_data['acc_z'].str.replace("]", '').str.replace("'", '').astype(float)
        
        return acc_data[['acc_x', 'acc_y', 'acc_z']]
    
    def load_and_preprocess_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and preprocess all data."""
        # Load RSSI data
        rssi_df = self.load_excel_files("*.xlsx")
        rssi_df = self.preprocess_timestamps(rssi_df)
        processed_rssi = self.process_rssi_data(rssi_df)
        
        # Load acceleration data
        acc_df = self.load_excel_files("all-location.xlsx")
        acc_df = self.preprocess_timestamps(acc_df)
        processed_acc = self.process_acceleration_data(acc_df)
        
        return processed_rssi, processed_acc 