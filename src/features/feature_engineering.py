import pandas as pd
import numpy as np
import pygame
from typing import Dict, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

class FeatureEngineer:
    def __init__(self):
        self.x_axis = {
            '0000004d': 0, '00000058': 4, '00000060': 8,
            '0000004e': 8, '00000061': 4, '00000059': 0,
            '000000a9': 0, '000000ae': 4, '000000af': 8
        }
        self.y_axis = {
            '0000004d': 0, '00000058': 4, '00000060': 0,
            '0000004e': 4, '00000061': 4, '00000059': 4,
            '000000a9': 8, '000000ae': 8, '000000af': 8
        }
        self.label_encoder = LabelEncoder()
        self.angle_scaler = MinMaxScaler()
        self.rssi_scaler = StandardScaler()
        
    def calculate_angles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate angles between target and transmitters."""
        def angle_of_vector(x: float, y: float) -> float:
            return pygame.math.Vector2(x, y).angle_to((1, 0))
        
        def angle_of_line(row: pd.Series) -> float:
            return angle_of_vector(row['X1']-row['location_X'], row['Y1']-row['location_Y'])
        
        def angle_correction(row: pd.Series) -> float:
            angle = row['angle_1']
            if angle == 0:
                return angle
            elif 0 > angle >= -180:
                return -1 * angle
            elif 0 < angle < 180:
                return 360 - angle
            return angle
        
        # Calculate angles for each transmitter
        for i in range(1, 4):
            df[f'angle_{i}'] = df.apply(lambda x: angle_of_vector(
                x[f'X{i}']-x['location_X'],
                x[f'Y{i}']-x['location_Y']
            ), axis=1)
            df[f'angle_{i}'] = df.apply(angle_correction, axis=1)
        
        return df
    
    def calculate_distances(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate distances between points."""
        # Distance from target to transmitters
        for i in range(1, 4):
            df[f'dis{i}'] = np.sqrt(
                ((df['location_X']-df[f'X{i}'])**2) +
                ((df['location_Y']-df[f'Y{i}'])**2)
            )
        
        # Distance between transmitters
        df['dis1_1'] = np.sqrt(((df['X2']-df['X1'])**2) + ((df['Y2']-df['Y1'])**2))
        df['dis2_1'] = np.sqrt(((df['X3']-df['X2'])**2) + ((df['Y3']-df['Y2'])**2))
        df['dis3_1'] = np.sqrt(((df['X1']-df['X3'])**2) + ((df['Y1']-df['Y3'])**2))
        
        return df
    
    def add_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add X and Y coordinates for each transmitter."""
        for i in range(1, 4):
            df[f'X{i}'] = df[f'Y{i}'].replace(self.x_axis)
            df[f'Y{i}'] = df[f'Y{i}'].replace(self.y_axis)
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for model training."""
        # Add coordinates
        df = self.add_coordinates(df)
        
        # Calculate angles and distances
        df = self.calculate_angles(df)
        df = self.calculate_distances(df)
        
        # Prepare feature matrix
        feature_columns = [
            'rssi_1', 'rssi_2', 'rssi_3',
            'angle_1', 'angle_2', 'angle_3',
            'dis1', 'dis2', 'dis3',
            'dis1_1', 'dis2_1', 'dis3_1'
        ]
        
        X = df[feature_columns].copy()
        
        # Scale features
        X[['angle_1', 'angle_2', 'angle_3']] = self.angle_scaler.fit_transform(
            X[['angle_1', 'angle_2', 'angle_3']]
        )
        
        X[['rssi_1', 'rssi_2', 'rssi_3', 'dis1', 'dis2', 'dis3', 'dis1_1', 'dis2_1', 'dis3_1']] = \
            self.rssi_scaler.fit_transform(
                X[['rssi_1', 'rssi_2', 'rssi_3', 'dis1', 'dis2', 'dis3', 'dis1_1', 'dis2_1', 'dis3_1']]
            )
        
        # Encode labels
        y = self.label_encoder.fit_transform(df['label'])
        
        return X, y
    
    def get_label_mapping(self) -> Dict[int, str]:
        """Get the mapping between encoded labels and original labels."""
        return dict(zip(range(len(self.label_encoder.classes_)), self.label_encoder.classes_)) 