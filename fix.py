import pandas as pd
import numpy as np
from typing import Optional, Union, Tuple

class FlyStepAnalyzer:
    def __init__(self, frame_rate: float = 33.0):
        self.frame_rate = frame_rate
        self.data: Optional[pd.DataFrame] = None
        self.x_key: str = 'X'
        self.y_key: str = 'Y'

    def load(self, source: Union[str, pd.DataFrame], 
             x_coord: str = None) -> 'FlyStepAnalyzer':
        if isinstance(source, pd.DataFrame):
            self.data = source
        else:
            self.data = pd.read_csv(source)
        
        if x_coord is not None:
            self.x_key = x_coord
        
        # Ensure frame index is sequential and numeric
        if 'frame' in self.data.columns:
            self.data['frame'] = self.data['frame'].astype(float)
            self.data = self.data.sort_values(by='frame').reset_index(drop=True)
            
            # Normalize X coordinate to 0-1 range for invariant zone detection
            x_vals = self.data[self.x_key]
            x_range = x_vals.max() - x_vals.min()
            self.data['norm_x'] = (x_vals - x_vals.min()) / x_range
            
        return self

    def _compute_velocity(self, window: int = 5) -> pd.Series:
        if self.x_key in self.data.columns:
            # Rolling window average reduces SLEAP jitter
            rolled = self.data[self.x_key].rolling(window=window, min_periods=1)
            self.data['velocity'] = rolled.diff()
            self.data['velocity'] = self.data['velocity'].abs()
        else:
            self.data['velocity'] = self.data['norm_x'].rolling(window=window, min_periods=1).diff()
            self.data['velocity'] = self.data['velocity'].abs()
        return self

    def _detect_crossings(self, mid_point: float = 0.5, 
                          min_speed: float = 2.0) -> pd.Series:
        self._compute_velocity()
        
        # Define 'active movement' mask
        self.data['is_active'] = self.data['velocity'] > min_speed
        
        # Create a state for crossing the midpoint
        # First, mask the midpoint logic onto the is_active data
        self.data['midpoint_cross'] = (self.data['norm_x'] > mid_point).astype(int)
        
        # Only count changes when actively moving to avoid edge jitter
        self.data['crossing_state'] = self.data['is_active'].apply(lambda x: 1 if x else np.nan)
        
        # Refine: Detect when 'midpoint_cross' changes from 0 to 1 or 1 to 0
        # Handle the edge case where movement starts in 'zone 1'
        self.data['crossing_state'] = (self.data['midpoint_cross'] - self.data['midpoint_cross'].shift(1)).fillna(0)
        self.data['crossing_state'] = self.data['crossing_state'] * self.data['is_active'].astype(float)
        
        return self.data['crossing_state']

    def count_steps(self, 
                    zones: Tuple[float, float] = (0.25, 0.75),
                    min_bout_speed: float = 2.0,
                    min_frames_per_bout: int = 3) -> pd.Series:
        if self.x_key not in self.data.columns:
            raise ValueError(f"Column '{self.x_key}' not found in DataFrame")
            
        self._compute_velocity()
        
        # Identify continuous bouts of movement
        self.data['is_bout'] = (self.data['velocity'] > min_bout_speed).astype(int)
        
        # Label bouts with IDs
        self.data['bout_id'] = self.data['is_bout'].cumsum()
        
        # Merge back original frame to count total unique bouts or steps
        # The step count is essentially the number of times the bout ID increments
        # but filtered by the 'norm_x' to determine direction if needed
        
        # Return the series representing the unique step count per frame
        self.data['step_count'] = self.data['bout_id'].astype(int)
        
        # Clean up for return
        result = self.data[['frame', self.x_key, 'step_count']]
        
        return result[['frame', self.x_key, 'step_count']]

    def summarize(self) -> pd.DataFrame:
        if self.data is None:
            return self.count_steps()
        return self.data.groupby('bout_id')['step_count'].count()

if __name__ == '__main__':
    # Simulated input data structure matching typical SLEAP export
    input_df = pd.DataFrame({
        'frame': range(60),
        'X': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150],
        'Y': [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        'prob': [0.5, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.6]
    })

    # Instantiate and Process
    analyzer = FlyStepAnalyzer(frame_rate=30.0)
    
    # Load and Fit
    tracked = analyzer.load(input_df)
    
    # Calculate Final Steps
    steps = tracked.count_steps(zones=(0.3, 0.7))
    
    # Print Result
    print(f"Total steps detected: {steps['step_count'].max()}")
    print(steps)

    # If running from command line with CSV path:
    # analyzer = FlyStepAnalyzer()
    # tracked = analyzer.load('output_sleap.csv')
    # print(analyzer.count_steps())
    
    pass