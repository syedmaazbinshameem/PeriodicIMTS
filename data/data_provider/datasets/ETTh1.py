from data.dependencies.MTS_Dataset.ETT_hour import Data as MTSData
import numpy as np

__all__ = ['Data']

class DataIMTS(MTSData):
    def __read_data__(self):
        super().__read_data__()  # call original read_data
        # inject irregularity
        np.random.seed(42)
        n_time, n_features = self.data_x.shape
        mask = np.random.rand(n_time, n_features) < 0.8
        self.data_x[mask] = np.nan
        self.data_y = self.data_x.copy()

# Redirect Data to our new class
Data = DataIMTS
