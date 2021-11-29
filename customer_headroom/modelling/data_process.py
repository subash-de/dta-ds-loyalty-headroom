from typing import Optional
import numpy as np
import pandas as pd
import surprise.dataset as surprise_ds
from surprise import Dataset
from surprise.reader import Reader
from dtaml.logging import get_logger

logger = get_logger("customer-headroom")


class DataProcessor(object):
    def __init__(
            self,
            feature_col: str,
            item_id: str,
            user_id: str = "cust_id",
            lognorm: bool = True,
            line_format: str = "user item rating",
            scale_tol: float = 0.1,
            min_lim: Optional[float] = None,
            max_lim: Optional[float] = None,
    ):
        self.feature_col = feature_col
        self.user_id = user_id
        self.item_id = item_id
        self.lognorm = lognorm
        self.line_format = line_format
        self.scale_tol = scale_tol
        self.min_lim = min_lim
        self.max_lim = max_lim

    def get(self, data: pd.DataFrame) -> surprise_ds.DatasetAutoFolds:
        """
        Preprocess `data` to create a dataset compatible with the Surprise Recommender Models.

        Often taking the lognorm of a column is better suited for recommendation algorithims since the values
        resemble a normal distributions more closely. This is enabled with lognorm=True when DataProcessor is initiated.
        """
        if self.lognorm:
            self.value_col = f"{self.feature_col}_lognorm"
            min_scale = (1 - self.scale_tol) * data[self.feature_col].min()
            self.min_col = min(i for i in [min_scale, self.min_lim] if i is not None)
            max_scale = (1 + self.scale_tol) * data[self.feature_col].max()
            self.max_col = max(i for i in [max_scale, self.max_lim] if i is not None)
        else:
            self.value_col = self.feature_col

        rec_data = self.transform(data)
        return rec_data

    def transform(self,
                  data: pd.DataFrame) -> surprise_ds.DatasetAutoFolds:
        """
        This task creates a surprise AutoFolds Dataset which is the input dataset to surprise models.
        The rating_scale and reader objects are defined here.
        """
        if self.lognorm:
            data = self._lognorm_col(data,
                                     col=self.feature_col,
                                     min_col=self.min_col,
                                     max_col=self.max_col)

        data_ = data.loc[:, [self.user_id, self.item_id, self.value_col]]

        self.rating_scale = (min(data_.loc[:, self.value_col]), max(data_.loc[:, self.value_col]))
        self.reader = Reader(line_format=self.line_format, rating_scale=self.rating_scale)
        rec_data = Dataset.load_from_df(data_, self.reader)
        return rec_data

    @staticmethod
    def _lognorm_col(df: pd.DataFrame,
                     col: str,
                     min_col: Optional[float] = None,
                     max_col: Optional[float] = None
                     ) -> pd.DataFrame:
        """
        StaticMethod for calculating the lognorm of a desired column.
        """
        if not min_col:
            min_col = 0.9 * df.loc[:, col].min()
        if not max_col:
            max_col = 1.1 * df.loc[:, col].max()
        #
        df.loc[:, col] = df.loc[:, col].astype("float64")
        df.loc[:, f"{col}_norm"] = (df.loc[:, col] - min_col) / (max_col - min_col)
        df.loc[:, f"{col}_lognorm"] = df.loc[:, f"{col}_norm"].apply(lambda x: np.log(x))
        return df
