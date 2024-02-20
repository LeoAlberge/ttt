import numpy as np


class TTTVolume:

    def __init__(self,
                 data: np.ndarray,
                 origin_lps: np.ndarray,
                 spacing: np.ndarray,
                 matrix_ijk_2_lps: np.ndarray
                 ):
        self.data = data
        self.origin_lps = origin_lps
        self.spacing = spacing
        self.matrix_ijk_2_lps = matrix_ijk_2_lps
