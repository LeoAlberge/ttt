import SimpleITK as sitk
import numpy as np

from src.python.core.volume import TTTVolume


def read_nii(path: str, dtype=np.float32) -> TTTVolume:
    reader = sitk.ImageFileReader()
    reader.SetImageIO("NiftiImageIO")
    reader.SetFileName(path)
    image = reader.Execute()
    numpy_arr = sitk.GetArrayFromImage(image)
    return TTTVolume(numpy_arr.astype(dtype),
                     np.array(image.GetOrigin()),
                     np.array(image.GetSpacing()),
                     np.array(image.GetDirection()).reshape(3, 3)
                     )
