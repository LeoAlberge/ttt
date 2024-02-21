import SimpleITK as sitk
# import nibabel as nib
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


# def read_nii_from_nib(path: str, dtype=np.float32) -> TTTVolume:
#     nii_img = nib.load(path)
#     numpy_arr = np.ascontiguousarray(np.transpose(nii_img.get_fdata(dtype=dtype), (2, 1,
#     0)))  # Transpose to match the expected shape
#     origin = np.array(nii_img.affine[:3, 3])
#     origin = np.array(
#         [-origin[0], -origin[1], origin[2]])  # Correcting for the discrepancy in origins
#
#     return TTTVolume(numpy_arr,
#                      origin,
#                      np.array(nii_img.header.get_zooms()[:3]),  # spacing
#                      np.array(nii_img.affine[:3, :3])  # direction
#                      )

if __name__ == '__main__':
    v1 = read_nii(
        r"C:\Users\LeoAlberge\work\personnal\data\Totalsegmentator_dataset_small_v201\s0011\ct"
        r".nii.gz")
    v2 = read_nii_from_nib(
        r"C:\Users\LeoAlberge\work\personnal\data\Totalsegmentator_dataset_small_v201\s0011\ct"
        r".nii.gz")
    np.testing.assert_almost_equal(v1.data, v2.data)
    np.testing.assert_almost_equal(v1.origin_lps, v2.origin_lps)
    np.testing.assert_almost_equal(v1.spacing, v2.spacing)
    np.testing.assert_almost_equal(v1.matrix_ijk_2_lps, v2.matrix_ijk_2_lps)
