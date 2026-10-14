"""One BLAS thread for every test: bit-for-bit reproducibility (the golden runs) depends on it. Set here, before NumPy
is first imported, because BLAS reads these variables only when it loads."""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
