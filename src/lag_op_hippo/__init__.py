__version__ = "0.1.0"
from .hippo_by_original_authors import HiPPO, variable_unroll_matrix
from .lag_hippo import (
    LagHiPPO,
    compute_A_infty,
    compute_A_gen,
    compute_B_zoh,
)
from .lag_hippo_freq import (
    FrequencyPreservingConfig,
    FrequencyPreservingBasis,
    make_frequency_preserving_basis,
    make_frequency_preserving_model,
)
from .freq_diagnostics import (
    ProjectionReconstructionResult,
    build_interpolated_signal,
    compute_inner_product,
    reconstruct_signal,
    run_projection_reconstruction,
    orthonormality_matrix_gauss,
    orthonormality_summary,
)
from .visualizers import (
    plot_reconstructed_signal,
    plot_basis_functions,
)
