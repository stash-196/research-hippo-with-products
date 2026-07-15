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
from .lag_hippo_translated import (
    TruncatedTranslateConfig,
    TruncatedTranslateBasis,
    LegTBasis,
    FouTBasis,
    make_truncated_translate_basis,
    make_legt_basis,
    make_fout_basis,
    make_legt_model,
    make_fout_model,
    legt_input_vector,
    fout_input_vector,
    compare_continuous_A_to_hippo,
    compare_discrete_A_to_hippo,
)
from .freq_diagnostics import (
    ProjectionReconstructionResult,
    WarpMeasureDensityResult,
    build_interpolated_signal,
    compute_inner_product,
    compute_warp_measure_densities,
    reconstruct_signal,
    run_projection_reconstruction,
    orthonormality_matrix_gauss,
    orthonormality_summary,
)
from .visualizers import (
    plot_reconstructed_signal,
    plot_basis_functions,
    plot_inner_product_matrix,
    plot_roundtrip_error,
    plot_measure_density_warp,
)
