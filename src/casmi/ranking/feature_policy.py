"""Model-feature governance: which pair/molecule-table columns a ranker is allowed to train on.

Motivating shortcut this exists to block: under the unseen-connectivity CV regime, the TRUE
candidate structurally has ZERO reference spectra (by construction -- see
`casmi.spectra.library`), while many FALSE candidates do have them. A flexible ranker handed
`has_reference_spectrum` (or `n_reference_spectra`, or which database a candidate came from) as
a raw feature can trivially learn "no reference spectrum -> true candidate", scoring
suspiciously well in CV for a reason that has nothing to do with spectral evidence and will not
generalize to real unseen structures. Fields like this are retained for PROVENANCE/DIAGNOSTICS
(explaining a result) but excluded from `model_features` by default; promoting one requires the
shortcut tests in `casmi.validation.test_like` (masked-spectrum / candidate-order controls) to
demonstrate it isn't functioning as a leakage shortcut.
"""
from dataclasses import dataclass, field


@dataclass
class FeaturePolicy:
    model_features: list = field(default_factory=list)
    diagnostic_features: list = field(default_factory=list)
    prohibited_features: list = field(default_factory=list)

    def __post_init__(self):
        overlap_md = set(self.model_features) & set(self.diagnostic_features)
        overlap_mp = set(self.model_features) & set(self.prohibited_features)
        if overlap_md:
            raise ValueError(f"features in both model_features and diagnostic_features: {sorted(overlap_md)}")
        if overlap_mp:
            raise ValueError(f"features in both model_features and prohibited_features: {sorted(overlap_mp)}")


# Default classification for this project's known reference-availability / provenance fields
# (spec section 2): every one of these defaults to provenance-only / diagnostic, NEVER a raw
# model feature, unless explicitly promoted after passing the shortcut controls.
DEFAULT_PROHIBITED_FEATURES = [
    "is_train_library", "candidate_source", "is_external_candidate",
    "has_reference_spectrum", "n_reference_spectra", "source_train", "source_coconut",
    "source_chebi", "source_names", "source_count", "candidate_sources",
]


def assert_no_prohibited_features(model_df, feature_policy):
    """Raises `AssertionError` (never continues silently) if any column in `model_df` is on
    `feature_policy.prohibited_features`. Call this immediately before handing a feature frame
    to a ranker -- the ranker fails loudly, not the CV score silently."""
    present = set(model_df.columns) & set(feature_policy.prohibited_features)
    assert not present, f"prohibited (leakage-prone) features present in model input: {sorted(present)}"


# The single feature policy notebook 04's classical baselines and notebook 09's LambdaMART both
# train/score against -- defined once here so every notebook imports the SAME split rather than
# each redefining its own copy that could silently drift out of sync.
SPECTRAL_RANKING_FEATURE_POLICY = FeaturePolicy(
    model_features=["abs_mass_error_ppm", "cosine_max", "cosine_top3_mean", "modified_cosine_max",
                     "modified_cosine_top3_mean", "peak_overlap_frac_max", "peak_overlap_frac_top3_mean",
                     "neutral_loss_cosine_max", "neutral_loss_cosine_top3_mean"],
    diagnostic_features=["n_reference_spectra", "has_reference_spectrum"],
    prohibited_features=DEFAULT_PROHIBITED_FEATURES,
)
