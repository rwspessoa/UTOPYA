"""
Ablation study configuration for multimodal UTOPYA.

Implements the 11-configuration matrix (A1–A11) described in the paper:
Section 5.3 and Table 4.

Bug 6 fix: A8 and A9 now use the zero-flag approach described in the paper
(line 1029): the full model (including all static encoders) is always instantiated,
but specific encoder outputs are zeroed during the forward pass to ablate their
contribution without changing the model architecture or parameter count.

A12 is NOT in the paper (only A1–A11 appear in Table 4). It is kept here as an
extension for reference but should not be cited as a paper result.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class AblationConfig:
    """Configuration for a single ablation scenario."""
    name: str              # e.g. "A1"
    description: str       # human readable description
    use_timeseries: bool   # TCN encoder (always True)
    use_gc: bool           # GC/molecular graph modality
    use_audio: bool        # Audio encoder
    use_tabular: bool      # Tabular static features
    use_text: bool         # Text encoder
    use_nmr: bool = False       # NMR composition encoder (Section 5.9 A15)
    use_image: bool = False     # Camera-frame ResNet-18 encoder (Section 5.9 A14)
    # Zero-flag fields (Bug 6): encoder is instantiated but output is zeroed
    zero_tabular: bool = False   # zero tabular output (keep encoder alive)
    zero_text: bool = False      # zero text output (keep encoder alive)
    zero_gc: bool = False        # zero GC/mol output (keep encoder alive)
    notes: Optional[str] = None


# Paper Table 4: Eleven ablation configurations
ABLATIONS = {
    "A1": AblationConfig(
        name="A1",
        description="TS only",
        use_timeseries=True,
        use_gc=False,
        use_audio=False,
        use_tabular=False,
        use_text=False,
    ),
    "A2": AblationConfig(
        name="A2",
        description="TS + GC",
        use_timeseries=True,
        use_gc=True,
        use_audio=False,
        use_tabular=False,
        use_text=False,
    ),
    "A3": AblationConfig(
        name="A3",
        description="TS + Audio",
        use_timeseries=True,
        use_gc=False,
        use_audio=True,
        use_tabular=False,
        use_text=False,
        notes="Early-stopped at epoch 15 due to training slowdown",
    ),
    "A4": AblationConfig(
        name="A4",
        description="TS + Static (tabular + text, no GC)",
        use_timeseries=True,
        use_gc=False,
        use_audio=False,
        use_tabular=True,
        use_text=True,
    ),
    "A5": AblationConfig(
        name="A5",
        description="TS + GC + Audio",
        use_timeseries=True,
        use_gc=True,
        use_audio=True,
        use_tabular=False,
        use_text=False,
    ),
    "A6": AblationConfig(
        name="A6",
        description="TS + GC + Static",
        use_timeseries=True,
        use_gc=True,
        use_audio=False,
        use_tabular=True,
        use_text=True,
    ),
    "A7": AblationConfig(
        name="A7",
        description="Full multimodal (all encoders)",
        use_timeseries=True,
        use_gc=True,
        use_audio=True,
        use_tabular=True,
        use_text=True,
    ),
    # --- A8 / A9: zero-flag ablations (Bug 6 fix) ---
    # Full model (same architecture as A7 minus audio) but with one static channel
    # zeroed during the forward pass (paper line 1029).
    "A8": AblationConfig(
        name="A8",
        description="Full static, zero text (tabular channel only active)",
        use_timeseries=True,
        use_gc=True,
        use_audio=False,
        use_tabular=True,
        use_text=True,        # encoder present, output zeroed below
        zero_text=True,       # <- zero-flag: suppresses text contribution
        notes="Paper zero-flag ablation: text encoder present but zeroed",
    ),
    "A9": AblationConfig(
        name="A9",
        description="Full static, zero tabular (text channel only active)",
        use_timeseries=True,
        use_gc=True,
        use_audio=False,
        use_tabular=True,     # encoder present, output zeroed below
        use_text=True,
        zero_tabular=True,    # <- zero-flag: suppresses tabular contribution
        notes="Paper zero-flag ablation: tabular encoder present but zeroed",
    ),
    "A10": AblationConfig(
        name="A10",
        description="TS + Tabular + Text (no GC, no audio)",
        use_timeseries=True,
        use_gc=False,
        use_audio=False,
        use_tabular=True,
        use_text=True,
        notes="Converges to same optimum as A4",
    ),
    "A11": AblationConfig(
        name="A11",
        description="TS + Audio + Tabular + Text (no GC)",
        use_timeseries=True,
        use_gc=False,
        use_audio=True,
        use_tabular=True,
        use_text=True,
    ),
    # NOTE: A12 is NOT in the paper (Table 4 lists only A1–A11).
    # Kept here as a supplementary extension.
    "A12": AblationConfig(
        name="A12",
        description="TS + GC + Text (extension, not in paper)",
        use_timeseries=True,
        use_gc=True,
        use_audio=False,
        use_tabular=False,
        use_text=True,
        notes="Extension beyond paper Table 4 — not a reported result",
    ),
}


def get_ablation_config(name: str) -> Optional[AblationConfig]:
    """Retrieve ablation config by name (e.g. 'A1', 'A7')."""
    return ABLATIONS.get(name)


def list_ablations() -> List[str]:
    """Return list of all ablation names in order."""
    return sorted(ABLATIONS.keys())


def get_static_modalities_enabled(config: AblationConfig) -> List[str]:
    """Return list of enabled static modalities for a config."""
    modalities = []
    if config.use_tabular:
        modalities.append("tabular")
    if config.use_text:
        modalities.append("text")
    if config.use_gc:
        modalities.append("gc")
    return modalities
