"""
Full UTOPYA model (Section 3, Figure 2).

Architecture (forward pass):
  1. TCN encoder           : x_ts → z_ts (B,d), Z_ts (B,T,d)
  2. Static encoders       : x_tab, x_text, x_mol → c ∈ R^{B×d}   (Eq. 6)
  3. FiLM conditioning     : Z_ts → Z_ts' conditioned on c          (Eq. 7)
  4. Cross-modal attention : Z_ts' → Z_ts'' (self-attn placeholder)
  5. Gated fusion          : Z_ts'' → z_fused (B,d)                (Eq. 11, normalised)
  6. Output heads:
       - PredictionHead       : z_fused → y_hat (B, H, N_cont)
       - ClassificationHead   : z_fused → cls_logits (B, 2)
       - LocalisationHead     : Z_ts'' → loc_logits (B, T, N_vars)

Bug fixes applied (vs. original):
  • Bug 4  – use_tabular flag now correctly gates the tabular encoder
  • Bug 6  – zero_tabular / zero_text zero-flags for ablation A8/A9 (paper §5.3)
  • Bug 7  – gated fusion denominator added (normalised Eq. 11)
  • Bug 3  – AudioEncoder integrated
  • Bug 1  – x_text wired through forward (requires dataset to supply SBERT vectors)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.models.tcn import TCNEncoder
from src.models.encoders import (
    TabularEncoder,
    TextEncoder,
    AudioEncoder,
    MolecularEncoder,
    NMREncoder,
    ImageEncoder,
    StaticContextAggregator,
    FiLMConditioner,
)
from src.models.fusion import (
    CrossModalAttention,
    GatedFusion,
    PredictionHead,
    AnomalyClassificationHead,
    AnomalyLocalisationHead,
    PhaseClassificationHead,
    ReconstructionHead,
)


def _apply_zero_mask(z: torch.Tensor, flag) -> torch.Tensor:
    """
    Zero out (part of) an encoder output.

    `flag` may be:
      - False / None            : no-op (return z unchanged)
      - True                    : zero the whole batch (legacy ablation behaviour)
      - torch.Tensor, shape (B,) or (B,1) : per-sample zero-mask (1=zero-out,
        0=keep), enabling independent per-window stochastic modality dropout
        (e.g. Bernoulli(p) sampled once per window) instead of an all-or-
        nothing per-batch flag.
    """
    if flag is False or flag is None:
        return z
    if flag is True:
        return torch.zeros_like(z)
    mask = flag.to(device=z.device, dtype=z.dtype)
    if mask.dim() == 1:
        mask = mask.unsqueeze(-1)
    return z * (1.0 - mask)


class UTOPYAModel(nn.Module):
    """
    Full UTOPYA model.

    Parameters
    ----------
    d_tab       : dimension of tabular feature vector (default 14, see tabular.py)
    n_vars      : number of input timeseries variables (default 31)
    n_cont      : number of continuous variables for prediction (default 25)
    horizon     : prediction horizon in timesteps (default 60)
    window      : input window size (default 120)
    dmodel      : internal embedding dimension (default 128)
    sbert_dim   : SBERT embedding dimension if text encoder is used (default 384)
    use_tabular : include TabularEncoder in static context (default True)
    use_text    : include TextEncoder in static context (requires SBERT embeddings)
    use_gc      : include MolecularEncoder/GCN in static context
    use_audio   : include AudioEncoder as additional dynamic-side encoder
    dropout     : dropout probability
    n_heads     : number of attention heads for cross-modal attention

    Forward zero-flag parameters (ablation only, Bug 6)
    ---------------------------------------------------
    zero_tabular : pass zeros in place of tab_enc output (encoder still computed)
    zero_text    : pass zeros in place of text_enc output
    zero_gc      : pass zeros in place of mol/GCN output

    GC modality setup (Bug 2)
    -------------------------
    Before training with use_gc=True, call register_molecule_graphs() with the
    molecule graphs built by src.data.molecular.build_molecule_graphs().
    During forward, mol_composition (B, n_mol) molar fractions are used to
    weight the per-molecule GCN embeddings into a single z_mol (B, dmodel).
    """

    def __init__(
        self,
        d_tab:     int   = 14,
        n_vars:    int   = 31,
        n_cont:    int   = 25,
        horizon:   int   = 60,
        window:    int   = 120,
        dmodel:    int   = 128,
        sbert_dim: int   = 384,
        use_tabular: bool = True,
        use_text:    bool = True,
        use_gc:      bool = False,
        use_audio:   bool = False,
        use_nmr:     bool = False,
        use_image:   bool = False,
        dropout:   float = 0.5,
        n_heads:   int   = 4,
        # backward-compat alias
        use_mol:   bool  = False,
    ):
        super().__init__()

        # Respect legacy use_mol parameter
        use_gc = use_gc or use_mol

        self.use_tabular = use_tabular
        self.use_text    = use_text
        self.use_gc      = use_gc
        self.use_audio   = use_audio
        self.use_nmr     = use_nmr
        self.use_image   = use_image
        self.dmodel      = dmodel

        # --- 1. TCN encoder (always present) ---
        self.tcn = TCNEncoder(
            n_input_vars=n_vars,
            dmodel=dmodel,
            dropout=dropout,
        )

        # --- 2. Static encoders (created based on flags) ---
        if use_tabular:
            self.tab_enc = TabularEncoder(d_tab, dmodel, dropout=dropout)
        if use_text:
            self.text_enc = TextEncoder(sbert_dim, dmodel)
        if use_gc:
            self.mol_enc = MolecularEncoder(dmodel)
        if use_audio:
            self.audio_enc = AudioEncoder(dmodel=dmodel)
        if use_nmr:
            self.nmr_enc = NMREncoder(dmodel=dmodel)
        if use_image:
            self.image_enc = ImageEncoder(dmodel=dmodel)

        # Static aggregator: always 3-way input matching paper Eq. 6
        # (tab; text; mol) → concatenated → projection.
        # If fewer than 3 encoders are active, the active ones are concatenated.
        n_static = int(use_tabular) + int(use_text) + int(use_gc)
        if n_static > 0:
            self.static_agg = nn.Sequential(
                nn.Linear(n_static * dmodel, dmodel),
                nn.ReLU(),
                nn.LayerNorm(dmodel),
            )
        else:
            self.static_agg = None   # A1: TS-only — context is all-zeros

        self._n_static = n_static

        # --- 3. FiLM conditioner (applied per-timestep on Z_ts) ---
        self.film = FiLMConditioner(dmodel)

        # --- 4. Cross-modal attention (self-attention on Z_ts') ---
        self.cross_attn = nn.MultiheadAttention(
            dmodel, n_heads, dropout=0.1, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(dmodel)

        # --- 5. Gated fusion (Eq. 11) ---
        self.gate_proj = nn.Linear(dmodel, dmodel)

        # --- 6. Output heads ---
        self.pred_head = PredictionHead(dmodel, horizon, n_cont)
        self.cls_head  = AnomalyClassificationHead(dmodel)
        self.loc_head  = AnomalyLocalisationHead(dmodel, n_vars)
        # Post-hoc probe head (not part of the original architecture) —
        # see PhaseClassificationHead docstring.
        self.phase_head = PhaseClassificationHead(dmodel)
        # Reconstruction head (Section 5.5 / 5.7.3): gated by w_recon=0 in the
        # production model, but instantiated so it can be trained standalone
        # on normal-only windows with the rest of the model frozen (see
        # scripts_reconstruction/train_reconstruction_head.py).
        self.recon_head = ReconstructionHead(dmodel, window=window, n_vars=n_vars)

    # ------------------------------------------------------------------
    def register_molecule_graphs(
        self,
        graphs: list,   # list of (node_feats, adj_norm) tuples, one per molecule
    ) -> None:
        """
        Register fixed molecule graphs as model buffers.

        Must be called before training when use_gc=True.  The graphs are stored
        on the same device as the model and moved automatically with .to(device).

        Parameters
        ----------
        graphs : list of (node_feats, adj_norm)
            Each element is a 2-tuple of tensors for one molecule:
              - node_feats : (N_atoms, N_ATOM_FEATURES) float32
              - adj_norm   : (N_atoms, N_atoms) float32
        """
        for i, (nf, adj) in enumerate(graphs):
            self.register_buffer(f"mol_nf_{i}",  nf)
            self.register_buffer(f"mol_adj_{i}", adj)
        self._n_molecules = len(graphs)

    # ------------------------------------------------------------------
    def _compute_z_mol(
        self,
        mol_composition: Optional[torch.Tensor],   # (B, n_mol)
    ) -> Optional[torch.Tensor]:
        """
        Run MolecularEncoder on each registered graph and compute the
        mole-fraction-weighted combination.

        Returns (B, dmodel) or None if no graphs are registered.
        """
        n_mol = getattr(self, "_n_molecules", 0)
        if n_mol == 0:
            return None

        # Run GCN on each fixed molecule graph — small graphs, fast
        z_mols = []
        for i in range(n_mol):
            nf  = getattr(self, f"mol_nf_{i}")    # (N_atoms, d_feat)
            adj = getattr(self, f"mol_adj_{i}")   # (N_atoms, N_atoms)
            z_i = self.mol_enc(nf, adj)            # (dmodel,)
            z_mols.append(z_i)
        z_stack = torch.stack(z_mols, dim=0)       # (n_mol, dmodel)

        if mol_composition is not None:
            # Weighted sum: (B, n_mol) @ (n_mol, dmodel) → (B, dmodel)
            comp = mol_composition.to(z_stack.device)  # (B, n_mol)
            return comp @ z_stack                       # (B, dmodel)
        else:
            # Equal weights if no composition provided
            return z_stack.mean(dim=0).unsqueeze(0).expand(1, -1)   # (1, dmodel)

    # ------------------------------------------------------------------
    def _build_static_context(
        self,
        x_tab:           torch.Tensor,                # (B, d_tab)
        x_text:          Optional[torch.Tensor],      # (B, sbert_dim) or None
        mol_composition: Optional[torch.Tensor],      # (B, n_mol) molar fractions or None
        zero_tabular: bool = False,
        zero_text:    bool = False,
        zero_gc:      bool = False,
    ) -> torch.Tensor:                                 # c ∈ R^{B × dmodel}
        """
        Build the FiLM context vector from available static modalities.

        The GC embedding z_mol is computed on-the-fly inside this method by
        running MolecularEncoder on the registered molecule graphs and weighting
        by mol_composition (enabling end-to-end gradient flow through the GCN).

        zero_* flags (Bug 6): zero-out the encoder output while keeping the
        encoder structurally active, as described in the paper for A8/A9.
        """
        B      = x_tab.size(0)
        device = x_tab.device
        parts  = []

        if self.use_tabular:
            z = self.tab_enc(x_tab)
            z = _apply_zero_mask(z, zero_tabular)
            parts.append(z)

        if self.use_text:
            if x_text is not None:
                z = self.text_enc(x_text)
            else:
                z = torch.zeros(B, self.dmodel, device=device)
            z = _apply_zero_mask(z, zero_text)
            parts.append(z)

        if self.use_gc:
            z_mol = self._compute_z_mol(mol_composition)
            if z_mol is not None:
                # Broadcast to (B, dmodel) if needed (e.g., when no composition)
                if z_mol.size(0) == 1 and B > 1:
                    z_mol = z_mol.expand(B, -1)
                z = z_mol
            else:
                z = torch.zeros(B, self.dmodel, device=device)
            z = _apply_zero_mask(z, zero_gc)
            parts.append(z)

        if not parts:
            return torch.zeros(B, self.dmodel, device=device)

        cat = torch.cat(parts, dim=-1)
        return self.static_agg(cat)

    # ------------------------------------------------------------------
    def forward(
        self,
        x_ts:            torch.Tensor,                    # (B, T, N_vars)
        x_tab:           torch.Tensor,                    # (B, d_tab)
        x_text:          Optional[torch.Tensor] = None,   # (B, sbert_dim) or None
        mol_composition: Optional[torch.Tensor] = None,   # (B, n_mol) molar fractions
        x_audio:         Optional[torch.Tensor] = None,   # (B, 1, n_mels, T_audio) or None
        x_nmr:           Optional[torch.Tensor] = None,   # (B, N_NMR_FEATURES) or None
        x_image:         Optional[torch.Tensor] = None,   # (B, N_IMAGE_FEATURES) or None
        zero_tabular:    bool = False,                    # ablation zero-flag (Bug 6)
        zero_text:       bool = False,                    # ablation zero-flag (Bug 6)
        zero_gc:         bool = False,                    # ablation zero-flag
        zero_audio:      bool = False,                    # ablation zero-flag
        zero_nmr:        bool = False,                    # ablation zero-flag (Section 5.9 A15)
        zero_image:      bool = False,                    # ablation zero-flag (Section 5.9 A14)
        return_embedding: bool = False,                   # expose z_fused (post-hoc, e.g. UMAP)
        return_phase:     bool = False,                   # expose phase_logits (post-hoc probe head)
        return_reconstruction: bool = False,               # expose recon_hat (Section 5.7.3 standalone head)
    ):
        """
        Returns
        -------
        y_hat      : (B, H, N_cont)   prediction
        cls_logits : (B, 2)           anomaly classification
        loc_logits : (B, T, N_vars)   anomaly localisation
        z_fused    : (B, dmodel)      fused embedding, only if return_embedding=True
        phase_logits : (B, 4)         phase classification, only if return_phase=True
        recon_hat  : (B, W, V_in)     reconstructed input window, only if return_reconstruction=True

        zero_tabular/zero_text/zero_gc/zero_audio/zero_nmr/zero_image accept
        either a Python bool (whole-batch ablation, legacy behaviour) or a
        (B,)/(B,1) tensor for independent per-window stochastic masking
        (e.g. training-time modality dropout, or the graduated per-modality
        Bernoulli-dropout robustness sweep) — see _apply_zero_mask.
        """
        # 1. TCN
        z_ts, Z_ts = self.tcn(x_ts)   # (B,d), (B,T,d)

        # 2. Static context (GCN computed in-graph for gradient flow)
        c = self._build_static_context(
            x_tab, x_text, mol_composition,
            zero_tabular=zero_tabular,
            zero_text=zero_text,
            zero_gc=zero_gc,
        )   # (B, d)

        # 3. FiLM conditioning (broadcast c over time axis)
        c_exp  = c.unsqueeze(1).expand_as(Z_ts)   # (B, T, d)
        Z_ts_f = self.film(Z_ts, c_exp)           # (B, T, d)

        # 3b. Additional dynamic-modality tokens (Section 3.2 / 5.9): audio,
        # NMR, and image embeddings are each appended as an extra key/value
        # token so the TCN embeddings can attend to them in cross-modal
        # attention (step 4). Query stays Z_ts_f only, so T (not T+k) output
        # tokens are returned.
        kv_tokens = [Z_ts_f]
        if self.use_audio and x_audio is not None:
            z_audio = self.audio_enc(x_audio)          # (B, d)
            z_audio = _apply_zero_mask(z_audio, zero_audio)
            kv_tokens.append(z_audio.unsqueeze(1))
        if self.use_nmr and x_nmr is not None:
            z_nmr = self.nmr_enc(x_nmr)                 # (B, d)
            z_nmr = _apply_zero_mask(z_nmr, zero_nmr)
            kv_tokens.append(z_nmr.unsqueeze(1))
        if self.use_image and x_image is not None:
            z_image = self.image_enc(x_image)           # (B, d)
            z_image = _apply_zero_mask(z_image, zero_image)
            kv_tokens.append(z_image.unsqueeze(1))
        kv = torch.cat(kv_tokens, dim=1) if len(kv_tokens) > 1 else Z_ts_f   # (B, T+k, d)

        # 4. Cross-modal attention (Z_ts attends to itself + extra tokens)
        Z_ts_a, _ = self.cross_attn(Z_ts_f, kv, kv)
        Z_ts_a    = self.attn_norm(Z_ts_f + Z_ts_a)   # residual + norm (B, T, d)

        # 5. Normalised gated fusion (Bug 7 fix – paper Eq. 11)
        #    z_fused = Σ(m_i · g_i · ẑ_i) / (Σ(m_i · g_i) + ε)
        #    For single modality m=1: z_fused = g · z̄ / (g + ε)
        z_bar   = Z_ts_a.mean(dim=1)                   # (B, d)
        gate    = torch.sigmoid(self.gate_proj(z_bar))  # (B, d)
        z_fused = (gate * z_bar) / (gate + 1e-6)       # (B, d)  normalised

        # 6. Output heads
        y_hat      = self.pred_head(z_fused)    # (B, H, N_cont)
        cls_logits = self.cls_head(z_fused)     # (B, 2)
        loc_logits = self.loc_head(Z_ts_a)      # (B, T, N_vars)

        out = [y_hat, cls_logits, loc_logits]
        if return_embedding:
            out.append(z_fused)
        if return_phase:
            out.append(self.phase_head(z_fused.detach()))
        if return_reconstruction:
            out.append(self.recon_head(z_fused.detach()))
        return tuple(out)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
