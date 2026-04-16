"""
Human ligand-receptor (LR) pair database.
Derived from CellChat human database covering major signaling pathways
relevant to pancreatic biology.
"""

import pandas as pd


def get_lr_database() -> pd.DataFrame:
    """
    Returns a DataFrame of human LR pairs with columns:
        ligand, receptor, pathway
    """
    pairs = [
        # ── Insulin / Pancreas-specific ──────────────────────────────────────
        ("INS",    "INSR",    "INSULIN"),
        ("INS",    "IGF1R",   "INSULIN"),
        ("GCG",    "GCGR",    "GLUCAGON"),
        ("SST",    "SSTR1",   "SST"),
        ("SST",    "SSTR2",   "SST"),
        ("SST",    "SSTR3",   "SST"),
        ("SST",    "SSTR5",   "SST"),
        ("GLP1",   "GLP1R",   "GLP1"),   # GCG-derived peptide
        # ── EGF family ───────────────────────────────────────────────────────
        ("EGF",    "EGFR",    "EGF"),
        ("TGFA",   "EGFR",    "EGF"),
        ("AREG",   "EGFR",    "EGF"),
        ("EREG",   "EGFR",    "EGF"),
        ("HBEGF",  "EGFR",    "EGF"),
        ("NRG1",   "ERBB2",   "EGF"),
        ("NRG1",   "ERBB3",   "EGF"),
        ("NRG1",   "ERBB4",   "EGF"),
        ("NRG2",   "ERBB3",   "EGF"),
        ("NRG2",   "ERBB4",   "EGF"),
        # ── FGF family ───────────────────────────────────────────────────────
        ("FGF1",   "FGFR1",   "FGF"),
        ("FGF1",   "FGFR2",   "FGF"),
        ("FGF1",   "FGFR3",   "FGF"),
        ("FGF2",   "FGFR1",   "FGF"),
        ("FGF2",   "FGFR2",   "FGF"),
        ("FGF7",   "FGFR2",   "FGF"),
        ("FGF10",  "FGFR2",   "FGF"),
        # ── IGF ──────────────────────────────────────────────────────────────
        ("IGF1",   "IGF1R",   "IGF"),
        ("IGF2",   "IGF1R",   "IGF"),
        ("IGF2",   "IGF2R",   "IGF"),
        # ── VEGF ─────────────────────────────────────────────────────────────
        ("VEGFA",  "FLT1",    "VEGF"),
        ("VEGFA",  "KDR",     "VEGF"),
        ("VEGFA",  "NRP1",    "VEGF"),
        ("VEGFB",  "FLT1",    "VEGF"),
        ("VEGFC",  "FLT4",    "VEGF"),
        # ── PDGF ─────────────────────────────────────────────────────────────
        ("PDGFA",  "PDGFRA",  "PDGF"),
        ("PDGFB",  "PDGFRB",  "PDGF"),
        ("PDGFC",  "PDGFRA",  "PDGF"),
        # ── Notch ────────────────────────────────────────────────────────────
        ("DLL1",   "NOTCH1",  "NOTCH"),
        ("DLL1",   "NOTCH2",  "NOTCH"),
        ("DLL1",   "NOTCH3",  "NOTCH"),
        ("DLL3",   "NOTCH1",  "NOTCH"),
        ("DLL3",   "NOTCH2",  "NOTCH"),
        ("DLL4",   "NOTCH1",  "NOTCH"),
        ("DLL4",   "NOTCH2",  "NOTCH"),
        ("DLL4",   "NOTCH4",  "NOTCH"),
        ("JAG1",   "NOTCH1",  "NOTCH"),
        ("JAG1",   "NOTCH2",  "NOTCH"),
        ("JAG1",   "NOTCH3",  "NOTCH"),
        ("JAG1",   "NOTCH4",  "NOTCH"),
        ("JAG2",   "NOTCH1",  "NOTCH"),
        ("JAG2",   "NOTCH2",  "NOTCH"),
        ("JAG2",   "NOTCH3",  "NOTCH"),
        # ── Wnt ──────────────────────────────────────────────────────────────
        ("WNT1",   "FZD1",    "WNT"),
        ("WNT1",   "FZD5",    "WNT"),
        ("WNT2",   "FZD1",    "WNT"),
        ("WNT2",   "FZD5",    "WNT"),
        ("WNT3",   "FZD1",    "WNT"),
        ("WNT3A",  "FZD1",    "WNT"),
        ("WNT4",   "FZD6",    "WNT"),
        ("WNT5A",  "FZD1",    "WNT"),
        ("WNT5A",  "FZD2",    "WNT"),
        ("WNT5A",  "ROR2",    "WNT"),
        ("WNT7A",  "FZD1",    "WNT"),
        ("WNT7B",  "FZD1",    "WNT"),
        ("WNT10B", "FZD1",    "WNT"),
        # ── TGF-β / BMP ──────────────────────────────────────────────────────
        ("TGFB1",  "TGFBR1",  "TGFb"),
        ("TGFB1",  "TGFBR2",  "TGFb"),
        ("TGFB2",  "TGFBR1",  "TGFb"),
        ("TGFB2",  "TGFBR2",  "TGFb"),
        ("TGFB3",  "TGFBR1",  "TGFb"),
        ("INHBA",  "ACVR1B",  "ACTIVIN"),
        ("INHBA",  "ACVR2A",  "ACTIVIN"),
        ("INHBA",  "ACVR2B",  "ACTIVIN"),
        ("BMP2",   "BMPR1A",  "BMP"),
        ("BMP2",   "BMPR2",   "BMP"),
        ("BMP4",   "BMPR1A",  "BMP"),
        ("BMP4",   "BMPR2",   "BMP"),
        ("BMP7",   "BMPR1B",  "BMP"),
        ("BMP7",   "BMPR2",   "BMP"),
        # ── HGF ──────────────────────────────────────────────────────────────
        ("HGF",    "MET",     "HGF"),
        # ── Angiopoietin ─────────────────────────────────────────────────────
        ("ANGPT1", "TEK",     "ANGPT"),
        ("ANGPT2", "TEK",     "ANGPT"),
        # ── Ephrin ───────────────────────────────────────────────────────────
        ("EFNA1",  "EPHA2",   "EPHRIN"),
        ("EFNA1",  "EPHA3",   "EPHRIN"),
        ("EFNB1",  "EPHB2",   "EPHRIN"),
        ("EFNB1",  "EPHB3",   "EPHRIN"),
        ("EFNB2",  "EPHB4",   "EPHRIN"),
        # ── Semaphorin ───────────────────────────────────────────────────────
        ("SEMA3A", "NRP1",    "SEMA"),
        ("SEMA3A", "NRP2",    "SEMA"),
        ("SEMA4D", "PLXNB1",  "SEMA"),
        # ── Hedgehog ─────────────────────────────────────────────────────────
        ("SHH",    "PTCH1",   "HH"),
        ("SHH",    "PTCH2",   "HH"),
        ("IHH",    "PTCH1",   "HH"),
        # ── Cytokines / Interleukins ─────────────────────────────────────────
        ("IL1A",   "IL1R1",   "IL1"),
        ("IL1B",   "IL1R1",   "IL1"),
        ("IL2",    "IL2RA",   "IL2"),
        ("IL2",    "IL2RB",   "IL2"),
        ("IL4",    "IL4R",    "IL4"),
        ("IL6",    "IL6R",    "IL6"),
        ("IL6",    "IL6ST",   "IL6"),
        ("IL10",   "IL10RA",  "IL10"),
        ("IL12A",  "IL12RB1", "IL12"),
        ("IL15",   "IL15RA",  "IL15"),
        ("IL17A",  "IL17RA",  "IL17"),
        ("IL18",   "IL18R1",  "IL18"),
        ("IL21",   "IL21R",   "IL21"),
        ("IFNA1",  "IFNAR1",  "IFN"),
        ("IFNG",   "IFNGR1",  "IFN"),
        ("IFNG",   "IFNGR2",  "IFN"),
        ("TNF",    "TNFRSF1A","TNF"),
        ("TNF",    "TNFRSF1B","TNF"),
        ("TNFSF10","TNFRSF10A","TRAIL"),
        ("TNFSF10","TNFRSF10B","TRAIL"),
        # ── Chemokines ───────────────────────────────────────────────────────
        ("CXCL1",  "CXCR2",   "CXCL"),
        ("CXCL5",  "CXCR2",   "CXCL"),
        ("CXCL8",  "CXCR1",   "CXCL"),
        ("CXCL8",  "CXCR2",   "CXCL"),
        ("CXCL9",  "CXCR3",   "CXCL"),
        ("CXCL10", "CXCR3",   "CXCL"),
        ("CXCL11", "CXCR3",   "CXCL"),
        ("CXCL12", "CXCR4",   "CXCL"),
        ("CXCL12", "ACKR3",   "CXCL"),
        ("CXCL16", "CXCR6",   "CXCL"),
        ("CCL2",   "CCR2",    "CCL"),
        ("CCL2",   "CCR4",    "CCL"),
        ("CCL3",   "CCR1",    "CCL"),
        ("CCL4",   "CCR1",    "CCL"),
        ("CCL5",   "CCR1",    "CCL"),
        ("CCL5",   "CCR3",    "CCL"),
        ("CCL5",   "CCR5",    "CCL"),
        ("CCL7",   "CCR2",    "CCL"),
        ("CCL11",  "CCR3",    "CCL"),
        ("CCL17",  "CCR4",    "CCL"),
        ("CCL20",  "CCR6",    "CCL"),
        ("CCL21",  "CCR7",    "CCL"),
        ("CCL22",  "CCR4",    "CCL"),
        # ── ECM / Integrins ──────────────────────────────────────────────────
        ("FN1",    "ITGA4",   "ECM"),
        ("FN1",    "ITGA5",   "ECM"),
        ("FN1",    "ITGAV",   "ECM"),
        ("FN1",    "ITGB1",   "ECM"),
        ("FN1",    "ITGB3",   "ECM"),
        ("FN1",    "CD44",    "ECM"),
        ("LAMA1",  "ITGA1",   "LAMININ"),
        ("LAMA1",  "ITGB1",   "LAMININ"),
        ("LAMA2",  "ITGA3",   "LAMININ"),
        ("LAMB1",  "ITGA6",   "LAMININ"),
        ("LAMC1",  "ITGB1",   "LAMININ"),
        ("COL1A1", "ITGA1",   "COLLAGEN"),
        ("COL1A1", "ITGA2",   "COLLAGEN"),
        ("COL1A1", "ITGB1",   "COLLAGEN"),
        ("COL4A1", "ITGA1",   "COLLAGEN"),
        ("COL4A2", "ITGA1",   "COLLAGEN"),
        ("SPP1",   "ITGAV",   "SPP1"),
        ("SPP1",   "ITGB1",   "SPP1"),
        ("SPP1",   "CD44",    "SPP1"),
        ("THBS1",  "CD36",    "THBS"),
        ("THBS1",  "ITGAV",   "THBS"),
        ("THBS2",  "CD36",    "THBS"),
        # ── TAM receptors (Tyro3/Axl/Mertk) ─────────────────────────────────
        ("PROS1",  "TYRO3",   "TAM"),
        ("PROS1",  "AXL",     "TAM"),
        ("PROS1",  "MERTK",   "TAM"),
        ("GAS6",   "TYRO3",   "TAM"),
        ("GAS6",   "AXL",     "TAM"),
        ("GAS6",   "MERTK",   "TAM"),
        # ── CD molecules / co-stimulatory ────────────────────────────────────
        ("CD274",  "PDCD1",   "PD-L1"),     # PD-L1 / PD-1
        ("PDCD1LG2","PDCD1",  "PD-L2"),
        ("CD80",   "CD28",    "CD28"),
        ("CD86",   "CD28",    "CD28"),
        ("CD80",   "CTLA4",   "CTLA4"),
        ("CD86",   "CTLA4",   "CTLA4"),
        ("CD40LG", "CD40",    "CD40"),
        ("TNFSF4", "TNFRSF4", "OX40"),
        ("FASLG",  "FAS",     "FAS"),
        # ── Gas6/Galectin ────────────────────────────────────────────────────
        ("LGALS1", "CD69",    "GALECTIN"),
        ("LGALS3", "LGALS3BP","GALECTIN"),
        ("LGALS9", "HAVCR2",  "GALECTIN"),
        # ── Neuropeptides / other ─────────────────────────────────────────────
        ("VIP",    "VIPR1",   "VIP"),
        ("VIP",    "VIPR2",   "VIP"),
        ("ADCYAP1","ADCYAP1R1","PACAP"),
        ("NTS",    "NTSR1",   "NTS"),
        # ── Complement ───────────────────────────────────────────────────────
        ("C3",     "C3AR1",   "COMPLEMENT"),
        ("C5",     "C5AR1",   "COMPLEMENT"),
        # ── MIF ──────────────────────────────────────────────────────────────
        ("MIF",    "CD74",    "MIF"),
        ("MIF",    "CXCR2",   "MIF"),
        ("MIF",    "CXCR4",   "MIF"),
    ]

    df = pd.DataFrame(pairs, columns=["ligand", "receptor", "pathway"])
    df = df.drop_duplicates().reset_index(drop=True)
    return df


if __name__ == "__main__":
    db = get_lr_database()
    print(f"LR database: {len(db)} pairs across {db['pathway'].nunique()} pathways")
    print(db.head(10).to_string())
