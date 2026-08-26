"""Export a MoBayes knowledge base to a static JSON asset for the browser engine.

Reuses the published loader so the Dirichlet math cannot diverge: it dumps the
expected likelihoods E[P(v|d)] = alpha_v / sum(alpha) that
DirichletLikelihoodModel already precomputes. The TS engine therefore needs no
counts, no Laplace smoothing, and no Dirichlet code.

Usage:
    PYTHONPATH=. python3 webapp/scripts/export_kb.py data/kbs/ddxplus
    PYTHONPATH=. python3 webapp/scripts/export_kb.py data/kbs/llmkb_gpt --prior-mode uniform
"""

import argparse
import json
from pathlib import Path

import world_model.knowledge.data_loader as dl


def patch_loader():
    """Fix two upstream bugs that silently drop likelihood counts.

    1. features.csv parses "[0, 1, ... 10]" as ints while likelihood_counts.csv
       yields strings, so the `val in alpha_table[fid]` guard never matches and
       alpha stays at 1 for every 0-10 scale feature. Stringifying schema values
       makes both sides agree.
    2. llmkb_* spell the type "numerical", which is absent from the
       ['categorical', 'binary'] list that gates alpha_table construction, so
       those features get no likelihoods at all.
    """
    orig_parse_values = dl.parse_values
    dl.parse_values = lambda s: [str(v) for v in orig_parse_values(s)]

    orig_schema = dl.FeatureSchema

    def schema(feature_id, ftype, values, metadata=None):
        return orig_schema(feature_id, "categorical" if ftype == "numerical" else ftype,
                           values, metadata)

    dl.FeatureSchema = schema


def export(kb_dir: Path, prior_mode: str) -> dict:
    wm, diseases = dl.load_world_model_from_schema(str(kb_dir), prior_mode=prior_mode)
    names = dl.load_disease_names(str(kb_dir))
    lm = wm.likelihood_model

    return {
        "id": kb_dir.name,
        "priorMode": prior_mode,
        "diseases": diseases,
        "names": [names.get(d, d) for d in diseases],
        "prior": wm.base_prior.tolist(),
        "features": [
            {
                "id": fid,
                "name": s.metadata.get("name", fid),
                "type": s.ftype,
                "values": [str(v) for v in s.values],
            }
            for fid, s in lm.schema.items()
        ],
        "likelihood": {
            fid: {str(v): vec.tolist() for v, vec in by_value.items()}
            for fid, by_value in lm.expected_likelihood.items()
        },
        "parents": wm.taxonomy.mapping,
        "roots": wm.taxonomy.roots,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kb_dir", type=Path)
    ap.add_argument("--prior-mode", default="empirical",
                    choices=["empirical", "uniform", "conditional"])
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "public" / "kb")
    args = ap.parse_args()

    patch_loader()
    kb = export(args.kb_dir, args.prior_mode)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{kb['id']}.json"
    out.write_text(json.dumps(kb, separators=(",", ":")))

    n_pairs = sum(len(v) for v in kb["likelihood"].values())
    print(f"{out}  {out.stat().st_size / 1e6:.2f} MB")
    print(f"  {len(kb['diseases'])} diseases, {len(kb['features'])} features, "
          f"{n_pairs} value-pairs, {len(kb['roots'])} roots")


if __name__ == "__main__":
    main()
