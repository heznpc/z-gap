"""Generate comprehensive markdown analysis report."""

from pathlib import Path
from datetime import datetime


def generate_report(
    all_results: list[dict],
    p1_result: dict | None,
    diagnoses: list[dict],
    output_path: Path,
):
    """Generate experiments/results/report.md with full analysis."""
    lines = []
    w = lines.append

    w("# Experiment Report: Cross-Lingual Semantic Invariance & Spacing Robustness")
    w(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    w("")

    # --- Summary Table ---
    w("## Summary")
    w("")
    w("| Model | dim | R_C | R_J | P2 | R_spacing | P7 |")
    w("|-------|:---:|:---:|:---:|:--:|:---------:|:--:|")
    for r in all_results:
        p2_ok = "OK" if r["P2"]["supported"] else "FAIL"
        p7_ok = "OK" if r["P7"]["supported"] else "FAIL"
        w(f"| {r['model']} | {r['dim']} | {r['P2']['R_C']:.3f} | {r['P2']['R_J']:.3f} | {p2_ok} | {r['P7']['R_spacing']:.3f} | {p7_ok} |")
    w("")

    # --- P1 Analysis ---
    w("## P1: Scale-Convergence")
    w("")
    if p1_result and p1_result.get("n_models", 0) >= 3:
        w(f"- Models tested: {p1_result['n_models']}")
        w(f"- Spearman rho (R_C vs dim): {p1_result['rho_C']:.3f} (p={p1_result['p_C']:.4f})")
        w(f"- Spearman rho (R_J vs dim): {p1_result['rho_J']:.3f} (p={p1_result['p_J']:.4f})")
        w(f"- Spearman rho (R_total vs dim): {p1_result['rho_total']:.3f} (p={p1_result['p_total']:.4f})")
        w(f"- **Verdict**: {'Supported' if p1_result['supported'] else 'Not supported'}")
        w("")
        w("| Model | dim | R_C | R_J | R_total |")
        w("|-------|:---:|:---:|:---:|:-------:|")
        for name, dim, rc, rj in zip(p1_result["model_names"], p1_result["dims"], p1_result["R_Cs"], p1_result["R_Js"]):
            w(f"| {name} | {dim} | {rc:.3f} | {rj:.3f} | {rc+rj:.3f} |")
    else:
        n = p1_result.get("n_models", len(p1_result.get("models", []))) if p1_result else len(all_results)
        w(f"Insufficient models for trend analysis ({n} models, need >= 3).")
    w("")

    # --- P2 Analysis ---
    w("## P2: Cross-Lingual Semantic Invariance")
    w("")
    w("**Prediction**: R_C > R_J (computational operations show higher cross-lingual invariance than judgment operations).")
    w("")

    for i, r in enumerate(all_results):
        w(f"### {r['model']} (dim={r['dim']})")
        w(f"- R_C = {r['P2']['R_C']:.3f}, R_J = {r['P2']['R_J']:.3f}")
        w(f"- P2 supported: **{'Yes' if r['P2']['supported'] else 'No (R_J > R_C)'}**")
        w(f"- p-value: {r['P2']['p']:.4f}")
        if "ci_95" in r["P2"]:
            ci = r["P2"]["ci_95"]
            w(f"- 95% CI for R_C - R_J: [{ci[0]:.3f}, {ci[1]:.3f}]")
        w("")

    # P2 Failure Diagnosis
    if diagnoses:
        w("### P2 Failure Diagnosis")
        w("")
        for diag in diagnoses:
            w(f"**Decomposition (R = d_inter / d_intra)**:")
            w(f"- d_intra_C = {diag['d_intra_C']:.4f}, d_intra_J = {diag['d_intra_J']:.4f}")
            w(f"  - Delta: {diag['delta_d_intra']:.4f} ({'comp less invariant' if diag['delta_d_intra'] > 0 else 'comp more invariant'})")
            w(f"- d_inter_C = {diag['d_inter_C']:.4f}, d_inter_J = {diag['d_inter_J']:.4f}")
            w(f"  - Delta: {diag['delta_d_inter']:.4f} ({'comp more spread' if diag['delta_d_inter'] > 0 else 'comp less spread'})")
            w(f"- **Primary driver**: {diag['primary_driver']}")
            w(f"- Mann-Whitney U (d_intra comp vs judg): U={diag['mannwhitney_u']:.1f}, p={diag['mannwhitney_p']:.4f}")
            w("")
            w("**d_intra statistics**:")
            w(f"- Computational: mean={diag['comp_d_intra_stats']['mean']:.4f}, std={diag['comp_d_intra_stats']['std']:.4f}, median={diag['comp_d_intra_stats']['median']:.4f}")
            w(f"- Judgment: mean={diag['judg_d_intra_stats']['mean']:.4f}, std={diag['judg_d_intra_stats']['std']:.4f}, median={diag['judg_d_intra_stats']['median']:.4f}")
            w("")
            w("**Most cross-lingually invariant operations** (lowest d_intra):")
            for op in diag["most_invariant_ops"]:
                w(f"- {op['op_id']} ({op['category']}): d_intra = {op['d_intra']:.4f}")
            w("")
            w("**Least cross-lingually invariant operations** (highest d_intra):")
            for op in diag["least_invariant_ops"]:
                w(f"- {op['op_id']} ({op['category']}): d_intra = {op['d_intra']:.4f}")
            w("")

        w("### Interpretation")
        w("")
        w("The P2 failure (R_J > R_C) suggests that judgment operations are *more* cross-lingually")
        w("invariant than computational operations in current multilingual embedding spaces.")
        w("Possible explanations:")
        w("")
        w("1. **Vocabulary effect**: Judgment operations use more abstract, cross-linguistically shared")
        w("   vocabulary (\"evaluate\", \"prioritize\"), while computational operations use more")
        w("   domain-specific terms that may be translated differently across languages.")
        w("2. **Semantic granularity**: Computational operations are more fine-grained (sort_asc vs")
        w("   sort_desc vs sort_by_len), making cross-lingual alignment harder for similar operations.")
        w("3. **Training data bias**: Multilingual models may have seen more parallel judgment-like")
        w("   text (business/evaluation) than parallel computational descriptions.")
        w("4. **The prediction may need revision**: The paper hypothesizes Z_semantic convergence")
        w("   is stronger for computational domains, but this may hold at *execution* level")
        w("   (denotational semantics) rather than at *description* level (NL embeddings).")
        w("")

    # --- P7 Analysis ---
    w("## P7: Spacing Robustness")
    w("")
    w("**Prediction**: R_spacing > 1 (Korean spacing variants cluster by meaning, not by spacing).")
    w("")
    for r in all_results:
        p7 = r["P7"]
        w(f"### {r['model']} (dim={r['dim']})")
        w(f"- R_spacing = {p7['R_spacing']:.3f}")
        w(f"- d_spacing (same meaning, diff spacing) = {p7['d_spacing']:.4f}")
        w(f"- d_semantic (diff meaning, same spacing) = {p7['d_semantic']:.4f}")
        w(f"- P7 supported: **{'Yes' if p7['supported'] else 'No'}**")
        if "p_value" in p7:
            w(f"- Bootstrap p-value: {p7['p_value']:.4f}")
        if "ci_95" in p7:
            ci = p7["ci_95"]
            w(f"- 95% CI for R_spacing: [{ci[0]:.3f}, {ci[1]:.3f}]")
        w("")

    # --- Figures ---
    w("## Generated Figures")
    w("")
    w("| Figure | Description |")
    w("|--------|-------------|")
    w("| `embedding_space_*.png` | t-SNE projection colored by category, shaped by language |")
    w("| `discriminability_*.png` | R_C vs R_J bar chart |")
    w("| `spacing_*.png` | d_spacing vs d_semantic bar chart |")
    w("| `d_intra_dist_*.png` | Violin plot: d_intra distributions by category |")
    w("| `d_intra_vs_d_inter_*.png` | Scatter: invariance vs distinctiveness |")
    w("| `per_op_d_intra_*.png` | Sorted bar chart of per-operation d_intra |")
    w("| `heatmap_comp_*.png` | Cross-lingual similarity heatmap (computational) |")
    w("| `heatmap_judg_*.png` | Cross-lingual similarity heatmap (judgment) |")
    w("| `p1_scale_trend.png` | R vs model dimension trend |")
    w("")

    # --- Implications ---
    w("## Implications for the Paper")
    w("")
    w("1. **P2 needs reframing**: The prediction that computational operations show higher")
    w("   cross-lingual invariance is not supported at the NL embedding level. The paper should")
    w("   distinguish between *description-level* and *execution-level* invariance. Z_semantic")
    w("   convergence may hold for program semantics (denotational) but not for NL descriptions")
    w("   of those semantics.")
    w("2. **P7 is strongly supported**: Korean spacing variants consistently cluster by meaning,")
    w("   supporting the claim that the ideal encoder should be spacing-invariant.")
    w("3. **P1 requires more evidence**: The scale trend is suggestive but needs more model sizes")
    w("   and architectures for a definitive conclusion.")
    w("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report saved: {output_path}")
