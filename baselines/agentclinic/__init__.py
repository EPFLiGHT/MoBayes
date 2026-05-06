"""AgentClinic-50 baseline pipeline.

Builds a 50-case (~28-disease) closed-world benchmark from the AgentClinic-MedQA
Extended dataset (Schmidgall et al., 2024) and the corresponding LLM-generated
KB sized for BMBE's Bayesian engine.

Steps live in ordered modules:
    sample_cases.py        — frequency-weighted sampling of 50 cases
    prepare_kb_input.py    — convert disease list to llmkb input format
    prompts_discriminative.py — overrides the default llmkb prompts to push
                                 features that are discriminative across diseases
                                 (key for posterior sharpening in Bayesian update)
    build_kb.sh            — chains existing scripts/data_prep/llmkb pipeline
    validate_kb.py         — IG / sparsity / coverage diagnostics
"""
