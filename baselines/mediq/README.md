# MediQ-Expert baseline (setup)

[MediQ](https://github.com/stellalisy/mediQ) (Li et al., NeurIPS 2024).
The bench row `baselines/mediq/row.py` drives `mediQ_benchmark.py` as a
subprocess. MediQ's own patient is replaced by the same simulator every
other row faces, via the modules in `patients/` (added to the subprocess
`PYTHONPATH` by the row):

| Module | Class | Columns |
|---|---|---|
| `patients/mobayes_patient_patientsim.py` | `BMBEPatientSim` | `patientsim`, `patientsim_v1` |
| `patients/mobayes_patient_ddxplus_htsim.py` | `DDxHTSimMediQPatient` | `htsim`, `htsim_v1` |
| `patients/mobayes_patient_agentclinic.py` | `ACMediQPatient` | `ac`, `ac_set2` |

## Clone + venv

```bash
git clone https://github.com/stellalisy/mediQ ~/Documents/GitHub/mediQ
cd ~/Documents/GitHub/mediQ
uv venv --python 3.12 && uv pip install openai==1.57.4 tqdm pyyaml
export MEDIQ_REPO=~/Documents/GitHub/mediQ
```

The full MediQ conda env pins ~120 packages including vLLM, CUDA,
transformers, xformers. We only use the OpenAI-API code path, so the
minimal deps above are sufficient.

## Required patches to the upstream mediQ repo

Apply these five small patches in the cloned `mediQ/` checkout. None of
them are upstreamable as-is (they encode our specific deployment
choices), but each is needed for MediQ to run against current OpenAI
SDKs and GPT-5.x reasoning models.

1. **`src/keys.py` — read API key from env.** Replace the hardcoded
   placeholder `mykey = {"mediQ": "sk-1234..."}` with
   `os.environ.get("OPENAI_API_KEY", "")` so the cloned repo never
   carries a key.

2. **`src/helper.py` — make `torch` import optional.** Wrap the
   top-level `import torch` in `try/except ImportError`. The OpenAI-API
   path does not need torch / vLLM / transformers / CUDA, but the
   unconditional import blocks startup.

3. **`src/helper.py` — GPT-5.x parameter compatibility.** GPT-5.x
   reasoning models reject `max_tokens` (require `max_completion_tokens`)
   and reject `temperature` / `top_p`. Route by model-prefix:

   ```python
   is_gpt5x = self.model_name.startswith(("gpt-5", "o"))
   if is_gpt5x:
       kwargs["max_completion_tokens"] = self.max_tokens
   else:
       kwargs.update(temperature=self.temperature,
                     max_tokens=self.max_tokens, top_p=self.top_p)
   ```

4. **`src/helper.py` — OpenAI SDK v1 + chat-completions API.** Two
   bugs in the post-call extraction: dict access on a Pydantic object
   (`response["usage"]["prompt_tokens"]`) and legacy completion-endpoint
   path (`response.choices[0].text`). Replace with attribute access and
   `response.choices[0].message.content`.

5. **`src/helper.py` — missing `return` on OpenAI path.** Line ~68:
   the OpenAI branch silently dropped its return value, so `generate()`
   returned `None` and downstream `expert_response_*` raised
   `TypeError: cannot unpack non-iterable NoneType object`. Add the
   missing `return`.

## Input format

The row converts the column's cases into MediQ's iMEDQA-style MCQ JSONL
(`<MEDIQ_REPO>/data/<column>_<start>_<end>_mediq.jsonl`). The options are
the column's disease list (diagnoses present in the selected cases first);
each row carries an `hadm_id` that the patient module uses to look up the
case. Raw MediQ logs land in `<MEDIQ_REPO>/logs/<column>_<start>_<end>/`.
The logs behind the paper tables are in
`data/benchmark_results/set_{a,b}/<benchmark>/baselines/mediq_raw/` (the HT-sim
set-a run is stored as `set_a/ddxplus_htsim/baselines/mediq_expert_gpt54nano*.jsonl`).

## Running

Once the patches are in place, MediQ runs through the unified driver:

```bash
PYTHONPATH=. python -m bench --row mediq --column {patientsim,htsim,ac}
```
