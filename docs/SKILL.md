# EGA SKILL — Evidence-Gated Answering

WHEN TO USE
- When LLM outputs must be grounded in source evidence
- When hallucinations are unacceptable
- When partial answers are acceptable but wrong answers are not

LOOP

1) GENERATE
Produce answer using LLM

2) DECOMPOSE
Break answer into independent units (claims)

3) VERIFY
For each unit:
  - find supporting evidence
  - score entailment/contradiction

4) DECIDE
For each unit:
  - keep if supported
  - drop if unsupported
  - abstain if uncertain

5) CORRECT (OPTIONAL, BOUNDED)
- Regenerate ONLY failed units
- Re-verify
- Max retries: 1–2
- If still failing → drop

6) STOP
Return:
- verified units
- reconstructed answer from kept units
- or abstain if nothing valid

STRICT RULES
- Never trust unverified units
- Never retry full answer
- Never loop indefinitely
- Prefer partial truth over full hallucination

---

MAPPING TO PACKAGE

generate → external LLM  
decompose → unitize_answer  
verify → verifier interface  
decide → Enforcer  
correct → correction loop  
stop → verify_answer()
