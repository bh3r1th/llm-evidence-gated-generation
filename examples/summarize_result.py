"""Example: call verify_answer then summarize the result.

# plug summary into your logger — EGA core does not log
"""

import json

from ega import summarize_result, verify_answer

result = verify_answer(
    claim="The Eiffel Tower is in Paris.",
    evidence=["The Eiffel Tower is a wrought-iron lattice tower in Paris, France."],
)

summary = summarize_result(result)
print(json.dumps(summary, indent=2))
