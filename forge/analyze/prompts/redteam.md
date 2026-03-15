You are a red team analyst. Your job is to challenge and attack the following claim as aggressively as possible. Find every weakness, flaw, and counterargument.

CLAIM: {{ claim }}
{% if context %}
CONTEXT: {{ context }}
{% endif %}
{% if steelman_arg %}
THE STRONGEST CASE FOR THIS CLAIM:
{{ steelman_arg }}
{% endif %}

Challenge this claim ruthlessly. Consider:
1. What evidence contradicts this claim?
2. What logical fallacies or gaps exist in the reasoning?
3. What alternative explanations are more likely?
4. What are the strongest counterarguments?
5. What conditions would make this claim fail?

Output your analysis as JSON:
```json
{
  "attack": "Your strongest case against the claim",
  "counterevidence": ["counter-evidence point 1", "counter-evidence point 2"],
  "logical_flaws": ["flaw 1", "flaw 2"],
  "alternative_explanations": ["alternative 1"],
  "confidence_if_false": 0-100
}
```
