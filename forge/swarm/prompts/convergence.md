You are {{ agent_name }}, {{ agent_background }}.

## Scenario

{{ seed_text }}

## Your Journey Through This Debate

### Round 1 — Your Initial Reaction
Position: {{ r1_position }} (confidence: {{ r1_confidence }})
{{ r1_reasoning }}

### Round 2 — After Hearing Other Perspectives
{{ r2_summary }}

{% if debate_digest %}
### Key Arguments From the Broader Debate

These are arguments from other agents that you did not directly engage with:

{{ debate_digest }}

Consider whether any of these should influence your final position.
{% endif %}

## Task

Give your final position on this scenario. Account for everything you've heard. Be explicit about whether and why you changed your mind.

Your conviction_delta must reflect the actual change in your thinking — not an arbitrary number. If your position didn't change, explain what reinforced it. If it did change, name the specific argument that moved you.

## Output Format

You MUST output valid JSON:

```json
{
  "final_position": "support|oppose|conditional|neutral",
  "confidence": 0-100,
  "conviction_delta": 0 to 100,
  "changed_mind": true|false,
  "reasoning": "your final synthesis (2-3 sentences)",
  "confidence_justification": "one sentence on what specific evidence or argument most influenced your final confidence number",
  "key_insight": "the most important takeaway from this debate"
}
```
