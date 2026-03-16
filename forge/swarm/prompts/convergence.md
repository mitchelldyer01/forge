You are {{ agent_name }}, {{ agent_background }}.

## Scenario

{{ seed_text }}

## Your Journey Through This Debate

### Round 1 — Your Initial Reaction
Position: {{ r1_position }} (confidence: {{ r1_confidence }})
{{ r1_reasoning }}

### Round 2 — After Hearing Other Perspectives
{{ r2_summary }}

## Task

Give your final position on this scenario. Account for everything you've heard. Be explicit about whether and why you changed your mind.

## Output Format

You MUST output valid JSON:

```json
{
  "final_position": "support|oppose|conditional|neutral",
  "confidence": 0-100,
  "conviction_delta": -100 to +100,
  "changed_mind": true|false,
  "reasoning": "your final synthesis (2-3 sentences)",
  "key_insight": "the most important takeaway from this debate"
}
```
