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

Give your final position on this scenario. You should NOT change your position unless a SPECIFIC argument from the debate directly refutes your core reasoning. Hearing good arguments from the other side is not enough — they must actually undermine your key claim. If no argument directly refuted your reasoning, hold your position and explain what reinforced it.

Before finalizing, consider: what is the strongest argument AGAINST your current position? If you cannot articulate one, your confidence may be too high.

If you are changing position, you MUST name the specific agent and argument that changed your mind. If you cannot point to one, you should not change.

`changed_mind` means your POSITION changed (e.g., support → oppose, conditional → support). A shift in confidence alone is NOT changing your mind. If your Round 1 position was "conditional" and your final position is still "conditional", `changed_mind` must be false.

Your reasoning must include at least one specific claim that is unique to YOUR perspective — a named precedent, a concrete mechanism, a quantitative estimate, or a falsifiable consequence. Do not simply summarize the debate or repeat common themes.

## Confidence

Your confidence should reflect YOUR specific expertise and the evidence encountered.
Do NOT default to 70-80. Use the full range.

- Confidence anchor: {{ confidence_anchor }}

If your confidence moved significantly from Round 1, explain what specific argument caused the shift.

## Output Format

You MUST output valid JSON:

```json
{
  "final_position": "support|oppose|conditional|neutral",
  "confidence": 0-100,
  "conviction_delta": "absolute change in confidence from Round 1 (e.g., if R1=65 and now 75, delta is 10)",
  "changed_mind": true|false,
  "reasoning": "your final synthesis (2-3 sentences)",
  "confidence_justification": "name the strongest argument AGAINST your position and explain why it did not change your mind (or did)",
  "key_insight": "the most important takeaway from this debate"
}
```
