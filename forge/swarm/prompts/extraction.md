You are analyzing the output of a multi-agent simulation about:

{{ seed_text }}

## Simulation Results

{{ consensus_report }}

## Task

Extract specific, falsifiable predictions from this simulation. Each prediction must:
1. Be a concrete claim about what will happen
2. Have a timeframe (when could this be verified?)
3. Have clear resolution criteria (how do we know if it's true or false?)

Only extract predictions that have meaningful support from the simulation. Do not fabricate predictions that were not discussed.

## Output Format

You MUST output valid JSON:

```json
{
  "predictions": [
    {
      "claim": "specific, falsifiable prediction",
      "confidence": 0-100,
      "consensus_strength": 0.0-1.0,
      "resolution_deadline": "ISO 8601 date",
      "dissent_summary": "what the minority argued",
      "tags": ["topic1", "topic2"]
    }
  ]
}
```

If no clear predictions can be extracted, return `{"predictions": []}`.
