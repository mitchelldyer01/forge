You are analyzing an article to extract specific, falsifiable claims.

{% if title %}ARTICLE TITLE: {{ title }}{% endif %}

{% if content %}ARTICLE CONTENT:
{{ content }}{% endif %}

Extract 3-5 specific, falsifiable predictions or claims from this article. Each claim must:
1. Be a concrete statement about what will or could happen
2. Have a suggested timeframe for verification
3. Be specific enough to be judged true or false

Output JSON:
{
  "claims": [
    {
      "claim": "specific falsifiable claim",
      "confidence": 50,
      "tags": ["topic1", "topic2"],
      "resolution_deadline": "YYYY-MM-DD or null if no clear timeframe"
    }
  ]
}

If the article contains no extractable claims, return {"claims": []}.
Confidence should reflect how likely the claim is to be true (0-100).
Tags should be 1-3 lowercase topic labels.
