import os
import csv
import json
import ast
import anthropic
from dotenv import load_dotenv

load_dotenv()

CATEGORIES = ["Nursery", "SEN School", "Food Store", "None"]

SYSTEM_PROMPT = """You classify commercial property listings for an acquisitions company that buys properties for three sectors only:

1. Nursery — childcare or early years use. Signals: existing nursery, D1/E(f) use class, outdoor space, OFSTED references, childcare or early years language.

2. SEN School — special educational needs provision. Signals: existing school use, D1 use class, large floor plates, SEN/SEMH/alternative provision references, therapeutic or accessible settings.

3. Food Store — food retail. Signals: existing convenience or supermarket use, A1/E use class, high footfall, chiller infrastructure, references to Co-op, Tesco, Lidl, Aldi or similar operators.

4. None — does not fit any of the above.

Rules:
- Read the description carefully. Use class alone is not enough.
- "Suitable for a variety of alternative uses" is a common hedge. Pick the strongest positive signal and reflect the uncertainty in your confidence level.
- If genuinely ambiguous, pick the most likely category and set confidence to Low or Medium. Do not force a High confidence answer.
- Respond with valid JSON only. No explanation outside the JSON.

Return exactly this structure:
{
  "category": "Nursery" | "SEN School" | "Food Store" | "None",
  "confidence": "High" | "Medium" | "Low",
  "reasoning": "One or two sentences explaining the decision."
}"""


def parse_key_features(raw: str) -> str:
    """Parse stringified list from keyFeatures column into readable text."""
    if not raw or not raw.strip():
        return ""
    try:
        items = ast.literal_eval(raw)
        if isinstance(items, list):
            return "; ".join(str(i) for i in items)
    except Exception:
        pass
    return raw.strip()


def build_listing_text(row: dict) -> str:
    """Extract and combine the most useful fields for classification."""
    parts = []

    if row.get("summary", "").strip():
        parts.append(f"Summary: {row['summary'].strip()}")

    if row.get("detailedDescription", "").strip():
        parts.append(f"Description: {row['detailedDescription'].strip()}")

    features = parse_key_features(row.get("keyFeatures", ""))
    if features:
        parts.append(f"Key Features: {features}")

    if row.get("useClass", "").strip():
        parts.append(f"Use Class: {row['useClass'].strip()}")

    if row.get("propertySubType", "").strip():
        parts.append(f"Property Type: {row['propertySubType'].strip()}")

    if row.get("keyword", "").strip():
        parts.append(f"Keyword: {row['keyword'].strip()}")

    if not parts:
        return "No usable listing data available."

    return "\n".join(parts)


def classify_listing(client: anthropic.Anthropic, listing_text: str) -> dict:
    """Send listing text to Claude and return structured classification."""
    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Classify this property listing:\n\n{listing_text}"
                }
            ]
        )

        raw = response.content[0].text.strip()
        result = json.loads(raw)

        # Validate fields
        if result.get("category") not in CATEGORIES:
            result["category"] = "None"
        if result.get("confidence") not in ["High", "Medium", "Low"]:
            result["confidence"] = "Low"
        if not result.get("reasoning"):
            result["reasoning"] = "No reasoning provided."

        return result

    except json.JSONDecodeError:
        return {
            "category": "None",
            "confidence": "Low",
            "reasoning": "Failed to parse LLM response as JSON."
        }
    except Exception as e:
        return {
            "category": "None",
            "confidence": "Low",
            "reasoning": f"Classification error: {str(e)}"
        }


def main():
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    input_file = "listings.csv"
    output_file = "results.csv"

    with open(input_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    output_fieldnames = fieldnames + ["predicted_category", "confidence", "reasoning"]

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()

        for i, row in enumerate(rows):
            listing_id = row.get("id", f"row-{i+1}")
            listing_text = build_listing_text(row)

            print(f"[{i+1}/{len(rows)}] Classifying {listing_id}...")

            result = classify_listing(client, listing_text)

            row["predicted_category"] = result["category"]
            row["confidence"] = result["confidence"]
            row["reasoning"] = result["reasoning"]

            writer.writerow(row)

            print(f"  → {result['category']} ({result['confidence']}): {result['reasoning']}")

    print(f"\nDone. Results written to {output_file}")


if __name__ == "__main__":
    main()