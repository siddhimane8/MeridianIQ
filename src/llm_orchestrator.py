import json
import os


def build_gemini_prompt(payload):
    compact_payload = json.dumps(payload, indent=2)[:14000]

    prompt = f"""
You are MeridianIQ, an AI-powered contract risk intelligence assistant.

Your task is to generate an executive-level contract review report using ONLY the structured data provided below.

Important Rules:
- Do not invent clauses, dates, parties, obligations, risks, or legal interpretations.
- Use only the detected clauses, risk drivers, recommendations, and evidence provided.
- Do not provide legal advice.
- If the evidence is weak, incomplete, or indirect, clearly say so.
- Write for a business decision-maker, not a lawyer.
- Keep the tone professional, concise, and decision-focused.

Required Report Sections:
1. Executive Summary
2. Overall Risk Assessment
3. Business Impact
4. Top Contract Risks
5. Evidence Highlights
6. Recommended Review Actions
7. Confidence Notes
8. Disclaimer

Structured Contract Data:
{compact_payload}
"""
    return prompt


# def generate_gemini_report(prompt, api_key, model_name="gemini-2.5-flash"):
#     from google import genai

#     client = genai.Client(api_key=api_key)

#     response = client.models.generate_content(
#         model=model_name,
#         contents=prompt
#     )

#     return response.text

def generate_gemini_report(prompt, api_key, model_name="gemini-2.5-flash"):
    import google.genai as genai

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model_name,
        contents=prompt
    )

    return response.text