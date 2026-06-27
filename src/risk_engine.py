import re
import pandas as pd

def to_snake_case(text):
    text = text.lower()
    text = text.replace("/", "_")
    text = text.replace("-", "_")
    text = text.replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]", "", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")

def score_contract(contract_row,risk_config):
  risk_score=0
  risk_drivers=[]

  for _, config in risk_config.iterrows():
    clause_name=config["clause_name"]
    clause_col=to_snake_case(clause_name)
    clause_role=config["clause_role"]
    weight=config["base_risk_weight"]

    if clause_col not in contract_row.index:
      continue
    clause_present=bool(contract_row[clause_col])

    triggered=False

    if clause_role=="Risk if Present" and clause_present:
      triggered=True

    elif clause_role=="Risk if Absent" and not clause_present:
      triggered=True

    if triggered:
      risk_score+=weight
      risk_drivers.append({
          "clause_name":clause_name,
          "risk_domain":config["risk_domain"],
          "clause_role":clause_role,
          "severity":config["severity"],
          "risk_weight":weight,
          "business_description":config["business_description"]

      })

      risk_score=min(risk_score,100)




  return risk_score,risk_drivers

def assign_risk_band(score):

  if score<=25:
    return "Low Risk"
  elif score<=50:
    return "Moderate Risk"
  elif score<=75:
    return "High Risk"
  else:
    return "Critical Risk"

recommendation_map = {
    "Uncapped Liability": "Review liability exposure and consider adding or narrowing liability caps.",
    "Cap On Liability": "Consider adding a liability cap to limit financial exposure.",
    "Insurance": "Consider adding insurance obligations to protect against operational or financial losses.",
    "Non-Compete": "Review competitive restrictions for enforceability and business flexibility.",
    "Exclusivity": "Review exclusivity obligations and assess whether they restrict future business opportunities.",
    "No-Solicit Of Customers": "Review customer non-solicitation restrictions for scope and duration.",
    "No-Solicit Of Employees": "Review employee non-solicitation restrictions for scope and duration.",
    "Termination For Convenience": "Consider adding a termination-for-convenience right for flexibility.",
    "Post-Termination Services": "Review post-termination obligations and estimate operational burden.",
    "Minimum Commitment": "Review minimum purchase or payment obligations.",
    "Revenue/Profit Sharing": "Review revenue or profit sharing obligations and financial impact.",
    "Price Restrictions": "Review restrictions on pricing flexibility.",
    "Volume Restriction": "Review usage or volume thresholds and related penalties.",
    "Ip Ownership Assignment": "Review intellectual property ownership transfer language carefully.",
    "Joint Ip Ownership": "Clarify rights and responsibilities for jointly owned intellectual property.",
    "Irrevocable Or Perpetual License": "Review long-term or irrevocable license rights for future business impact.",
    "Unlimited/All-You-Can-Eat-License": "Review broad license grants and usage limitations.",
    "Liquidated Damages": "Review preset damages or termination fees for financial exposure."
}

def build_risk_driver_table(risk_drivers, recommendation_map=None):
    risk_driver_table = pd.DataFrame(risk_drivers)

    if risk_driver_table.empty:
        return risk_driver_table

    if recommendation_map:
        risk_driver_table["recommendation"] = (
            risk_driver_table["clause_name"]
            .map(recommendation_map)
            .fillna("Review this clause with legal or business stakeholders.")
        )

    return risk_driver_table