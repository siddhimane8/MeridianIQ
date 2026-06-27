import sys
import json
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path
from datetime import datetime, timezone
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# Path Setup
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

from text_processing import clean_contract_text, chunk_contract_text
from clause_detection import predict_clauses
from evidence_retrieval import retrieve_evidence
from risk_engine import score_contract, assign_risk_band, build_risk_driver_table
from report_generator import build_report_payload
from llm_orchestrator import generate_gemini_report


# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(
    page_title="MeridianIQ",
    page_icon="🧭",
    layout="wide"
)


# --------------------------------------------------
# Cached Loaders
# --------------------------------------------------

@st.cache_resource
def load_models():
    tfidf = joblib.load(ROOT / "models" / "tfidf_vectorizer.pkl")
    clause_detector = joblib.load(ROOT / "models" / "baseline_clause_detector.pkl")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return tfidf, clause_detector, embedding_model


@st.cache_data
def load_risk_config():
    risk_config = pd.read_csv(ROOT / "data" / "processed" / "clause_risk_config.csv")
    return risk_config[risk_config["is_mvp_clause"] == True].copy()


# --------------------------------------------------
# Product Helpers
# --------------------------------------------------

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


def health_emoji(risk_band):
    if risk_band == "Low Risk":
        return "🟢"
    if risk_band == "Moderate Risk":
        return "🟡"
    if risk_band == "High Risk":
        return "🟠"
    return "🔴"


def manual_review_status(risk_band, risk_driver_count):
    if risk_band in ["High Risk", "Critical Risk"] or risk_driver_count >= 3:
        return "Yes"
    if risk_band == "Moderate Risk":
        return "Recommended"
    return "No"


def executive_health_message(risk_band):
    if risk_band == "Low Risk":
        return "This contract appears to have a lower risk profile based on MeridianIQ's configured review rules."
    if risk_band == "Moderate Risk":
        return "This contract contains some terms that should be reviewed before signing."
    if risk_band == "High Risk":
        return "This contract contains several terms that may expose the business to financial, operational, or legal obligations."
    return "This contract contains serious risk signals and should be reviewed carefully before signing."


def business_clause_label(clause):
    mapping = {
        "Uncapped Liability": "Unlimited liability exposure",
        "Cap On Liability": "Liability limit found",
        "Insurance": "Insurance obligations found",
        "Non-Compete": "Competition restriction found",
        "Exclusivity": "Exclusivity obligation found",
        "No-Solicit Of Customers": "Customer non-solicitation restriction found",
        "No-Solicit Of Employees": "Employee non-solicitation restriction found",
        "Termination For Convenience": "Termination flexibility found",
        "Post-Termination Services": "Post-termination obligations found",
        "Minimum Commitment": "Minimum business commitment found",
        "Revenue/Profit Sharing": "Revenue or profit sharing found",
        "Price Restrictions": "Pricing restriction found",
        "Volume Restriction": "Volume-based restriction found",
        "Ip Ownership Assignment": "IP ownership assignment found",
        "Joint Ip Ownership": "Joint IP ownership found",
        "License Grant": "License rights found",
        "Irrevocable Or Perpetual License": "Long-term or irrevocable license found",
        "Unlimited/All-You-Can-Eat-License": "Broad usage license found",
        "Audit Rights": "Audit rights found",
        "Warranty Duration": "Warranty period found",
        "Liquidated Damages": "Preset damages or penalty terms found",
        "Anti-Assignment": "Assignment restriction found",
        "Change Of Control": "Change of control provision found",
    }
    return mapping.get(clause, clause)


def build_business_report(
    filename,
    risk_score,
    risk_band,
    detected_present,
    risk_driver_table,
    evidence_df,
    review_status
):
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = []

    lines.append("# MeridianIQ Contract Review Report")
    lines.append("")
    lines.append(f"**Generated At:** {generated_at}")
    lines.append(f"**Contract File:** {filename}")
    lines.append("")
    lines.append("## 1. Executive Summary")
    lines.append("")
    lines.append(
        f"This contract has been assessed as **{risk_band}** with a contract health score of **{int(risk_score)}/100**."
    )
    lines.append(executive_health_message(risk_band))
    lines.append("")
    lines.append(f"MeridianIQ found **{len(detected_present)} important clause(s)** and **{len(risk_driver_table)} review item(s)**.")
    lines.append("")
    lines.append("## 2. Recommended Review Actions")
    lines.append("")

    if risk_driver_table.empty:
        lines.append("No major score-changing review items were found based on MeridianIQ's current rules.")
    else:
        for _, row in risk_driver_table.iterrows():
            clause = business_clause_label(row.get("clause_name", "Review Item"))
            reason = row.get("business_description", "This clause may affect business risk.")
            action = row.get("recommendation", "Review this clause with legal or business stakeholders.")
            area = row.get("risk_domain", "General Risk")
            severity = row.get("severity", "Review")

            lines.append(f"### {clause}")
            lines.append("")
            lines.append(f"**Why this matters:** {reason}")
            lines.append("")
            lines.append(f"**Suggested action:** {action}")
            lines.append("")
            lines.append(f"**Risk area:** {area}  ")
            lines.append(f"**Severity:** {severity}")
            lines.append("")

    lines.append("## 3. Key Clauses Found")
    lines.append("")

    if detected_present.empty:
        lines.append("No target business clauses were detected.")
    else:
        for clause in detected_present["clause_name"].tolist():
            lines.append(f"- {business_clause_label(clause)}")

    lines.append("")
    lines.append("## 4. Supporting Contract Evidence")
    lines.append("")

    if evidence_df.empty:
        lines.append("No supporting evidence passages were retrieved.")
    else:
        for clause_name in evidence_df["clause_name"].unique():
            clause_evidence = (
                evidence_df[evidence_df["clause_name"] == clause_name]
                .sort_values("rank")
                .head(1)
            )

            lines.append(f"### {business_clause_label(clause_name)}")
            lines.append("")
            for _, row in clause_evidence.iterrows():
                evidence_text = str(row["evidence_text"]).strip()
                lines.append("MeridianIQ found this contract section relevant:")
                lines.append("")
                lines.append(f"> {evidence_text[:1200]}...")
                lines.append("")

    lines.append("## 5. MeridianIQ Decision")
    lines.append("")

    if review_status == "No":
        lines.append("No major manual review trigger was found based on MeridianIQ's current rules.")
    else:
        lines.append(
            f"Manual review is recommended because MeridianIQ found {len(risk_driver_table)} review item(s) in this contract."
        )

    lines.append("")
    lines.append("## 6. Disclaimer")
    lines.append("")
    lines.append(
        "This report is generated by MeridianIQ for business review support. It does not constitute legal advice. "
        "Legal or business stakeholders should review important contracts before signing."
    )

    return "\n".join(lines)


def build_clean_gemini_prompt(business_report):
    return f"""
You are MeridianIQ, a business-facing contract review assistant.

Rewrite the following structured contract review into a clear, executive-style report.

Rules:
- Do not mention similarity scores.
- Do not mention evidence ranks.
- Do not mention model internals, embeddings, chunks, or debug details.
- Keep the language simple and business-friendly.
- Keep the report structured with headings.
- Make it readable for a non-technical business decision-maker.
- Do not provide legal advice.
- Include a short disclaimer.

Structured review:
{business_report}
"""


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🧭 MeridianIQ")
st.subheader("Hybrid Clause Extraction and LLM Orchestration for Contract Risk Intelligence")

st.markdown(
    """
    MeridianIQ reviews contract text files and turns them into simple business-facing risk insights.

    Upload a `.txt` contract to detect important clauses, find supporting evidence, assess contract health,
    and generate an executive-style review report.
    """
)

st.divider()


# --------------------------------------------------
# Load Assets
# --------------------------------------------------

try:
    tfidf, clause_detector, embedding_model = load_models()
    mvp_config = load_risk_config()
except Exception as e:
    st.error("MeridianIQ could not load required models or configuration files.")
    st.exception(e)
    st.stop()


# --------------------------------------------------
# Upload Section
# --------------------------------------------------

st.header("📄 Upload Contract")

uploaded_file = st.file_uploader(
    "Upload a contract text file",
    type=["txt"],
    key="contract_uploader"
)

if uploaded_file is None:
    st.info("Upload a `.txt` contract file to begin.")
    st.stop()

st.success(f"Uploaded: {uploaded_file.name}")

analyze_button = st.button(
    "Analyze Contract",
    type="primary",
    key="analyze_button"
)


# --------------------------------------------------
# Pipeline
# --------------------------------------------------

if analyze_button:
    try:
        with st.status("Running MeridianIQ analysis...", expanded=True) as status:
            st.write("Reading contract...")
            raw_text = uploaded_file.getvalue().decode("utf-8", errors="ignore")

            st.write("Cleaning contract text...")
            clean_text = clean_contract_text(raw_text)

            st.write("Splitting contract into readable sections...")
            chunks_df = chunk_contract_text(clean_text, max_chars=1500)

            st.write("Detecting important clauses...")
            predicted_fingerprint, detected_clauses_df = predict_clauses(
                clean_text=clean_text,
                tfidf=tfidf,
                clause_detector=clause_detector,
                mvp_config=mvp_config,
                filename=uploaded_file.name
            )

            st.write("Finding supporting evidence...")
            evidence_df = retrieve_evidence(
                chunks_df=chunks_df,
                detected_clauses_df=detected_clauses_df,
                mvp_config=mvp_config,
                embedding_model=embedding_model,
                filename=uploaded_file.name,
                top_k=2
            )

            st.write("Assessing contract health...")
            risk_score, risk_drivers = score_contract(
                predicted_fingerprint.iloc[0],
                mvp_config
            )

            risk_band = assign_risk_band(risk_score)

            risk_driver_table = build_risk_driver_table(
                risk_drivers,
                recommendation_map=recommendation_map
            )

            detected_present = detected_clauses_df[
                detected_clauses_df["detected"] == True
            ].copy()

            review_status = manual_review_status(
                risk_band,
                len(risk_driver_table)
            )

            st.write("Preparing executive report...")

            report_payload = build_report_payload(
                filename=uploaded_file.name,
                risk_score=risk_score,
                risk_band=risk_band,
                detected_clauses_df=detected_clauses_df,
                risk_driver_table=risk_driver_table,
                evidence_df=evidence_df
            )

            business_report = build_business_report(
                filename=uploaded_file.name,
                risk_score=risk_score,
                risk_band=risk_band,
                detected_present=detected_present,
                risk_driver_table=risk_driver_table,
                evidence_df=evidence_df,
                review_status=review_status
            )

            gemini_prompt = build_clean_gemini_prompt(business_report)

            st.session_state["analysis_done"] = True
            st.session_state["filename"] = uploaded_file.name
            st.session_state["raw_text"] = raw_text
            st.session_state["clean_text"] = clean_text
            st.session_state["chunks_df"] = chunks_df
            st.session_state["predicted_fingerprint"] = predicted_fingerprint
            st.session_state["detected_clauses_df"] = detected_clauses_df
            st.session_state["detected_present"] = detected_present
            st.session_state["evidence_df"] = evidence_df
            st.session_state["risk_score"] = risk_score
            st.session_state["risk_band"] = risk_band
            st.session_state["risk_driver_table"] = risk_driver_table
            st.session_state["review_status"] = review_status
            st.session_state["report_payload"] = report_payload
            st.session_state["business_report"] = business_report
            st.session_state["gemini_prompt"] = gemini_prompt
            st.session_state["gemini_report"] = None

            status.update(
                label="MeridianIQ analysis complete.",
                state="complete",
                expanded=False
            )

        st.success("Contract analysis completed successfully.")

    except Exception as e:
        st.error("Something went wrong while analyzing the contract.")
        st.exception(e)
        st.stop()


if not st.session_state.get("analysis_done"):
    st.info("Click **Analyze Contract** to run MeridianIQ.")
    st.stop()


# --------------------------------------------------
# Read Session State
# --------------------------------------------------

filename = st.session_state["filename"]
raw_text = st.session_state["raw_text"]
clean_text = st.session_state["clean_text"]
chunks_df = st.session_state["chunks_df"]
predicted_fingerprint = st.session_state["predicted_fingerprint"]
detected_clauses_df = st.session_state["detected_clauses_df"]
detected_present = st.session_state["detected_present"]
evidence_df = st.session_state["evidence_df"]
risk_score = st.session_state["risk_score"]
risk_band = st.session_state["risk_band"]
risk_driver_table = st.session_state["risk_driver_table"]
review_status = st.session_state["review_status"]
report_payload = st.session_state["report_payload"]
business_report = st.session_state["business_report"]
gemini_prompt = st.session_state["gemini_prompt"]
health_icon = health_emoji(risk_band)


# --------------------------------------------------
# Tabs
# --------------------------------------------------

st.divider()


active_section = st.radio(
    "Navigation",
    ["Analysis", "Clauses", "Report", "Technical Details"],
    horizontal=True,
    key="active_section",
    label_visibility="collapsed"
)

# --------------------------------------------------
# Tab 1: Analysis
# --------------------------------------------------


if active_section == "Analysis":
    st.markdown(f"# {health_icon} Contract Health: {risk_band}")

    st.markdown(
        f"""
        {executive_health_message(risk_band)}

        **MeridianIQ recommendation:** {"Manual review is recommended before signing." if review_status != "No" else "No major manual review trigger was found based on current rules."}
        """
    )

    st.progress(min(int(risk_score), 100) / 100)
    st.caption(f"Contract Health Score: {int(risk_score)}/100")

    st.header("📊 Contract Snapshot")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Contract Health", f"{health_icon} {risk_band}")
    col2.metric("Important Clauses Found", len(detected_present))
    col3.metric("Review Items", len(risk_driver_table))
    col4.metric("Manual Review", review_status)

    st.caption(
        f"Processed {len(raw_text):,} characters into {len(chunks_df):,} readable contract sections."
    )

    st.divider()

    st.header("⚠️ Top Things To Review")

    if risk_driver_table.empty:
        st.success("No major score-changing review items were found.")
    else:
        for _, row in risk_driver_table.iterrows():
            severity_icon = (
                "🔴" if row.get("severity") == "Critical"
                else "🟠" if row.get("severity") == "High"
                else "🟡"
            )

            with st.container(border=True):
                st.markdown(f"### {severity_icon} {business_clause_label(row['clause_name'])}")

                st.markdown("**Why this matters**")
                st.write(row.get("business_description", "This clause may affect business risk."))

                st.markdown("**Suggested action**")
                st.write(row.get("recommendation", "Review this clause with legal or business stakeholders."))

                st.caption(
                    f"Area: {row.get('risk_domain', 'General Risk')} | Severity: {row.get('severity', 'Review')}"
                )


# --------------------------------------------------
# Tab 2: Clauses
# --------------------------------------------------


elif active_section == "Clauses":
    st.header("✅ What MeridianIQ Found")

    if detected_present.empty:
        st.info("No important clauses were detected.")
    else:
        found_items = [
            business_clause_label(clause)
            for clause in detected_present["clause_name"].tolist()
        ]

        cols = st.columns(2)

        for idx, item in enumerate(found_items):
            with cols[idx % 2]:
                st.markdown(f"✓ {item}")

    st.divider()

    st.header("📖 Where MeridianIQ Found It")

    if evidence_df.empty:
        st.info("No supporting evidence was retrieved.")
    else:
        for clause_name in evidence_df["clause_name"].unique():
            clause_evidence = (
                evidence_df[evidence_df["clause_name"] == clause_name]
                .sort_values("rank")
                .head(1)
            )

            with st.expander(f"📌 {business_clause_label(clause_name)}"):
                for _, row in clause_evidence.iterrows():
                    st.markdown("MeridianIQ found this section relevant:")
                    evidence_text = str(row["evidence_text"]).strip()
                    st.markdown(f"> {evidence_text[:1200]}...")


# --------------------------------------------------
# Tab 3: Report
# --------------------------------------------------


elif active_section == "Report":
    st.header("🤖 Executive Summary")

    report_choice = st.radio(
        "Choose report format",
        ["Structured Report", "Gemini Executive Report"],
        horizontal=True,
        key="report_choice"
    )

    if report_choice == "Gemini Executive Report":
        st.write(
            "Generate a polished executive report using the same MeridianIQ analysis."
        )

        gemini_api_key = st.text_input(
            "Gemini API Key",
            type="password",
            key="gemini_api_key"
        )

        if st.button("Generate Gemini Executive Report", key="generate_gemini_button"):
            if not gemini_api_key:
                st.warning("Please enter your Gemini API key first.")
            else:
                try:
                    with st.spinner("Generating Gemini executive report..."):
                        gemini_report = generate_gemini_report(
                            prompt=gemini_prompt,
                            api_key=gemini_api_key,
                            model_name="gemini-2.5-flash"
                        )

                    st.session_state["gemini_report"] = gemini_report
                    st.success("Gemini executive report generated successfully.")

                except Exception as e:
                    st.error("Gemini report could not be generated. Showing structured report instead.")
                    st.exception(e)

        if st.session_state.get("gemini_report"):
            st.markdown(st.session_state["gemini_report"])
        else:
            st.info("Gemini report has not been generated yet.")
            with st.expander("Show structured report instead"):
                st.markdown(business_report)

    else:
        st.markdown(business_report)

    st.divider()

    st.header("✅ MeridianIQ Decision")

    if review_status == "No":
        st.success("No major manual review trigger was found based on MeridianIQ's current rules.")
    else:
        st.warning(
            f"Manual review is recommended because MeridianIQ found {len(risk_driver_table)} review item(s) in this contract."
        )

    st.divider()

    st.header("⬇️ Download")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            "Download Structured Report",
            data=business_report,
            file_name="meridianiq_structured_report.md",
            mime="text/markdown",
            key="download_structured_report"
        )

    with col2:
        if st.session_state.get("gemini_report"):
            st.download_button(
                "Download Gemini Report",
                data=st.session_state["gemini_report"],
                file_name="meridianiq_gemini_report.md",
                mime="text/markdown",
                key="download_gemini_report"
            )
        else:
            st.button("Download Gemini Report", disabled=True)

    with col3:
        st.download_button(
            "Download Report Payload",
            data=json.dumps(report_payload, indent=2),
            file_name="meridianiq_report_payload.json",
            mime="application/json",
            key="download_payload"
        )


# --------------------------------------------------
# Tab 4: Technical Details
# --------------------------------------------------


elif active_section == "Technical Details":
    st.header("🔧 Technical Details")

    st.caption(
        "This section contains model outputs, retrieval metadata, similarity scores, ranks, processed chunks, and payload data for debugging."
    )

    st.subheader("Detected Clause Predictions")
    st.dataframe(detected_clauses_df, use_container_width=True)

    st.subheader("Predicted Fingerprint")
    st.dataframe(predicted_fingerprint, use_container_width=True)

    st.subheader("Risk Driver Table")
    st.dataframe(risk_driver_table, use_container_width=True)

    st.subheader("Evidence Retrieval Details")
    st.dataframe(evidence_df, use_container_width=True)

    st.subheader("Contract Chunks")
    st.dataframe(
        chunks_df[["chunk_id", "char_count", "word_count"]],
        use_container_width=True
    )

    st.subheader("Cleaned Contract Preview")
    st.text_area(
        "Cleaned Contract Text",
        clean_text[:5000],
        height=300,
        key="cleaned_contract_preview"
    )

    st.subheader("Gemini Prompt")
    st.text_area(
        "Gemini Prompt",
        gemini_prompt,
        height=300,
        key="gemini_prompt_preview"
    )

    st.subheader("Report Payload")
    st.json(report_payload)
st.divider()

st.markdown(
    """
    <div style="
        text-align: center;
        color: #8b8b8b;
        font-size: 14px;
        line-height: 1.8;
        padding-top: 10px;
        padding-bottom: 10px;
    ">
        <strong>🧭 MeridianIQ v1.0</strong><br>
        Hybrid Clause Extraction and LLM Orchestration for Contract Risk Intelligence <br><br>
        Built by <strong>Siddhi Mane</strong><br><br>
        <em>MeridianIQ provides AI-assisted contract review for informational purposes only and does not constitute legal advice.</em>
    </div>
    """,
    unsafe_allow_html=True,
)