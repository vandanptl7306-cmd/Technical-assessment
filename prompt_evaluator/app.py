"""
Streamlit Web Application: Is the New Prompt Better?
Professional AI/ML Assistant Evaluation Dashboard with Safety-First Decisions & Human Review.
"""

import os
import sys
import io
import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Prompt Evaluator | AI Assistant Benchmark",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom SaaS Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-title {
        color: #64748b;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        color: #0f172a;
        font-size: 1.85rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .metric-sub {
        color: #475569;
        font-size: 0.85rem;
        margin-top: 4px;
    }
    .badge-win-new {
        background-color: #dcfce7;
        color: #15803d;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.82rem;
    }
    .badge-win-curr {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.82rem;
    }
    .badge-tie {
        background-color: #f1f5f9;
        color: #475569;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.82rem;
    }
    .badge-critical {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #f87171;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.82rem;
    }
    .badge-high {
        background-color: #ffedd5;
        color: #c2410c;
        border: 1px solid #fb923c;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.82rem;
    }
    .badge-medium {
        background-color: #fef9c3;
        color: #854d0e;
        border: 1px solid #facc15;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.82rem;
    }
    .banner-danger {
        background-color: #fef2f2;
        border-left: 5px solid #ef4444;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }
    .banner-warning {
        background-color: #fffbeb;
        border-left: 5px solid #f59e0b;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }
    .banner-success {
        background-color: #f0fdf4;
        border-left: 5px solid #22c55e;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 20px;
    }
    .banner-transparency {
        background-color: #f8fafc;
        border-left: 5px solid #0284c7;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .review-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Imports from modular engine
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

try:
    from prompt_evaluator.data_loader import (
        load_dataset,
        validate_dataset,
        get_default_dataset_path,
        DataValidationError,
    )
    from prompt_evaluator.evaluator import AnswerEvaluator, DEFAULT_TIE_THRESHOLD
    from prompt_evaluator.report_generator import (
        compute_metrics,
        generate_verdicts_csv,
        generate_scores_csv,
        generate_evaluation_report,
    )
except ImportError:
    from data_loader import (
        load_dataset,
        validate_dataset,
        get_default_dataset_path,
        DataValidationError,
    )
    from evaluator import AnswerEvaluator, DEFAULT_TIE_THRESHOLD
    from report_generator import (
        compute_metrics,
        generate_verdicts_csv,
        generate_scores_csv,
        generate_evaluation_report,
    )


@st.cache_data(show_spinner="Evaluating prompt responses across 300 questions...")
def run_evaluation(df_data: pd.DataFrame, tie_threshold: float):
    evaluator = AnswerEvaluator()
    results_df = evaluator.evaluate_dataframe(df_data, tie_threshold=tie_threshold)
    metrics = compute_metrics(results_df, tie_threshold=tie_threshold)
    return results_df, metrics


# Initialize Session State for Human Reviews
if "human_reviews" not in st.session_state:
    st.session_state["human_reviews"] = {}

# -------------------------------------------------------------
# Sidebar Configuration & Navigation
# -------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/scale.png", width=54)
st.sidebar.title("Prompt Evaluator")
st.sidebar.caption("AI Assistant Version Benchmark")

nav_choice = st.sidebar.radio(
    "Navigation",
    [
        "📊 Dashboard",
        "🛡️ Human Review (Safety)",
        "📈 Analytics & Charts",
        "🔍 Individual Evaluation",
        "📋 Full Question Explorer",
        "📁 Dataset Management",
        "📑 Report & Exports",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Evaluation Parameters")

tie_threshold = st.sidebar.slider(
    "Tie Threshold (± pts)",
    min_value=0.0,
    max_value=10.0,
    value=DEFAULT_TIE_THRESHOLD,
    step=0.5,
    help=(
        "Documented Tie Threshold: If absolute score difference |New - Current| is within "
        "this threshold (±2.5 pts), the verdict is recorded as a 'tie' to avoid noise-driven bias."
    ),
)

# Manage Dataset Source
if "uploaded_df" not in st.session_state:
    st.session_state["uploaded_df"] = None
    st.session_state["dataset_name"] = "questions.csv (Default)"

uploaded_file = st.sidebar.file_uploader(
    "Upload Alternative CSV",
    type=["csv"],
    help="Must contain columns: id, topic, question, answer_current, answer_new",
)

if uploaded_file is not None:
    try:
        custom_df, meta = load_dataset(uploaded_file)
        st.session_state["uploaded_df"] = custom_df
        st.session_state["dataset_name"] = uploaded_file.name
        st.sidebar.success(f"Loaded: {uploaded_file.name} ({len(custom_df)} rows)")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {str(e)}")

# Load Active Dataset
if st.session_state["uploaded_df"] is not None:
    active_df = st.session_state["uploaded_df"]
else:
    default_path = get_default_dataset_path()
    try:
        active_df, _ = load_dataset(default_path)
    except Exception as e:
        st.error(f"Failed to load default dataset: {str(e)}")
        st.stop()

# Run Evaluation
results_df, metrics = run_evaluation(active_df, tie_threshold)

# Pre-populate human reviews for flagged questions if not present
for _, r in results_df.iterrows():
    qid = str(r["id"])
    has_risk = bool(r["new_policy_risks"] and len(r["new_policy_risks"]) > 0)
    if has_risk and qid not in st.session_state["human_reviews"]:
        st.session_state["human_reviews"][qid] = {
            "status": "Pending",
            "notes": "",
            "reviewed_at": None,
        }

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    **Active Dataset:** `{st.session_state['dataset_name']}`  
    **Evaluated Rows:** `{len(results_df)}`  
    **Policy Risks Flagged:** `{metrics['risk_count']}`  
    **Critical Security Risks:** `{metrics.get('critical_security_count', 0)}`  
    **Safety Principle:** `Safety-First Governance`
    """
)


# -------------------------------------------------------------
# Section A: Dashboard
# -------------------------------------------------------------
if nav_choice == "📊 Dashboard":
    st.title("⚖️ AI Assistant Prompt Evaluation Dashboard")
    st.markdown(
        "A rigorous, explainable evaluation comparing the **Live Assistant Prompt** against the **Proposed New Prompt** "
        "across 300 customer support questions under **Safety-First Governance**."
    )

    # Final Recommendation Banner with Safety-First Enforcement
    rec = metrics["recommendation"]
    if rec == "DO NOT SHIP":
        st.markdown(
            f"""
            <div class="banner-danger">
                <h3 style="color: #991b1b; margin-top:0;">🛑 FINAL RECOMMENDATION: DO NOT SHIP (IN CURRENT FORM)</h3>
                <p style="color: #7f1d1d; font-size: 1.05rem; margin-bottom: 8px;">
                    <b>Safety-First Finding:</b> {metrics['recommendation_summary']}
                </p>
                <p style="color: #991b1b; font-size: 0.95rem; margin-bottom: 0;">
                    <b>Safety Directive:</b> High average scores or actionability improvements <u>cannot override</u> 
                    critical security vulnerabilities (such as 2FA bypass in Q-081) or policy hallucinations in <code>refunds</code>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif rec == "SHIP WITH CONDITIONS":
        st.markdown(
            f"""
            <div class="banner-warning">
                <h3 style="color: #92400e; margin-top:0;">⚠️ FINAL RECOMMENDATION: SHIP WITH CONDITIONS</h3>
                <p style="color: #78350f; font-size: 1.05rem; margin-bottom: 8px;">
                    <b>Safety-First Finding:</b> {metrics['recommendation_summary']}
                </p>
                <p style="color: #92400e; font-size: 0.95rem; margin-bottom: 0;">
                    <b>Required Conditions:</b> Review flagged items in the Human Review module and add negative constraints to prompt.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="banner-success">
                <h3 style="color: #166534; margin-top:0;">✅ FINAL RECOMMENDATION: SHIP</h3>
                <p style="color: #14532d; font-size: 1.05rem; margin-bottom: 0;">
                    {metrics['recommendation_summary']}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Evaluation Transparency & Scoring Methodology Banner
    st.markdown(
        """
        <div class="banner-transparency">
            <h4 style="color: #0369a1; margin-top: 0; margin-bottom: 8px;">🔬 Evaluation Transparency & Methodological Disclosure</h4>
            <div style="font-size: 0.92rem; color: #334155; line-height: 1.5;">
                <p style="margin-bottom: 6px;">
                    • <b>Heuristic Scores are Estimates:</b> Automated scores are rule-based and NLP approximations (TF-IDF cosine similarity, keyword coverage, and syntactic density) designed for objective regression testing, not absolute ground truth.
                </p>
                <p style="margin-bottom: 6px;">
                    • <b>Policy Flags Require Human Verification:</b> Automated policy and security risk flags are screening alerts indicating potential hallucinations or security bypasses. Domain experts must verify each flag in the <b>Human Review</b> section before production action.
                </p>
                <p style="margin-bottom: 0;">
                    • <b>Scoring Methodology & Weights:</b> Total Score (0–100) = 2.5×Relevance (25%) + 2.5×Completeness (25%) + 2.0×Actionability (20%) + 1.5×Clarity (15%) + 1.5×Conciseness (15%). Documented Tie Threshold: <b>±2.5 pts</b>.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPI Summary Cards
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Questions</div>
                <div class="metric-value">{metrics['total_questions']}</div>
                <div class="metric-sub">100% Evaluated</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Current Avg</div>
                <div class="metric-value">{metrics['curr_avg_score']:.1f}</div>
                <div class="metric-sub">Out of 100</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        diff_color = "#15803d" if metrics['score_diff_avg'] >= 0 else "#b91c1c"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">New Avg</div>
                <div class="metric-value">{metrics['new_avg_score']:.1f}</div>
                <div class="metric-sub" style="color: {diff_color}; font-weight:600;">{metrics['score_diff_avg']:+.1f} pts diff</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">New Win Rate</div>
                <div class="metric-value" style="color: #15803d;">{metrics['new_win_rate']:.1f}%</div>
                <div class="metric-sub">{metrics['new_wins']} questions</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Current Win Rate</div>
                <div class="metric-value" style="color: #b91c1c;">{metrics['curr_win_rate']:.1f}%</div>
                <div class="metric-sub">{metrics['curr_wins']} questions</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col6:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Tie Rate</div>
                <div class="metric-value" style="color: #475569;">{metrics['tie_rate']:.1f}%</div>
                <div class="metric-sub">{metrics['ties']} questions</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # High-level Visuals Grid
    chart_col1, chart_col2 = st.columns([3, 2])

    with chart_col1:
        st.subheader("📊 Average Score by Topic (Current vs New)")
        topic_data = []
        for t, m in metrics["topic_metrics"].items():
            topic_data.append({"Topic": t.title(), "Version": "Current", "Score": m["curr_avg"]})
            topic_data.append({"Topic": t.title(), "Version": "New", "Score": m["new_avg"]})
        tdf = pd.DataFrame(topic_data)

        fig_topic = px.bar(
            tdf,
            x="Topic",
            y="Score",
            color="Version",
            barmode="group",
            color_discrete_map={"Current": "#3b82f6", "New": "#10b981"},
            text_auto=".1f",
            height=340,
        )
        fig_topic.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis_range=[0, 100],
        )
        st.plotly_chart(fig_topic, use_container_width=True)

    with chart_col2:
        st.subheader("🥧 Pairwise Outcome Distribution")
        pie_df = pd.DataFrame(
            {
                "Outcome": ["Current Wins", "New Wins", "Ties"],
                "Count": [metrics["curr_wins"], metrics["new_wins"], metrics["ties"]],
            }
        )
        fig_pie = px.pie(
            pie_df,
            names="Outcome",
            values="Count",
            color="Outcome",
            color_discrete_map={
                "Current Wins": "#ef4444",
                "New Wins": "#22c55e",
                "Ties": "#94a3b8",
            },
            hole=0.45,
            height=340,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Criteria Comparison Table & Summary
    st.subheader("🎯 Evaluation Criteria Score Breakdown")
    crit_cols = st.columns(5)
    criteria_names = ["relevance", "completeness", "actionability", "clarity", "conciseness"]
    criteria_labels = {
        "relevance": ("Relevance", "25%", "🎯"),
        "completeness": ("Completeness", "25%", "📚"),
        "actionability": ("Actionability", "20%", "⚡"),
        "clarity": ("Clarity", "15%", "🔍"),
        "conciseness": ("Conciseness", "15%", "✂️"),
    }

    for idx, c in enumerate(criteria_names):
        c_meta = metrics["criteria_metrics"][c]
        name, weight, icon = criteria_labels[c]
        d_val = c_meta["delta"]
        d_color = "#15803d" if d_val > 0 else ("#b91c1c" if d_val < 0 else "#475569")
        with crit_cols[idx]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">{icon} {name} ({weight})</div>
                    <div style="font-size: 1.1rem; font-weight:600; margin: 4px 0;">
                        Curr: <b>{c_meta['current']:.1f}</b> | New: <b>{c_meta['new']:.1f}</b>
                    </div>
                    <div style="color: {d_color}; font-weight:700; font-size: 0.95rem;">
                        Delta: {d_val:+.2f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.info(
        "💡 **Key Insight**: The New Prompt achieved higher **Actionability (+1.00)** and **Completeness (+0.66)** "
        "by giving direct UI navigation instructions. However, it suffered a **massive penalty in Conciseness (-6.24)** "
        "due to generic greetings/signoffs added to 87.7% of answers, and introduced **30 critical policy hallucinations**."
    )


# -------------------------------------------------------------
# Section B: Human Review (Safety & Policy Audit)
# -------------------------------------------------------------
elif nav_choice == "🛡️ Human Review (Safety)":
    st.title("🛡️ Human Review: Safety & Policy Risk Audit")
    st.markdown(
        "Human-in-the-loop audit interface for flagged policy violations and security risks. "
        "Reviewers verify automated detections and record audit statuses. "
        "*(Note: The original automated verdicts in `verdicts.csv` remain strictly unchanged for governance auditability).*"
    )

    # Filter flagged questions
    flagged_df = results_df[results_df["new_policy_risks"].apply(lambda r: len(r) > 0)].copy()

    # Calculate review counts
    confirmed_count = sum(1 for qid in flagged_df["id"] if st.session_state["human_reviews"].get(str(qid), {}).get("status") == "Confirmed")
    fp_count = sum(1 for qid in flagged_df["id"] if st.session_state["human_reviews"].get(str(qid), {}).get("status") == "False Positive")
    pending_count = len(flagged_df) - confirmed_count - fp_count
    critical_count = sum(1 for _, row in flagged_df.iterrows() if any("[Critical]" in x for x in row["new_policy_risks"]))

    # Summary KPI Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Flagged Questions</div>
                <div class="metric-value">{len(flagged_df)}</div>
                <div class="metric-sub">{critical_count} Critical Severity</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Pending Review</div>
                <div class="metric-value" style="color: #d97706;">{pending_count}</div>
                <div class="metric-sub">Awaiting verification</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Confirmed Risks</div>
                <div class="metric-value" style="color: #dc2626;">{confirmed_count}</div>
                <div class="metric-sub">Action required</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">False Positives</div>
                <div class="metric-value" style="color: #16a34a;">{fp_count}</div>
                <div class="metric-sub">Cleared findings</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Review Filters
    rf1, rf2, rf3 = st.columns(3)
    with rf1:
        status_filter = st.selectbox("Filter by Review Status", ["All", "Pending", "Confirmed", "False Positive"])
    with rf2:
        severity_filter = st.selectbox("Filter by Severity", ["All", "Critical", "High", "Medium"])
    with rf3:
        topic_rf = st.selectbox("Filter by Topic", ["All"] + sorted(list(flagged_df["topic"].unique())))

    # Apply filters
    display_flagged = flagged_df.copy()
    if topic_rf != "All":
        display_flagged = display_flagged[display_flagged["topic"] == topic_rf]
    if severity_filter != "All":
        display_flagged = display_flagged[
            display_flagged["new_policy_risks"].apply(lambda risks: any(f"[{severity_filter}]" in r for r in risks))
        ]
    if status_filter != "All":
        display_flagged = display_flagged[
            display_flagged["id"].apply(lambda qid: st.session_state["human_reviews"].get(str(qid), {}).get("status") == status_filter)
        ]

    st.write(f"Showing **{len(display_flagged)}** flagged question(s) matching filter:")

    # Render review cards
    for idx, row in display_flagged.iterrows():
        qid = str(row["id"])
        review_data = st.session_state["human_reviews"].get(qid, {"status": "Pending", "notes": ""})

        # Determine severity badge
        has_crit = any("[Critical]" in x for x in row["new_policy_risks"])
        has_high = any("[High]" in x for x in row["new_policy_risks"])
        sev_label = "CRITICAL" if has_crit else ("HIGH" if has_high else "MEDIUM")
        sev_class = "badge-critical" if has_crit else ("badge-high" if has_high else "badge-medium")

        automated_verdict = row["better"].upper()
        verdict_badge_class = "badge-win-new" if row["better"] == "new" else ("badge-win-curr" if row["better"] == "current" else "badge-tie")

        with st.container():
            st.markdown(
                f"""
                <div class="review-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-weight: 700; font-size: 1.15rem; color: #0f172a;">
                            {row['id']} — Topic: <code>{row['topic']}</code>
                        </span>
                        <div>
                            <span class="{sev_class}">SEVERITY: {sev_label}</span>
                            <span class="{verdict_badge_class}" style="margin-left: 8px;">Automated Verdict: {automated_verdict} (Unchanged)</span>
                        </div>
                    </div>
                    <div style="font-size: 1.0rem; color: #1e293b; margin-bottom: 8px;">
                        <b>Customer Question:</b> {row['question']}
                    </div>
                    <div style="background-color: #fef2f2; border: 1px solid #fecaca; border-radius: 6px; padding: 10px 14px; margin-bottom: 12px;">
                        <b style="color: #991b1b;">⚠️ Detected Risk(s):</b>
                        <ul style="margin: 4px 0 0 0; padding-left: 20px; color: #7f1d1d;">
                            {"".join(f"<li>{r}</li>" for r in row['new_policy_risks'])}
                        </ul>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            c_ans1, c_ans2 = st.columns(2)
            with c_ans1:
                st.caption(f"Current Answer (Score: {row['curr_total_score']:.1f})")
                st.info(row["answer_current"])
            with c_ans2:
                st.caption(f"New Answer (Score: {row['new_total_score']:.1f})")
                st.warning(row["answer_new"])

            # Reviewer controls
            r_col1, r_col2 = st.columns([1, 2])
            with r_col1:
                new_status = st.radio(
                    f"Review Status for {qid}",
                    ["Pending", "Confirmed", "False Positive"],
                    index=["Pending", "Confirmed", "False Positive"].index(review_data["status"]),
                    key=f"status_select_{qid}",
                    horizontal=True,
                )
            with r_col2:
                new_notes = st.text_input(
                    f"Reviewer Notes for {qid}",
                    value=review_data.get("notes", ""),
                    placeholder="e.g., Confirmed 2FA vulnerability; model must require backup code.",
                    key=f"notes_input_{qid}",
                )

            # Update session state
            if new_status != review_data["status"] or new_notes != review_data.get("notes", ""):
                st.session_state["human_reviews"][qid] = {
                    "status": new_status,
                    "notes": new_notes,
                    "reviewed_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }

            st.markdown("---")

    # Audit Log Export
    st.subheader("📥 Export Human Review Audit Log")
    log_rows = []
    for idx, row in flagged_df.iterrows():
        qid = str(row["id"])
        r_meta = st.session_state["human_reviews"].get(qid, {"status": "Pending", "notes": "", "reviewed_at": None})
        has_crit = any("[Critical]" in x for x in row["new_policy_risks"])
        sev = "Critical" if has_crit else "High"
        log_rows.append(
            {
                "id": qid,
                "topic": row["topic"],
                "severity": sev,
                "detected_risks": "; ".join(row["new_policy_risks"]),
                "automated_verdict": row["better"],
                "human_review_status": r_meta["status"],
                "reviewer_notes": r_meta["notes"],
                "reviewed_at": r_meta.get("reviewed_at", "N/A"),
            }
        )
    log_df = pd.DataFrame(log_rows)
    st.download_button(
        label="⬇️ Download Human Review Audit Log (CSV)",
        data=log_df.to_csv(index=False),
        file_name="human_review_audit_log.csv",
        mime="text/csv",
    )


# -------------------------------------------------------------
# Section C: Analytics & Charts
# -------------------------------------------------------------
elif nav_choice == "📈 Analytics & Charts":
    st.title("📈 Performance Analytics & Deep Dive")

    tab1, tab2, tab3 = st.tabs(["Dimension Radar & Scores", "Topic Regressions", "Policy Risk Heatmap"])

    with tab1:
        st.subheader("Radar Chart: Multi-Criteria Profile")
        categories = ["Relevance", "Completeness", "Actionability", "Clarity", "Conciseness"]
        curr_vals = [metrics["criteria_metrics"][c.lower()]["current"] for c in categories]
        new_vals = [metrics["criteria_metrics"][c.lower()]["new"] for c in categories]

        fig_radar = go.Figure()
        fig_radar.add_trace(
            go.Scatterpolar(r=curr_vals, theta=categories, fill="toself", name="Current Version", line_color="#3b82f6")
        )
        fig_radar.add_trace(
            go.Scatterpolar(r=new_vals, theta=categories, fill="toself", name="New Version", line_color="#10b981")
        )
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
            showlegend=True,
            height=450,
            margin=dict(l=60, r=60, t=20, b=20),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        st.subheader("Score Differential Distribution (New - Current)")
        fig_hist = px.histogram(
            results_df,
            x="score_diff",
            color="better",
            nbins=35,
            color_discrete_map={"current": "#ef4444", "new": "#22c55e", "tie": "#94a3b8"},
            labels={"score_diff": "Score Difference (pts)", "better": "Verdict"},
            height=320,
        )
        fig_hist.add_vline(x=tie_threshold, line_dash="dash", line_color="green", annotation_text="+Tie Bound")
        fig_hist.add_vline(x=-tie_threshold, line_dash="dash", line_color="red", annotation_text="-Tie Bound")
        fig_hist.update_layout(margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_hist, use_container_width=True)

    with tab2:
        st.subheader("Topic Performance & Win Rates")
        topic_summary_data = []
        for t, m in metrics["topic_metrics"].items():
            topic_summary_data.append(
                {
                    "Topic": t.title(),
                    "Total Questions": m["total"],
                    "Current Avg": m["curr_avg"],
                    "New Avg": m["new_avg"],
                    "Delta": m["diff"],
                    "New Win %": m["new_win_rate"],
                    "Current Win %": m["curr_win_rate"],
                    "Tie %": m["tie_rate"],
                    "Regression Status": "⚠️ Regressed" if m["regressed"] else "✅ Healthy",
                }
            )
        st.dataframe(pd.DataFrame(topic_summary_data), use_container_width=True, hide_index=True)

        st.subheader("Topic Win/Loss Share")
        t_chart_data = []
        for t, m in metrics["topic_metrics"].items():
            t_chart_data.append({"Topic": t.title(), "Outcome": "New Wins", "Count": m["new_wins"]})
            t_chart_data.append({"Topic": t.title(), "Outcome": "Current Wins", "Count": m["curr_wins"]})
            t_chart_data.append({"Topic": t.title(), "Outcome": "Ties", "Count": m["ties"]})
        t_chart_df = pd.DataFrame(t_chart_data)

        fig_tbar = px.bar(
            t_chart_df,
            x="Topic",
            y="Count",
            color="Outcome",
            barmode="stack",
            color_discrete_map={"Current Wins": "#ef4444", "New Wins": "#22c55e", "Ties": "#94a3b8"},
            height=350,
        )
        st.plotly_chart(fig_tbar, use_container_width=True)

    with tab3:
        st.subheader("⚠️ Detected Policy Risks and Hallucinations in New Prompt")
        risky_df = results_df[results_df["new_policy_risks"].apply(lambda r: len(r) > 0)].copy()
        
        st.warning(
            f"Found **{len(risky_df)} questions** where the New Prompt hallucinated unauthorized policies, "
            f"made false promises, or introduced security bypass vulnerabilities."
        )

        risk_topic_counts = risky_df["topic"].value_counts().reset_index()
        risk_topic_counts.columns = ["Topic", "Risk Count"]
        
        fig_risk = px.bar(
            risk_topic_counts,
            x="Topic",
            y="Risk Count",
            color="Risk Count",
            color_continuous_scale="Reds",
            text_auto=True,
            height=300,
        )
        st.plotly_chart(fig_risk, use_container_width=True)

        st.markdown("#### Sample Flagged Hallucinations")
        display_risks = risky_df[["id", "topic", "question", "new_policy_risks", "answer_new"]].head(8)
        st.dataframe(display_risks, use_container_width=True, hide_index=True)


# -------------------------------------------------------------
# Section D: Individual Question Evaluation
# -------------------------------------------------------------
elif nav_choice == "🔍 Individual Evaluation":
    st.title("🔍 Individual Question Inspector")
    st.markdown("Inspect granular criterion-level scores and transparent decision rationales for any question.")

    # Selection controls
    col_sel1, col_sel2 = st.columns([1, 3])
    with col_sel1:
        topic_filter = st.selectbox("Filter by Topic", ["All"] + sorted(list(active_df["topic"].unique())))
    with col_sel2:
        winner_filter = st.selectbox("Filter by Winner", ["All", "new", "current", "tie"])

    filtered_df = results_df.copy()
    if topic_filter != "All":
        filtered_df = filtered_df[filtered_df["topic"] == topic_filter]
    if winner_filter != "All":
        filtered_df = filtered_df[filtered_df["better"] == winner_filter]

    q_options = filtered_df["id"].tolist()
    if not q_options:
        st.warning("No questions match the selected filters.")
        st.stop()

    selected_id = st.selectbox(
        "Select Question ID",
        q_options,
        format_func=lambda x: f"{x} - {results_df[results_df['id'] == x]['question'].values[0][:80]}...",
    )

    q_data = results_df[results_df["id"] == selected_id].iloc[0]

    # Question Details Header
    winner_badge = {
        "new": '<span class="badge-win-new">WINNER: NEW VERSION</span>',
        "current": '<span class="badge-win-curr">WINNER: CURRENT VERSION</span>',
        "tie": '<span class="badge-tie">VERDICT: TIE</span>',
    }[q_data["better"]]

    st.markdown(
        f"""
        <div style="background-color: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 16px 20px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-weight: 700; font-size: 1.15rem; color: #1e293b;">{q_data['id']} (Topic: {q_data['topic'].title()})</span>
                {winner_badge}
            </div>
            <div style="font-size: 1.05rem; color: #0f172a; font-weight: 500;">
                ❓ <b>Customer Question:</b> {q_data['question']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Policy Risk Warning if applicable
    if q_data["new_policy_risks"]:
        has_crit = any("[Critical]" in x for x in q_data["new_policy_risks"])
        alert_title = "CRITICAL SECURITY/POLICY RISK" if has_crit else "POLICY VIOLATION ALERT"
        st.error(f"🚨 **{alert_title} in New Version:** {'; '.join(q_data['new_policy_risks'])}")

    # Side-by-side Answer Comparison
    ans_col1, ans_col2 = st.columns(2)
    with ans_col1:
        st.markdown(f"### Current Version (Score: **{q_data['curr_total_score']:.1f}**/100)")
        st.info(q_data["answer_current"])
        st.caption(f"Word Count: {len(q_data['answer_current'].split())} words")

    with ans_col2:
        st.markdown(f"### New Version (Score: **{q_data['new_total_score']:.1f}**/100)")
        st.success(q_data["answer_new"])
        st.caption(f"Word Count: {len(q_data['answer_new'].split())} words")

    # Transparent Rationale Card
    st.markdown(
        f"""
        <div style="background-color: #f1f5f9; border-left: 4px solid #3b82f6; border-radius: 6px; padding: 12px 16px; margin: 16px 0;">
            <b>💡 Decision Rationale:</b> {q_data['explanation']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Detailed Rubric Table
    st.subheader("Detailed Scoring Rubric Breakdown")
    crit_rows = [
        {
            "Criterion": "Relevance (25%)",
            "Current Score": f"{q_data['curr_relevance']} / 10",
            "Current Notes": q_data["curr_relevance_exp"],
            "New Score": f"{q_data['new_relevance']} / 10",
            "New Notes": q_data["new_relevance_exp"],
        },
        {
            "Criterion": "Completeness (25%)",
            "Current Score": f"{q_data['curr_completeness']} / 10",
            "Current Notes": q_data["curr_completeness_exp"],
            "New Score": f"{q_data['new_completeness']} / 10",
            "New Notes": q_data["new_completeness_exp"],
        },
        {
            "Criterion": "Actionability (20%)",
            "Current Score": f"{q_data['curr_actionability']} / 10",
            "Current Notes": q_data["curr_actionability_exp"],
            "New Score": f"{q_data['new_actionability']} / 10",
            "New Notes": q_data["new_actionability_exp"],
        },
        {
            "Criterion": "Clarity (15%)",
            "Current Score": f"{q_data['curr_clarity']} / 10",
            "Current Notes": q_data["curr_clarity_exp"],
            "New Score": f"{q_data['new_clarity']} / 10",
            "New Notes": q_data["new_clarity_exp"],
        },
        {
            "Criterion": "Conciseness (15%)",
            "Current Score": f"{q_data['curr_conciseness']} / 10",
            "Current Notes": q_data["curr_conciseness_exp"],
            "New Score": f"{q_data['new_conciseness']} / 10",
            "New Notes": q_data["new_conciseness_exp"],
        },
        {
            "Criterion": "Weighted Total (100%)",
            "Current Score": f"{q_data['curr_total_score']:.1f} / 100",
            "Current Notes": "Calculated via weighted formula",
            "New Score": f"{q_data['new_total_score']:.1f} / 100",
            "New Notes": f"Score Delta: {q_data['score_diff']:+.1f} pts",
        },
    ]
    st.dataframe(pd.DataFrame(crit_rows), use_container_width=True, hide_index=True)


# -------------------------------------------------------------
# Section E: Full Question Explorer
# -------------------------------------------------------------
elif nav_choice == "📋 Full Question Explorer":
    st.title("📋 Full Question Explorer & Table")
    st.markdown("Browse, search, sort, and filter all 300 customer questions and evaluated verdicts.")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        f_topic = st.selectbox("Topic Filter", ["All"] + sorted(list(results_df["topic"].unique())))
    with col_f2:
        f_winner = st.selectbox("Verdict Filter", ["All", "new", "current", "tie"])
    with col_f3:
        f_search = st.text_input("Search in Question or Answers", "")

    view_df = results_df.copy()
    if f_topic != "All":
        view_df = view_df[view_df["topic"] == f_topic]
    if f_winner != "All":
        view_df = view_df[view_df["better"] == f_winner]
    if f_search:
        s_term = f_search.lower()
        view_df = view_df[
            view_df["question"].str.lower().str.contains(s_term)
            | view_df["answer_current"].str.lower().str.contains(s_term)
            | view_df["answer_new"].str.lower().str.contains(s_term)
        ]

    st.write(f"Showing **{len(view_df)}** of {len(results_df)} questions:")

    display_cols = [
        "id",
        "topic",
        "better",
        "score_diff",
        "curr_total_score",
        "new_total_score",
        "question",
        "answer_current",
        "answer_new",
        "explanation",
    ]
    st.dataframe(view_df[display_cols], use_container_width=True, hide_index=True)


# -------------------------------------------------------------
# Section F: Dataset Management
# -------------------------------------------------------------
elif nav_choice == "📁 Dataset Management":
    st.title("📁 Dataset Management & Schema Validator")
    st.markdown(
        "Inspect the current dataset, upload new evaluation test suites, and validate data integrity."
    )

    st.subheader("Current Dataset Summary")
    val_report = validate_dataset(active_df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", val_report["row_count"])
    c2.metric("Unique Topics", len(val_report["topic_counts"]))
    c3.metric("Missing Columns", len(val_report["missing_columns"]))
    c4.metric("Duplicate IDs", val_report["duplicate_ids"])

    if val_report["warnings"]:
        for w in val_report["warnings"]:
            st.warning(f"⚠️ {w}")
    else:
        st.success("✅ Dataset schema and data integrity checks passed 100%.")

    st.subheader("Topic Distribution")
    st.bar_chart(pd.Series(val_report["topic_counts"]))

    st.subheader("Raw Data Preview")
    st.dataframe(active_df.head(15), use_container_width=True)


# -------------------------------------------------------------
# Section G: Report & Exports
# -------------------------------------------------------------
elif nav_choice == "📑 Report & Exports":
    st.title("📑 Export Artifacts & Full Report")
    st.markdown("Download required evaluation artifacts or inspect the generated Markdown report.")

    exp_col1, exp_col2, exp_col3 = st.columns(3)

    # In-memory CSV buffers
    verdicts_csv_str = results_df[["id", "better"]].to_csv(index=False)
    scores_csv_str = results_df.to_csv(index=False)

    report_path = "prompt_evaluator/outputs/evaluation_report.md"
    if not os.path.exists(report_path):
        report_path = "outputs/evaluation_report.md"

    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            report_text = f.read()
    else:
        report_text = "# Report not yet generated."

    with exp_col1:
        st.download_button(
            label="⬇️ Download verdicts.csv (id,better)",
            data=verdicts_csv_str,
            file_name="verdicts.csv",
            mime="text/csv",
            help="Contains exactly two columns: id and better for all 300 questions.",
            use_container_width=True,
        )

    with exp_col2:
        st.download_button(
            label="⬇️ Download scores.csv (Full Details)",
            data=scores_csv_str,
            file_name="scores.csv",
            mime="text/csv",
            help="Detailed scoring breakdown across all 5 criteria for both versions.",
            use_container_width=True,
        )

    with exp_col3:
        st.download_button(
            label="⬇️ Download evaluation_report.md",
            data=report_text,
            file_name="evaluation_report.md",
            mime="text/markdown",
            help="Full executive report with methodology, results, and recommendations.",
            use_container_width=True,
        )

    st.markdown("---")
    st.subheader("📖 Live Evaluation Report Preview")
    st.markdown(report_text)
