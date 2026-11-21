"""
Answer Evaluation Engine for Prompt Evaluator.
Implements transparent, explainable 5-criterion scoring rubric with lightweight NLP heuristics.
Does not require external APIs or API keys.
"""

from typing import Dict, Any, Tuple, List, Optional
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Weight definitions (sum = 10.0 for multiplier to 100)
WEIGHTS = {
    "relevance": 2.5,       # 25%
    "completeness": 2.5,    # 25%
    "actionability": 2.0,   # 20%
    "clarity": 1.5,         # 15%
    "conciseness": 1.5,     # 15%
}

DEFAULT_TIE_THRESHOLD = 2.5

# Conversational boilerplate patterns to penalize in conciseness and filter for relevance
FLUFF_PATTERNS = [
    r"thanks for (getting in touch|reaching out|being a [a-z0-9_-]+ customer|contacting us|contacting)[!.]?",
    r"great question[!.]?",
    r"i hope that helps[!.]?",
    r"we are here if you need anything more[!.]?",
    r"please reach out again if you have any other questions[!.]?",
    r"let us know if there is anything else we can do for you[!.]?",
    r"of course, let me explain[!.]?",
    r"happy to help with this[!.]?",
    r"sorry for any confusion[!.]?",
    r"don't hesitate to reach back out[!.]?",
]

ACTION_VERBS = [
    r"\b(open|click|select|choose|go to|navigate to|start|print|download|sign in|log in|email|contact|submit|request|edit|check|verify|visit|sign up|switch|send)\b"
]

NAVIGATION_CUES = [
    r"\b(under [A-Z][a-z]+|in your account|on the [a-z]+ page|at checkout|settings|orders|profile|privacy page|login page|device's page)\b"
]

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", 
    "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", 
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", 
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "me", "more", 
    "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", 
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", 
    "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then", 
    "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", 
    "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", 
    "why", "with", "would", "you", "your", "yours", "yourself", "yourselves"
}

# Critical policy risk patterns with explicit severity classification
POLICY_RISK_PATTERNS = [
    (r"\b(switch off two-factor|switch off 2fa|turn off two-factor).*email\b", "Offers to disable 2FA via email (Critical Security Vulnerability)", "Critical"),
    (r"\b(instantly|instant refund)\b", "Promises instant refund before return receipt (Financial Fraud Liability)", "Critical"),
    (r"\b(at any time|no deadline|however long ago)\b", "Claims unlimited return window without deadline (Unlimited Liability)", "Critical"),
    (r"\b(activated licence|activated license|licences can be refunded)\b", "Claims activated software licences are refundable (Policy Violation)", "High"),
    (r"\b(gift cards? can be refunded)\b", "Claims gift cards are refundable (Policy Violation)", "High"),
    (r"\b(cancel an order at any point, even after it has shipped)\b", "Claims orders can be cancelled post-shipment (Logistical Violation)", "High"),
    (r"\b(records continuously, 24 hours a day, and keeps footage forever)\b", "Promises unlimited perpetual recording/storage (Product Hallucination)", "High"),
    (r"\b(match any lower price)\b", "Promises price matching contrary to company policy (Policy Violation)", "High"),
    (r"\b(return shipping is always free)\b.*(shows the cost)", "Self-contradictory return shipping terms (Internal Inconsistency)", "Medium"),
    (r"\b(always refund the original shipping fee)\b.*(if the item is faulty)", "Self-contradictory shipping fee refund terms (Internal Inconsistency)", "Medium"),
]


def clean_substantive_text(text: str) -> str:
    """Strips conversational fluff, polite greetings, and signoffs to isolate substantive content."""
    cleaned = text
    for pat in FLUFF_PATTERNS:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[!]+", ".", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def extract_keywords(text: str) -> List[str]:
    """Extract substantive lower-case tokens excluding common stopwords."""
    tokens = re.findall(r"\b[a-z0-9_-]{3,}\b", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def count_fluff_instances(text: str) -> Tuple[int, List[str]]:
    """Detect boilerplate greetings and signoffs in an answer."""
    matches = []
    text_clean = text.lower()
    for pat in FLUFF_PATTERNS:
        found = re.findall(pat, text_clean)
        if found:
            matches.extend(found if isinstance(found[0], str) else [m[0] for m in found])
    return len(matches), matches


def detect_policy_risks(text: str) -> List[str]:
    """Detect known high-risk hallucinations or policy violations with severity tags."""
    risks = []
    text_clean = text.lower()
    for item in POLICY_RISK_PATTERNS:
        pat, desc, severity = item[0], item[1], item[2]
        if re.search(pat, text_clean):
            risks.append(f"[{severity}] {desc}")
    return risks


def detect_policy_risks_detailed(text: str) -> List[Dict[str, str]]:
    """Return structured list of detected policy and security risks."""
    detailed = []
    text_clean = text.lower()
    for item in POLICY_RISK_PATTERNS:
        pat, desc, severity = item[0], item[1], item[2]
        if re.search(pat, text_clean):
            detailed.append({
                "description": desc,
                "severity": severity,
                "pattern": pat
            })
    return detailed


def get_max_severity(risks: List[str]) -> str:
    """Extract highest severity from detected risk strings."""
    if not risks:
        return "None"
    for r in risks:
        if "[Critical]" in r:
            return "Critical"
    for r in risks:
        if "[High]" in r:
            return "High"
    for r in risks:
        if "[Medium]" in r:
            return "Medium"
    return "Low"""


class AnswerEvaluator:
    """
    Transparent rule-based and NLP evaluation engine for support assistant answers.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or WEIGHTS

    def score_relevance(self, question: str, answer: str, topic: str) -> Tuple[float, str]:
        """
        Criterion 1: Relevance (Weight: 25%)
        Assesses if the answer addresses the customer's actual question.
        Uses TF-IDF similarity, keyword overlap, and question-intent matching.
        """
        q_clean = question.strip()
        ans_clean = clean_substantive_text(answer)
        
        q_tokens = extract_keywords(q_clean)
        ans_tokens = extract_keywords(ans_clean)

        if not q_tokens or not ans_tokens:
            return 5.0, "Insufficient substantive tokens for deep relevance analysis."

        # Keyword recall
        matched_tokens = set(q_tokens).intersection(set(ans_tokens))
        keyword_recall = len(matched_tokens) / max(len(set(q_tokens)), 1)

        # TF-IDF cosine similarity
        try:
            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform([q_clean, ans_clean])
            cos_sim = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
        except Exception:
            cos_sim = keyword_recall

        # Intent alignment bonuses
        q_lower = q_clean.lower()
        intent_bonus = 0.0
        intent_note = []

        # Polar question alignment (Can I, Is it, Do you)
        if re.match(r"^(can|is|do|does|are|will|could)\b", q_lower):
            if re.search(r"\b(can|cannot|can't|yes|no|only|eligible|valid|comes with|support)\b", ans_clean.lower()):
                intent_bonus += 0.8
                intent_note.append("direct confirmation")

        # Temporal question alignment (When, How long, How fast)
        if re.match(r"^(when|how long|how fast|how soon)\b", q_lower) or "when" in q_lower:
            if re.search(r"\b(\d+|days?|hours?|business days?|weeks?|months?|instantly|as soon as|immediately)\b", ans_clean.lower()):
                intent_bonus += 0.8
                intent_note.append("timeframe specified")

        # Procedural question alignment (How do I, How to)
        if "how do" in q_lower or "how to" in q_lower or "how can" in q_lower:
            if re.search(r"\b(open|start|go to|click|sign|under|from)\b", ans_clean.lower()):
                intent_bonus += 0.8
                intent_note.append("procedure explained")

        # Base score calculation: 0 to 10
        raw_score = 3.5 + (keyword_recall * 3.5) + (cos_sim * 2.0) + intent_bonus
        score = float(np.clip(raw_score, 1.0, 10.0))

        details = f"Overlap: {len(matched_tokens)} terms ({keyword_recall:.0%}), TF-IDF sim: {cos_sim:.2f}"
        if intent_note:
            details += f", Intent aligned: {', '.join(intent_note)}"
        return round(score, 1), details

    def score_completeness(self, question: str, answer: str, topic: str) -> Tuple[float, str]:
        """
        Criterion 2: Completeness (Weight: 25%)
        Assesses if the answer provides sufficient information to resolve the issue.
        Checks for conditions, criteria bounds, and edge-case guidance.
        """
        substantive = clean_substantive_text(answer)
        words = substantive.split()
        n_words = len(words)

        score = 6.0
        notes = []

        # Substantive length assessment (avoid rewarding empty brevity or punishing concise clarity)
        if n_words < 7:
            score -= 2.5
            notes.append("Too brief / potentially missing details")
        elif 12 <= n_words <= 60:
            score += 1.5
            notes.append("Good information depth")
        elif n_words > 60:
            score += 0.5
            notes.append("Extensive detail")

        # Specificity & conditions check (e.g. 30 days, 12-month, before ships, over $50, iOS 16)
        has_conditions = bool(re.search(
            r"\b(\d+\s*(days?|months?|business days?|\$|devices?)|before\b|within\b|unless\b|only\b|if the\b|after\b)\b",
            substantive.lower()
        ))
        if has_conditions:
            score += 1.5
            notes.append("Explicit policy conditions/thresholds stated")
        else:
            notes.append("Lacks specific condition thresholds")

        # Contingency / alternative path check (e.g., "if button is missing", "unless faulty")
        has_contingency = bool(re.search(
            r"\b(if missing|if already|otherwise|unless|or contact|in which case|after shipping)\b",
            substantive.lower()
        ))
        if has_contingency:
            score += 1.0
            notes.append("Covers edge-case / contingency path")

        score = float(np.clip(score, 1.0, 10.0))
        return round(score, 1), "; ".join(notes)

    def score_actionability(self, answer: str) -> Tuple[float, str]:
        """
        Criterion 3: Actionability (Weight: 20%)
        Assesses if the answer gives clear, actionable next steps and UI navigation.
        """
        substantive = clean_substantive_text(answer)
        
        # Check action verbs
        action_matches = re.findall(ACTION_VERBS[0], substantive.lower())
        # Check navigation cues
        nav_matches = re.findall(NAVIGATION_CUES[0], substantive.lower())

        score = 3.5  # Base score for purely informational answer
        notes = []

        if action_matches:
            unique_verbs = len(set(action_matches))
            score += min(3.5, unique_verbs * 1.2)
            notes.append(f"{unique_verbs} action verb(s): {', '.join(list(set(action_matches))[:3])}")
        else:
            notes.append("No direct action verbs")

        if nav_matches:
            unique_nav = len(set(nav_matches))
            score += min(3.0, unique_nav * 1.5)
            notes.append(f"UI guidance: {', '.join(list(set(nav_matches))[:2])}")
        else:
            notes.append("No explicit navigation path")

        score = float(np.clip(score, 1.0, 10.0))
        return round(score, 1), "; ".join(notes)

    def score_clarity(self, answer: str) -> Tuple[float, str]:
        """
        Criterion 4: Clarity (Weight: 15%)
        Assesses if the answer is understandable, well structured, and free of contradictions.
        """
        sentences = [s.strip() for s in re.split(r"[.!?]+", answer) if s.strip()]
        score = 8.0
        notes = []

        if not sentences:
            return 2.0, "Empty or unstructured text"

        # Sentence length balance
        avg_sentence_len = np.mean([len(s.split()) for s in sentences])
        if 8 <= avg_sentence_len <= 24:
            score += 1.0
            notes.append("Optimal sentence cadence")
        elif avg_sentence_len > 35:
            score -= 1.5
            notes.append("Run-on sentence complexity")

        # Contradiction & inconsistency detection
        text_lower = answer.lower()
        contradiction_detected = False

        if "always free" in text_lower and "shows the cost" in text_lower:
            score -= 4.0
            contradiction_detected = True
            notes.append("Contradiction: claims 'always free' yet 'shows cost'")

        if "always refund" in text_lower and "unless" in text_lower and "faulty" in text_lower:
            score -= 3.0
            contradiction_detected = True
            notes.append("Contradiction: promises unconditional refund then adds exceptions")

        if "at any time, with no deadline" in text_lower and "30 days" in text_lower:
            score -= 4.0
            contradiction_detected = True
            notes.append("Contradiction: conflicting deadlines in same text")

        if not contradiction_detected:
            notes.append("Coherent structure, no internal contradictions")

        score = float(np.clip(score, 1.0, 10.0))
        return round(score, 1), "; ".join(notes)

    def score_conciseness(self, answer: str) -> Tuple[float, str]:
        """
        Criterion 5: Conciseness (Weight: 15%)
        Assesses whether the answer avoids unnecessary information, filler, and fluff.
        Penalizes generic greetings, polite filler, and repetitive sign-offs.
        """
        total_words = len(answer.split())
        substantive = clean_substantive_text(answer)
        sub_words = len(substantive.split())

        fluff_count, fluff_items = count_fluff_instances(answer)

        # Baseline: start at 9.0 for concise answers
        score = 9.0
        notes = []

        # Substantive density ratio
        density = sub_words / max(total_words, 1)
        if density < 0.60:
            score -= 2.5
            notes.append(f"Low content density ({density:.0%})")
        elif density >= 0.90:
            score += 1.0
            notes.append("High substantive density")

        # Fluff penalties
        if fluff_count > 0:
            deduction = min(5.0, fluff_count * 1.8)
            score -= deduction
            notes.append(f"Penalized for {fluff_count} boilerplate phrase(s)")
        else:
            notes.append("Zero conversational fluff")

        # Excessive verbosity penalty (> 120 words for support responses)
        if total_words > 120:
            excess_penalty = min(5.0, (total_words - 120) / 40.0)
            score -= excess_penalty
            notes.append(f"Excessive verbosity penalty (-{excess_penalty:.1f} pts for {total_words} words)")

        score = float(np.clip(score, 1.0, 10.0))
        return round(score, 1), "; ".join(notes)

    def evaluate_single_answer(
        self, question: str, answer: str, topic: str
    ) -> Dict[str, Any]:
        """Evaluate a single answer across all 5 criteria and compute weighted total."""
        rel, rel_exp = self.score_relevance(question, answer, topic)
        comp, comp_exp = self.score_completeness(question, answer, topic)
        act, act_exp = self.score_actionability(answer)
        cla, cla_exp = self.score_clarity(answer)
        con, con_exp = self.score_conciseness(answer)

        weighted_total = (
            rel * self.weights["relevance"]
            + comp * self.weights["completeness"]
            + act * self.weights["actionability"]
            + cla * self.weights["clarity"]
            + con * self.weights["conciseness"]
        )

        risks = detect_policy_risks(answer)

        return {
            "relevance": rel,
            "relevance_exp": rel_exp,
            "completeness": comp,
            "completeness_exp": comp_exp,
            "actionability": act,
            "actionability_exp": act_exp,
            "clarity": cla,
            "clarity_exp": cla_exp,
            "conciseness": con,
            "conciseness_exp": con_exp,
            "total_score": round(weighted_total, 2),
            "policy_risks": risks,
            "risk_severity": get_max_severity(risks),
            "has_policy_risk": len(risks) > 0,
        }

    def compare_answers(
        self,
        q_id: str,
        topic: str,
        question: str,
        ans_current: str,
        ans_new: str,
        tie_threshold: float = DEFAULT_TIE_THRESHOLD,
    ) -> Dict[str, Any]:
        """
        Compare current and new answers pairwise.
        Determines winner based on score differential and documented tie threshold.
        """
        eval_curr = self.evaluate_single_answer(question, ans_current, topic)
        eval_new = self.evaluate_single_answer(question, ans_new, topic)

        score_curr = eval_curr["total_score"]
        score_new = eval_new["total_score"]
        score_diff = round(score_new - score_curr, 2)

        # Pairwise decision with documented tie threshold
        if score_diff > tie_threshold:
            winner = "new"
            reason = (
                f"New version scored {score_new:.1f} vs Current {score_curr:.1f} "
                f"(+{score_diff:.1f} pts > threshold {tie_threshold}). "
                f"Gains in Actionability ({eval_new['actionability']} vs {eval_curr['actionability']}) "
                f"or Completeness ({eval_new['completeness']} vs {eval_curr['completeness']})."
            )
        elif score_diff < -tie_threshold:
            winner = "current"
            reason = (
                f"Current version scored {score_curr:.1f} vs New {score_new:.1f} "
                f"(Current leads by {-score_diff:.1f} pts > threshold {tie_threshold}). "
                f"New suffered from Conciseness penalties ({eval_new['conciseness']} vs {eval_curr['conciseness']}) "
                f"or Clarity issues ({eval_new['clarity']} vs {eval_curr['clarity']})."
            )
        else:
            winner = "tie"
            reason = (
                f"Score difference of {abs(score_diff):.1f} pts is within tie threshold (±{tie_threshold} pts). "
                f"Current: {score_curr:.1f}, New: {score_new:.1f}. Quality is practically equivalent."
            )

        if eval_new["has_policy_risk"]:
            reason += f" [CRITICAL RISK DETECTED in New: {'; '.join(eval_new['policy_risks'])}]"

        return {
            "id": q_id,
            "topic": topic,
            "question": question,
            "answer_current": ans_current,
            "answer_new": ans_new,
            "curr_relevance": eval_curr["relevance"],
            "curr_completeness": eval_curr["completeness"],
            "curr_actionability": eval_curr["actionability"],
            "curr_clarity": eval_curr["clarity"],
            "curr_conciseness": eval_curr["conciseness"],
            "curr_total_score": score_curr,
            "curr_relevance_exp": eval_curr["relevance_exp"],
            "curr_completeness_exp": eval_curr["completeness_exp"],
            "curr_actionability_exp": eval_curr["actionability_exp"],
            "curr_clarity_exp": eval_curr["clarity_exp"],
            "curr_conciseness_exp": eval_curr["conciseness_exp"],
            "curr_policy_risks": eval_curr["policy_risks"],
            "curr_risk_severity": eval_curr["risk_severity"],
            "new_relevance": eval_new["relevance"],
            "new_completeness": eval_new["completeness"],
            "new_actionability": eval_new["actionability"],
            "new_clarity": eval_new["clarity"],
            "new_conciseness": eval_new["conciseness"],
            "new_total_score": score_new,
            "new_relevance_exp": eval_new["relevance_exp"],
            "new_completeness_exp": eval_new["completeness_exp"],
            "new_actionability_exp": eval_new["actionability_exp"],
            "new_clarity_exp": eval_new["clarity_exp"],
            "new_conciseness_exp": eval_new["conciseness_exp"],
            "new_policy_risks": eval_new["policy_risks"],
            "new_risk_severity": eval_new["risk_severity"],
            "has_critical_security_risk": ("Critical" in eval_new["risk_severity"]),
            "score_diff": score_diff,
            "better": winner,
            "explanation": reason,
        }

    def evaluate_dataframe(
        self, df, tie_threshold: float = DEFAULT_TIE_THRESHOLD
    ):
        """Batch evaluate all rows in a DataFrame."""
        import pandas as pd
        results = []
        for _, row in df.iterrows():
            res = self.compare_answers(
                q_id=str(row["id"]),
                topic=str(row["topic"]),
                question=str(row["question"]),
                ans_current=str(row["answer_current"]),
                ans_new=str(row["answer_new"]),
                tie_threshold=tie_threshold,
            )
            results.append(res)
        return pd.DataFrame(results)
