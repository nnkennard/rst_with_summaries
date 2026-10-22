import re

from typing import Optional, cast

_COARSE = {
    "Attribution": ["attribution", "attribution-negative"],
    "Background": ["background", "circumstance"],
    "Cause": [
        "cause", "result", "consequence", "consequence-n", "consequence-s",
        "cause-result"
    ],
    "Comparison": ["comparison", "preference", "analogy", "proportion"],
    "Condition": ["condition", "hypothetical", "contingency", "otherwise"],
    "Contrast": ["contrast", "concession", "antithesis"],
    "Elaboration": [
        "elaboration-additional", "elaboration-general-specific",
        "elaboration-part-whole", "elaboration-process-step",
        "elaboration-object-attribute", "elaboration-set-member", "example",
        "definition"
    ],
    "Enablement": ["purpose", "enablement"],
    "Evaluation": [
        "evaluation", "evaluation-n", "evaluation-s", "interpretation",
        "interpretation-n", "interpretation-s", "conclusion", "comment"
    ],
    "Explanation": ["evidence", "explanation-argumentative", "reason"],
    "Joint": ["list", "disjunction"],
    "Manner-Means": ["manner", "means"],
    "Topic-Comment": [
        "problem-solution", "problem-solution-n", "problem-solution-s",
        "question-answer", "question-answer-n", "question-answer-s",
        "statement-response", "statement-response-n", "statement-response-s",
        "topic-comment", "comment-topic", "rhetorical-question"
    ],
    "Summary": ["summary", "summary-n", "summary-s", "restatement"],
    "Temporal": [
        "temporal-before", "temporal-after", "temporal-same-time", "sequence",
        "inverted-sequence"
    ],
    "Topic-Change": ["topic-shift", "topic-drift"],
    "Textual-Organization": ["textualorganization", "textual-organization"],
    "Same-Unit": ["same-unit"],
}

FINE2COARSE = {fine: coarse for coarse, fines in _COARSE.items() for fine in fines}

ERROR_INDICATOR = "//TT_ERR"


# EDU text (_!..._!) is matched first because it may contain parentheses.
_TOKEN_RE = re.compile(r"_!.*?_!|\(|\)|[^\s()]+", re.S)

def is_nuclearity(label:str) -> bool:
    return label in "Root Nucleus Satellite".split()

def listify_dis(filename: str) -> Optional[list]:
    with open(filename, encoding="utf-8", errors="replace") as f:
        text = f.read()

    tokens = _TOKEN_RE.findall(text)
    if ERROR_INDICATOR in tokens:
        return None

    stack = [[]]
    for tok in tokens:
        if tok == "(":
            stack.append([])
        elif tok == ")":
            done = stack.pop()
            stack[-1].append(done)
        else:
            stack[-1].append(tok)
    if len(stack) != 1 or len(stack[0]) != 1:
        raise ValueError("Unbalanced parentheses in .dis file")
    return stack[0][0]
