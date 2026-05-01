"""
Shared EPPP domain constants — Single Source of Truth for Python scripts.

ALL scripts that reference domain names MUST import from here instead of
hardcoding. Canonical source: EPPP-Domain-Design/anchor_points_by_domain/
"""

DOMAIN_CODES = {
    1: "PMET", 2: "LDEV", 3: "CPAT", 4: "PTHE", 5: "SOCU",
    6: "WDEV", 7: "BPSY", 8: "CASS", 9: "PETH",
}

CODE_TO_ID = {v: k for k, v in DOMAIN_CODES.items()}

DOMAIN_NAMES = {
    "PMET": "Psychometrics & Research Methods",
    "LDEV": "Lifespan & Developmental Stages",
    "CPAT": "Clinical Psychopathology",
    "PTHE": "Psychotherapy Models & Interventions",
    "SOCU": "Social & Cultural Psychology",
    "WDEV": "Workforce Development & Leadership",
    "BPSY": "Biopsychology",
    "CASS": "Clinical Assessment & Interpretation",
    "PETH": "Psychopharmacology & Ethics",
}

ALL_CODES = list(CODE_TO_ID.keys())
