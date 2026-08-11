"""
Rule-based construction safety recommendations.

Maps construction activities to general preventive safety controls.
These are educational MVP recommendations and do not replace
site-specific risk assessments or qualified safety professionals.
"""

RECOMMENDATIONS = {
    "Working at Height": [
        "Verify fall protection controls before starting work.",
        "Check edge protection and guardrails.",
        "Inspect ladders, scaffolds, and access equipment.",
        "Conduct a pre-task safety briefing."
    ],

    "Lifting": [
        "Verify lifting equipment inspection status.",
        "Establish and maintain an exclusion zone.",
        "Confirm the load and equipment capacity.",
        "Conduct a pre-task lifting briefing."
    ],

    "Scaffolding": [
        "Inspect scaffold condition before use.",
        "Verify safe access and egress.",
        "Check required edge protection.",
        "Remove obstructions from scaffold platforms."
    ],

    "Excavation": [
        "Inspect excavation conditions before work begins.",
        "Verify appropriate barricading.",
        "Ensure safe access and egress.",
        "Review relevant site hazards before starting work."
    ],

    "Electrical Work": [
        "Inspect electrical cables and equipment.",
        "Apply appropriate isolation procedures where required.",
        "Restrict access to hazardous electrical areas.",
        "Verify appropriate protective equipment."
    ],

    "Material Handling": [
        "Use appropriate material-handling procedures.",
        "Keep pathways clear of obstructions.",
        "Verify loads are stable before movement.",
        "Conduct a pre-task safety briefing."
    ],

    "Welding": [
        "Inspect welding equipment before use.",
        "Keep combustible materials away from the work area.",
        "Verify appropriate protective equipment.",
        "Maintain suitable ventilation and work-area controls."
    ],

    "Confined Space": [
        "Review the confined-space hazards before entry.",
        "Verify required atmospheric and access controls.",
        "Maintain appropriate communication and emergency arrangements.",
        "Use required protective equipment."
    ],

    "Vehicle Movement": [
        "Separate vehicles and pedestrians where possible.",
        "Verify designated traffic routes.",
        "Maintain clear visibility around moving equipment.",
        "Conduct a pre-task traffic safety briefing."
    ],

    "Housekeeping": [
        "Keep walkways and work areas clear.",
        "Remove unnecessary materials and obstructions.",
        "Store materials safely.",
        "Inspect the area regularly for housekeeping hazards."
    ]
}


DEFAULT_RECOMMENDATIONS = [
    "Review the activity hazards before starting work.",
    "Verify appropriate safety controls are in place.",
    "Conduct a pre-task safety briefing.",
    "Ensure required protective equipment is available and appropriate."
]


def get_recommendations(activity_type):
    """
    Return preventive controls for the given construction activity.
    """
    return RECOMMENDATIONS.get(activity_type, DEFAULT_RECOMMENDATIONS)


if __name__ == "__main__":
    print("Testing recommendations.py\n")

    activity = "Working at Height"

    print(f"Activity: {activity}")
    print("Recommended controls:")

    for recommendation in get_recommendations(activity):
        print(f"- {recommendation}")

    print("\nrecommendations.py self-test passed.")