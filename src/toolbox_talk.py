"""
Rule-based toolbox-talk topics for construction safety activities.
"""

TOOLBOX_TOPICS = {
    "Working at Height": [
        "Fall protection",
        "Edge protection",
        "Safe scaffold and ladder access",
        "Pre-use inspection of access equipment",
        "Emergency response for elevated work"
    ],

    "Lifting": [
        "Safe lifting practices",
        "Lifting equipment inspection",
        "Exclusion zones",
        "Load capacity and stability",
        "Communication during lifting operations"
    ],

    "Scaffolding": [
        "Scaffold inspection",
        "Safe scaffold access",
        "Edge protection",
        "Platform housekeeping",
        "Safe use of scaffolding"
    ],

    "Excavation": [
        "Excavation hazards",
        "Barricading and access control",
        "Safe access and egress",
        "Excavation inspections",
        "Emergency response"
    ],

    "Electrical Work": [
        "Electrical hazard awareness",
        "Cable and equipment inspection",
        "Energy isolation",
        "Electrical area access control",
        "Appropriate protective equipment"
    ],

    "Material Handling": [
        "Safe material handling",
        "Manual handling hazards",
        "Load stability",
        "Clear pathways",
        "Material storage"
    ],

    "Welding": [
        "Welding hazards",
        "Fire prevention",
        "Welding equipment inspection",
        "Protective equipment",
        "Ventilation and work-area controls"
    ],

    "Confined Space": [
        "Confined-space hazards",
        "Atmospheric hazards",
        "Access and communication",
        "Emergency arrangements",
        "Required protective equipment"
    ],

    "Vehicle Movement": [
        "Vehicle and pedestrian separation",
        "Site traffic routes",
        "Visibility around vehicles",
        "Safe vehicle movement",
        "Traffic safety briefing"
    ],

    "Housekeeping": [
        "Preventing slips and trips",
        "Clear walkways",
        "Safe material storage",
        "Removing obstructions",
        "Routine housekeeping inspections"
    ]
}


DEFAULT_TOPICS = [
    "General construction hazard awareness",
    "Pre-task safety planning",
    "Use of appropriate protective equipment",
    "Emergency preparedness",
    "Reporting unsafe conditions"
]


def get_toolbox_topics(activity_type, count=5):
    """
    Return toolbox-talk topics for an activity.
    """
    topics = TOOLBOX_TOPICS.get(activity_type, DEFAULT_TOPICS)
    return topics[:count]


if __name__ == "__main__":
    print("Testing toolbox_talk.py\n")

    activity = "Working at Height"

    print(f"Activity: {activity}")
    print("Toolbox-talk topics:")

    for topic in get_toolbox_topics(activity):
        print(f"- {topic}")

    print("\ntoolbox_talk.py self-test passed.")