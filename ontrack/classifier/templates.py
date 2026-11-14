"""Fixed safe responses used when the triage gate blocks free generation."""

EMERGENCY_TEMPLATE = (
    "This could be a medical emergency. Please get help now:\n"
    "- Call 112 or 193 (National Ambulance Service, Ghana), or go to the nearest hospital emergency unit.\n"
    "- If someone may have low blood sugar and is awake and able to swallow, give a sugary drink or sugar "
    "(about 15 g of fast sugar, e.g. half a cup of regular soft drink or juice, or 1 tablespoon of sugar or honey), "
    "then recheck after 15 minutes.\n"
    "- If the person is unconscious, having a seizure or cannot swallow, do NOT put food or drink in their mouth. "
    "Lay them on their side and call for help.\n"
    "- Vomiting, fruity-smelling breath, fast deep breathing or very high readings need urgent hospital care.\n"
    "OnTrack is an education tool and cannot manage emergencies."
)

CLINICIAN_TEMPLATE = (
    "Thank you for asking. This question is about diagnosis, test results, or your medicines or insulin doses, "
    "and the safe answer depends on your personal health details. OnTrack cannot diagnose or change any medicine "
    "or dose. Please speak with your doctor, nurse, diabetes educator or pharmacist before making any change, "
    "and do not stop or change your medicine on your own. If you feel very unwell, call 112 or go to the nearest "
    "hospital. I can still help with general questions about food, portions, exercise and understanding diabetes."
)


def template_for(label: str) -> str:
    return {"emergency": EMERGENCY_TEMPLATE, "needs_clinician": CLINICIAN_TEMPLATE}[label]
