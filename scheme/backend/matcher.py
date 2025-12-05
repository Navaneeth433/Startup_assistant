import json

def match_schemes(startup_profile, scheme_file="schemes.json"):
    with open(scheme_file, "r") as f:
        schemes = json.load(f)

    matched = []
    for scheme in schemes:
        eligibility = scheme["eligibility"]
        match = True

        # Domain
        if "all" not in eligibility["domain"]:
            if startup_profile.get("domain") not in eligibility["domain"]:
                match = False

        # Registration
        if startup_profile.get("registration") not in eligibility["registration"]:
            match = False

        # Stage
        if startup_profile.get("stage") not in eligibility["stage"]:
            match = False

        if match:
            matched.append({
                "name": scheme["name"],
                "benefits": scheme["benefits"],
                "link": scheme["link"]
            })
    return matched
