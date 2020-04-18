import math

def geo2Angle(lat1, lon1, lat2, lon2):
    dLon = (lon2 - lon1)

    y = math.sin(dLon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)

    brng = math.atan2(y, x)

    brng = math.degrees(brng)
    brng = (brng + 360) % 360
    #brng = 360 - brng # count degrees clockwise - remove to make counter-clockwise
    return brng