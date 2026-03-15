"""
Treatment registry: canonical display and semantic properties for each condition.

A treatment is identified by (face_id, eyes_covered). Sessions carry all these
properties at load time — plots read them directly, no lookup at render time.

Properties:
  label             — display label for plots
  color             — bar/plot color
  image             — filename (ID008.png for sighted; ID008_blindfold.png for blindfold)
  gender            — 'female' | 'male'
  sight_type        — 'sighted' | 'blindfold'
  threat_level      — 'low' | 'high'
  audience_rating   — Likert 1–5 from audience (face feedback study)

To add a new condition: add an entry to _TREATMENTS with (face_id, eyes_covered) key.
Colors and images flow automatically to all plots.
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMG_DIR = os.path.join(ROOT, 'data', 'tube', 'images')


# (face_id, eyes_covered) → display + semantic properties
# Images: sighted = ID008.png (no suffix); blindfold = ID008_blindfold.png
# Audience ratings: Likert 1–5 (audience rated faces for threat/feedback)
_TREATMENTS = {
    ('ID008', False): dict(
        label='Eyes Open',
        color='#4A90D9',
        image='ID008.png',
        gender='female',
        sight_type='sighted',
        threat_label='low',
        threat_rating=1.75,
    ),
    ('ID008', True): dict(
        label='Eyes Covered',
        color='#A0A0A0',
        image='ID008_blindfold.png',
        gender='female',
        sight_type='blindfold',
        threat_label='low',
        threat_rating=1.75,
    ),
    ('ID015', False): dict(
        label='ID015\n(low threat)',
        color='#7EC88B',
        image='ID015.png',
        gender='male',
        sight_type='sighted',
        threat_label='low',
        threat_rating=1.78,
    ),
    ('ID017', False): dict(
        label='ID017\n(high threat)',
        color='#D9534F',
        image='ID017.png',
        gender='male',
        sight_type='sighted',
        threat_label='high',
        threat_rating=4.43,
    ),
    ('ID030', False): dict(
        label='ID030\n(high threat)',
        color='#C06040',
        image='ID030.png',
        gender='male',
        sight_type='sighted',
        threat_label='high',
        threat_rating=4.50,
    ),
}


def resolve(face_id: str, eyes_covered) -> dict:
    """
    Return all treatment properties for a (face_id, eyes_covered) key.

    Returns dict with: label, color, image_path, gender, sight_type, threat_level.
    image_path is absolute and verified to exist (None if file missing).
    Falls back to defaults for unknown treatments.
    """
    if isinstance(eyes_covered, (bool, type(None))):
        key = (face_id, eyes_covered)
    else:
        key = (face_id, bool(eyes_covered))

    t = _TREATMENTS.get(key)
    if t is None:
        return dict(
            label=str(face_id), color='#888888', image_path=None,
            gender=None, sight_type=None, threat_level=None,
            threat_rating=None,
        )
    img_path = os.path.join(IMG_DIR, t['image'])
    return dict(
        label=t['label'],
        color=t['color'],
        image_path=img_path if os.path.exists(img_path) else None,
        gender=t['gender'],
        sight_type=t['sight_type'],
        threat_level=t['threat_label'],
        threat_rating=t.get('threat_rating'),
    )
