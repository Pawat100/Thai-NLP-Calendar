"""
Validation utilities for event data
Ensures data completeness and provides safe defaults
"""

from typing import Dict, Tuple, List
from datetime import datetime


def validate_event_data(slots: Dict) -> Tuple[bool, List[str], Dict]:
    """
    Validate extracted event slots
    
    Args:
        slots: Dictionary with date, time, description, attendees, location
    
    Returns:
        (is_valid, missing_fields, safe_defaults)
        - is_valid: True if minimum requirements met
        - missing_fields: List of critical missing fields to ask user
        - safe_defaults: Dictionary of safe auto-fill values
    """
    missing_critical = []
    safe_defaults = {}
    
    # Critical field checks
    has_activity = bool(slots.get('description'))
    has_date = bool(slots.get('date'))
    
    # Minimum requirement: MUST have activity OR date
    is_valid = has_activity or has_date
    
    # Track what's missing
    if not has_activity:
        missing_critical.append('activity')
    
    if not has_date:
        missing_critical.append('date')
    
    # Safe defaults (only for non-critical fields)
    if not slots.get('time'):
        safe_defaults['time'] = '09:00'
    
    if not slots.get('location'):
        safe_defaults['location'] = '-'
    
    return is_valid, missing_critical, safe_defaults


def apply_safe_defaults(slots: Dict, defaults: Dict) -> Dict:
    """
    Apply safe default values to slots
    
    Args:
        slots: Original slots dictionary
        defaults: Dictionary of default values to apply
    
    Returns:
        Updated slots with defaults applied
    """
    updated_slots = slots.copy()
    
    for field, default_value in defaults.items():
        if not updated_slots.get(field):
            updated_slots[field] = default_value
    
    return updated_slots


def format_missing_fields_message(missing_fields: List[str]) -> str:
    """
    Create user-friendly message for missing fields
    Ask for ONE field at a time to avoid confusion
    
    Args:
        missing_fields: List of field names
    
    Returns:
        Thai language message asking for the FIRST missing field only
    """
    if not missing_fields:
        return ""
    
    messages = {
        'activity': 'กิจกรรมคืออะไรคะ? (เช่น ประชุม, เรียน, นัดหมาย)',
        'date': 'วันไหนคะ? (เช่น พรุ่งนี้, วันจันทร์, 15 กุมภาพันธ์)',
        'time': 'เวลาเท่าไหร่คะ? (เช่น 10 โมง, บ่าย 2 โมง)',
    }
    
    # Return only the FIRST missing field
    first_missing = missing_fields[0]
    return messages.get(first_missing, f'{first_missing}?')


def is_event_saveable(slots: Dict) -> bool:
    """
    Final check before saving event
    Must have BOTH activity AND date to save
    
    Args:
        slots: Event slots dictionary
    
    Returns:
        True if event can be saved, False otherwise
    """
    has_activity = bool(slots.get('description'))
    has_date = bool(slots.get('date'))
    
    # CRITICAL: Do not save if both are missing
    return has_activity and has_date
