"""
NLP Utilities for Thai Calendar Event Extraction

This module provides functions extracted from THENLP.ipynb for use in the Streamlit app.
Includes NER, POS validation, slot mapping, and date/time parsing.
"""

import spacy
import json
import re
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import pytz
try:
    from pythainlp import normalize
    import dateparser
except ImportError:
    print("Warning: pythainlp or dateparser not installed")

# Constants
TZ = pytz.timezone('Asia/Bangkok')
EVENTS_FILE = "events.json"

# Global NLP model (loaded once)
_nlp_model = None

# =========================
# Normalization Dictionaries (from NLP_PROJECT.ipynb)
# =========================

SLANG_DICT = {
    # 📅 วัน / วันที่
    "พน.": "พรุ่งนี้",
    "พน": "พรุ่งนี้",
    "มะรืน": "วันถัดไป",
    "มะลืนนี้": "วันถัดไป",
    "มะวาน": "เมื่อวาน",
    
    # ⏰ ช่วงเวลา (Intent)
    "ตอนเช้า": "ช่วงเช้า",
    "เช้า": "ช่วงเช้า",
    "ตอนสาย": "ช่วงสาย",
    "สาย": "ช่วงสาย",
    "ตอนบ่าย": "ช่วงบ่าย",
    "บ่าย": "ช่วงบ่าย",
    "ตอนเย็น": "ช่วงเย็น",
    "เย็น": "ช่วงเย็น",
    "ตอนค่ำ": "ช่วงค่ำ",
    "ค่ำ": "ช่วงค่ำ",
    "ตอนดึก": "ช่วงดึก",
    "ดึก": "ช่วงดึก",
    
    # 🕒 เวลา
    "เที่ยง": "12:00",
    "เที่ยงคืน": "00:00",
    "บ่ายโมง": "13:00",
    "บ่ายสอง": "14:00",
    "บ่ายสาม": "15:00",
    "บ่ายสี่": "16:00",
    "บ่ายห้า": "17:00",
    "หกโมงเย็น": "18:00",
    "หนึ่งทุ่ม": "19:00",
    "สองทุ่ม": "20:00",
    "สามทุ่ม": "21:00",
    
    # 👤 บุคคล
    "จาร": "อาจารย์",
    "อจ": "อาจารย์",
    "อ.": "อาจารย์",
    "บอส": "ผู้บังคับบัญชา",
    "หัวหน้า": "ผู้บังคับบัญชา",
    
    # 🗣️ กริยา
    "นัดเจอ": "นัดพบ",
    "เจอกัน": "พบ",
    "ไปหา": "ไปพบ",
    "เข้าไปหา": "ไปพบ",
    "คุยงาน": "ประชุม",
    "เข้าไปคุย": "ประชุม",
    "เลื่อนนัด": "เลื่อน",
    "ยกเลิกนัด": "ยกเลิก",
    
    # 🏫 สถานที่
    "มทร.": "มหาวิทยาลัย",
    "มทร": "มหาวิทยาลัย",
    "มอ": "มหาวิทยาลัย",
    "มหาลับ": "มหาวิทยาลัย",
    "ราชมงคลพระนคร": "มหาวิทยาลัย",
    "rmutp": "มหาวิทยาลัย",
    "ตึกเรียน": "อาคารเรียน",
    "ตึก": "อาคาร",
    
    # 🎓 คณะ
    "คณะวิศวะ": "คณะวิศวกรรมศาสตร์",
    "วิศวะ": "คณะวิศวกรรมศาสตร์",
    "คณะบริหาร": "คณะบริหารธุรกิจ",
    "บริหาร": "คณะบริหารธุรกิจ",
    "คณะไอที": "คณะเทคโนโลยีสารสนเทศ",
    "ไอที": "คณะเทคโนโลยีสารสนเทศ",
}

LOANWORD_DICT = {
    # กิจกรรม
    "video call": "โทร",
    "google meet": "ออนไลน์",
    "ms teams": "ออนไลน์",
    "meeting": "ประชุม",
    "meet": "ประชุม",
    "mtg": "ประชุม",
    "meetup": "ประชุม",
    "briefing": "ชี้แจง",
    "brief": "ชี้แจง",
    "presentation": "นำเสนอ",
    "present": "นำเสนอ",
    "review": "ทบทวน",
    "report": "รายงาน",
    "update": "อัปเดต",
    
    # เวลา
    "tomorrow": "พรุ่งนี้",
    "today": "วันนี้",
    "tonight": "คืนนี้",
    "morning": "ช่วงเช้า",
    "afternoon": "ช่วงบ่าย",
    "evening": "ช่วงเย็น",
    
    # ออนไลน์
    "zoom": "ออนไลน์",
    "online": "ออนไลน์",
}

SPLIT_WORD_CORRECTION = {
    ("มหา", "ลับ"): "มหาวิทยาลัย",
    ("มหา", "ลัย"): "มหาวิทยาลัย",
    ("วิศ", "วะ"): "คณะวิศวกรรมศาสตร์",
    ("วิศว", "ะ"): "คณะวิศวกรรมศาสตร์",
    ("โรง", "บาล"): "โรงพยาบาล",
    ("ตอน", "เช้า"): "ช่วงเช้า",
    ("ตอน", "สาย"): "ช่วงสาย",
    ("ตอน", "บ่าย"): "ช่วงบ่าย",
    ("ตอน", "เย็น"): "ช่วงเย็น",
}


def load_ner_model(model_path: str = "./my_ner_model"):
    """
    Load spaCy NER model
    """
    global _nlp_model
    
    if _nlp_model is not None:
        return _nlp_model
    
    try:
        _nlp_model = spacy.load(model_path)
        print(f"✓ Loaded model from {model_path}")
    except OSError:
        print(f"⚠ Model not found at {model_path}, creating blank model")
        _nlp_model = spacy.blank("th")
        ner = _nlp_model.add_pipe("ner")
        for label in ["DATE", "TIME", "ACTIVITY", "EVENT", "PERSON", "LOCATION"]:
            ner.add_label(label)
    
    return _nlp_model


def normalize_thai_text(text: str) -> str:
    """
    Normalize Thai text using pythainlp and custom dictionaries
    Applies: Unicode normalization, slang normalization, loanword conversion
    """
    # Step 1: Basic unicode normalization
    try:
        text = normalize(text)
    except:
        pass
    
    # Step 2: Lowercase for matching
    text_lower = text.lower()
    
    # Step 3: Apply loanword dictionary (case-insensitive)
    for loanword, thai_word in LOANWORD_DICT.items():
        text_lower = text_lower.replace(loanword.lower(), thai_word)
    
    # Step 4: Apply slang dictionary
    for slang, formal in SLANG_DICT.items():
        text_lower = text_lower.replace(slang, formal)
    
    # Step 5: Whitespace cleanup
    text_lower = re.sub(r'\s+', ' ', text_lower).strip()
    
    return text_lower


def get_current_datetime():
    """Get current datetime in Bangkok timezone"""
    return datetime.now(TZ)


def parse_thai_date(date_str: str, reference_date: Optional[datetime] = None) -> Optional[str]:
    """
    Parse Thai date expressions to YYYY-MM-DD format
    Enhanced to handle complex formats like "Monday 10 Jan 69"
    """
    if not date_str:
        return None
    
    if reference_date is None:
        reference_date = get_current_datetime()
    
    date_str = date_str.strip().lower()
    
    # Thai relative dates
    thai_relative_dates = {
        'วันนี้': 0,
        'พรุ่งนี้': 1,
        'มะรืนนี้': 2,
        'วันถัดไป': 2,
        'เ มื่อวาน': -1,
        'เมื่อวานนี้': -1,
        'เมื่อวานซืน': -2,
        'วานนี้': -1,
    }
    
    for thai_word, days in thai_relative_dates.items():
        if thai_word in date_str:
            target_date = reference_date + timedelta(days=days)
            return target_date.strftime('%Y-%m-%d')
    
    # Thai months
    thai_months = {
        'มกราคม': 1, 'ม.ค.': 1, 'กุมภาพันธ์': 2, 'ก.พ.': 2,
        'มีนาคม': 3, 'มี.ค.': 3, 'เมษายน': 4, 'เม.ย.': 4,
        'พฤษภาคม': 5, 'พ.ค.': 5, 'มิถุนายน': 6, 'มิ.ย.': 6,
        'กรกฎาคม': 7, 'ก.ค.': 7, 'สิงหาคม': 8, 'ส.ค.': 8,
        'กันยายน': 9, 'ก.ย.': 9, 'ตุลาคม': 10, 'ต.ค.': 10,
        'พฤศจิกายน': 11, 'พ.ย.': 11, 'ธันวาคม': 12, 'ธ.ค.': 12,
    }
    
    # Extract all numbers
    numbers = re.findall(r'\d+', date_str)
    
    # Try to parse with month (higher priority than weekday alone)
    for thai_month, month_num in thai_months.items():
        if thai_month in date_str:
            day = int(numbers[0]) if numbers else 1
            year = reference_date.year
            year_specified = False  # Track if user gave an explicit year
            
            # Check if year is also specified (2-digit or 4-digit)
            if len(numbers) >= 2:
                year_candidate = int(numbers[1])
                year_specified = True
                # Handle 2-digit year: always treat as B.E. (พ.ศ.) short form
                # e.g. "69" → B.E. 2569 → C.E. 2026
                if year_candidate < 100:
                    current_be_year = reference_date.year + 543
                    current_be_century = (current_be_year // 100) * 100  # e.g. 2500
                    be_year = current_be_century + year_candidate
                    year = be_year - 543
                elif year_candidate > 2500:  # Full Buddhist year (e.g. 2569)
                    year = year_candidate - 543
                else:
                    year = year_candidate
            
            try:
                target_date = datetime(year, month_num, day, tzinfo=TZ)
                # Only auto-advance to next year if user did NOT specify a year
                if not year_specified and target_date < reference_date:
                    target_date = datetime(year + 1, month_num, day, tzinfo=TZ)
                return target_date.strftime('%Y-%m-%d')
            except ValueError:
                pass
    
    # Thai weekdays (fallback if no month specified)
    thai_weekdays = {
        'จันทร์': 0, 'อังคาร': 1, 'พุธ': 2, 'พฤหัสบดี': 3,
        'พฤหัส': 3, 'ศุกร์': 4, 'เสาร์': 5, 'อาทิตย์': 6,
    }
    
    for thai_day, weekday in thai_weekdays.items():
        if thai_day in date_str:
            current_weekday = reference_date.weekday()
            days_ahead = weekday - current_weekday
            if days_ahead <= 0:
                days_ahead += 7
            target_date = reference_date + timedelta(days=days_ahead)
            return target_date.strftime('%Y-%m-%d')
    
    # Fallback to dateparser
    try:
        parsed = dateparser.parse(
            date_str,
            languages=['th', 'en'],
            settings={'TIMEZONE': 'Asia/Bangkok', 'RELATIVE_BASE': reference_date.replace(tzinfo=None)}
        )
        if parsed:
            return parsed.strftime('%Y-%m-%d')
    except:
        pass
    
    return None


def parse_thai_time(time_str: str) -> Optional[str]:
    """
    Parse Thai time expressions to HH:MM format
    Enhanced to handle time ranges - extracts the START time
    Supports both : and . as separators (10:00 or 10.00)
    Examples: "10:00–12:00" -> "10:00", "10.00-12.00" -> "10:00"
    """
    if not time_str:
        return None
    
    time_str = time_str.strip().lower()
    
    # Handle time ranges - extract first time only  
    # Common separators: – (en-dash), - (hyphen), ~, ถึง
    for separator in ['–', '-', '~', 'ถึง', 'to']:
        if separator in time_str:
            parts = time_str.split(separator)
            if parts:
                time_str = parts[0].strip()  # Take only the start time
            break
    
    # Find time patterns - support both : and . as separators
    # Pattern 1: HH:MM or HH.MM
    time_pattern = re.findall(r'(\d{1,2})[:.](\d{2})', time_str)
    if time_pattern:
        hour, minute = int(time_pattern[0][0]), int(time_pattern[0][1])
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return f"{hour:02d}:{minute:02d}"
    
    # Extract numbers for non-formatted times
    numbers = re.findall(r'\d+', time_str)
    
    hour = 0
    minute = 0
    
    if numbers:
        hour = int(numbers[0])
        if len(numbers) > 1:
            minute = int(numbers[1])
    
    # Thai time period adjustments
    if any(word in time_str for word in ['บ่าย', 'เย็น', 'ค่ำ']):
        if hour < 12:
            hour += 12
    
    if 'ครึ่ง' in time_str:
        minute = 30
    
    if 0 <= hour <= 23 and 0 <= minute <= 59:
        return f"{hour:02d}:{minute:02d}"
    
    return None


def extract_entities_with_pos(text: str, nlp_model=None) -> List[Tuple[str, str, str]]:
    """Extract entities with POS tags for validation"""
    if nlp_model is None:
        nlp_model = load_ner_model()
    
    text = normalize_thai_text(text)
    doc = nlp_model(text)
    
    results = []
    for ent in doc.ents:
        pos_tags = [token.pos_ for token in ent]
        main_pos = pos_tags[0] if pos_tags else "UNKNOWN"
        results.append((ent.text, ent.label_, main_pos))
    
    return results


def split_by_separators(text: str) -> List[str]:
    """
    Split text into multiple event segments using common separators.
    
    Separators include:
    - และ, แล้ว, กับ (Thai 'and', 'then', 'with')
    - and, then (English)
    - Commas, semicolons, slashes
    
    Returns list of text segments
    """
    if not text:
        return []
    
    # Define separator patterns (order matters!)
    separators = [
        r'\s+และ\s+',      # Thai 'and'
        r'\s+แล้ว\s+',     # Thai 'then'  
        r'\s+แล้วก็\s+',   # Thai 'and then'
        r'\s+พร้อม\s+',    # Thai 'along with'
        r'\s+,\s*และ\s+',  # ', and'
        r'\s+;\s*',        # semicolon
        r'\s+/\s+',        # slash separator
        r'\s*,\s+(?=.{10,})', # comma (but only if followed by substantial text)
        r'\s+and\s+',      # English 'and'
        r'\s+then\s+',     # English 'then'
    ]
    
    # Combine all separators into one pattern
    combined_pattern = '|'.join(f'({sep})' for sep in separators)
    
    # Split text
    segments = re.split(combined_pattern, text, flags=re.IGNORECASE)
    
    # Filter out the separator matches themselves and empty strings
    segments = [seg.strip() for i, seg in enumerate(segments) 
                if i % 2 == 0 and seg and seg.strip()]
    
    return segments if segments else [text]


def extract_multiple_events(text: str, nlp_model=None) -> List[Dict[str, any]]:
    """
    Split text by separators and extract multiple events.
    
    Example:
        Input: "ประชุมวันจันทร์ 10 โมง และส่งเอกสารพรุ่งนี้"
        Output: [
            {date: '2026-02-10', time: '10:00', description: 'ประชุม', ...},
            {date: '2026-02-11', time: None, description: 'ส่งเอกสาร', ...}
        ]
    """
    # Split into segments
    segments = split_by_separators(text)
    
    # Process each segment
    events = []
    for segment in segments:
        slots = extract_slots(segment, nlp_model)
        
        # Only add if it has at least a description or date
        if slots.get('description') or slots.get('date'):
            # Add original segment as raw_text
            slots['raw_text'] = segment
            events.append(slots)
    
    # If no events extracted, return single event from full text
    if not events:
        return [extract_slots(text, nlp_model)]
    
    return events


def extract_slots(text: str, nlp_model=None) -> Dict[str, any]:
    """
    Extract calendar event slots using HYBRID approach:
    1. Rule-based extraction for DATE and TIME (always works)
    2. NER for ACTIVITY, PERSON, LOCATION (if model is trained)
    
    This ensures basic functionality even without a trained model!
    """
    # Normalize text first
    normalized_text = normalize_thai_text(text)
    
    slots = {
        'date': None,
        'time': None,
        'description': None,
        'attendees': [],
        'location': None,
        'raw_text': text
    }
    
    # STEP 1: Rule-based DATE extraction (works without model!)
    # Try to parse date from the original text
    date_parsed = parse_thai_date(normalized_text)
    if date_parsed:
        slots['date'] = date_parsed
    
    # STEP 2: Rule-based TIME extraction (works without model!)
    # Look for time patterns in text
    time_parsed = parse_thai_time(normalized_text)
    if time_parsed:
        slots['time'] = time_parsed
    
    # STEP 3: Try NER for ACTIVITY, PERSON, LOCATION (if model available)
    try:
        entities = extract_entities_with_pos(normalized_text, nlp_model)
        
        for ent_text, label, pos in entities:
            # Only use NER for activity, person, location
            # (Date/time already handled by rules)
            if label in ['ACTIVITY', 'EVENT'] and pos in ['VERB', 'NOUN', 'PROPN', 'UNKNOWN']:
                if not slots['description']:
                    slots['description'] = ent_text
                else:
                    slots['description'] += f", {ent_text}"
            
            elif label == 'PERSON' and pos in ['PROPN', 'NOUN', 'UNKNOWN']:
                slots['attendees'].append(ent_text)
            
            elif label == 'LOCATION' and pos in ['PROPN', 'NOUN', 'UNKNOWN']:
                slots['location'] = ent_text
    except Exception as e:
        # NER failed (model not trained?) - that's OK, we have dates/times from rules
        print(f"NER extraction failed (this is OK if model isn't trained): {e}")
    
    # STEP 4: Fallback - pattern-based extraction for person and location
    if not slots['description']:
        # List of common activity keywords
        activity_keywords = [
            # Meetings & work
            'ประชุม', 'meeting', 'นัด', 'เจอ', 'พบ',
            'เรียน', 'สอบ', 'นำเสนอ', 'presentation',
            'สัมมนา', 'workshop', 'ส่งงาน', 'รายงาน',
            
            # Food & dining
            'กินข้าว', 'กินอาหาร', 'ทานข้าว', 'ทานอาหาร',
            'อาหาร', 'มื้อ', 'เลี้ยง', 'ดินเนอร์',
            
            # Social activities
            'เที่ยว', 'ไปเที่ยว', 'ไปเดิน', 'ช้อปปิ้ง', 'ดูหนัง',
            'ดูคอนเสิร์ต', 'งานปาร์ตี้', 'ปาร์ตี้',
            
            # Health & wellness
            'หมอ', 'คลินิก', 'รักษา', 'ตรวจ', 'โรงพยาบาล',
            
            # Sports & fitness
            'ออกกำลังกาย', 'ฟิตเนส', 'วิ่ง', 'ว่ายน้ำ', 'โยคะ',
        ]
        
        for keyword in activity_keywords:
            if keyword in normalized_text:
                slots['description'] = keyword
                break
    
    # STEP 5: Pattern-based PERSON detection
    if not slots['attendees']:
        found_names = []
        
        # Expanded Thai titles and roles
        titles = [
            'รศ\\.ดร\\.', 'รศ\\.', 'ผศ\\.ดร\\.', 'ผศ\\.', 'ดร\\.', 'พญ\\.', 'นพ\\.',
            'อาจารย์', 'คุณ', 'นาย', 'นางสาว', 'นาง', 'น\\.ส\\.', 
            'ท่าน', 'พี่', 'เพื่อน'
        ]
        
        roles = [
            'ผอ\\.', 'ผู้อำนวยการ', 'ประธาน', 'เลขานุการ', 'นศ\\.', 'นักศึกษา'
        ]
        
        person_patterns = [
            # Full names: FirstName LastName (both must be Thai, 2+ chars each)
            r'([ก-ฮ]{2,15})\s+([ก-ฮ]{2,20})(?=\s|$|ที่|ตอน|เวลา)',
            
            # Title + Full Name (e.g., "รศ.ดร. ศิรวิชญ์")
            rf'(?:{"|".join(titles)})\s+([ก-ฮ][ก-ฮะ-ูเ-ไ์่้๊๋ํ]{{2,25}})(?:\s+([ก-ฮ]{{2,20}}))?(?=\s|$|ที่|ตอน)',
            
            # Role + name (e.g., "ผอ. สมชัย")
            rf'(?:{"|".join(roles)})\s+([ก-ฮ][ก-ฮะ-ูเ-ไ์่้๊๋ํ]{{2,20}})(?=\s|$|ที่)',
            
            # "กับ" + name/nickname
            r'กับ\s+([ก-ฮ][ก-ฮะ-ูเ-ไ์่้๊๋ํ]{1,20})(?=\s|$|ที่|และ)',
            
            # Verbs + person (พบ, เจอ, นัด, etc.)
            r'(?:พบ|เจอ|นัด|หา|ติดต่อ)\s+([ก-ฮ][ก-ฮะ-ูเ-ไ์่้๊๋ํ]{1,20})(?=\s|$|ที่)',
            
            # Group/department descriptors (e.g., "อาจารย์สาขาวิชาวิทยาการคอมฯ")
            r'(อาจารย์(?:สาขา)?(?:วิชา)?[ก-ฮะ-ูเ-ไ์่้๊๋ํฯ\s]{3,40})(?=\s|$|ที่|ตอน|เวลา|วัน)',
            r'(นักศึกษา[ก-ฮะ-ูเ-ไ์่้๊๋ํ\s]{0,20})(?=\s|$|ที่)',
        ]
        
        for pattern in person_patterns:
            matches = re.findall(pattern, normalized_text)
            for match in matches:
                if isinstance(match, tuple):
                    # Handle captured groups (e.g., first name + last name)
                    name_parts = [m.strip() for m in match if m and m.strip()]
                    if name_parts:
                        found_names.extend(name_parts)
                else:
                    found_names.append(match.strip())
        
        # Exclusion filters
        excluded = {
            'วัน', 'เวลา', 'ที่', 'ตอน', 'เดือน', 'ปี', 'ประชุม',
            'ม.ค.', 'ก.พ.', 'มี.ค.', 'เม.ย.', 'พ.ค.', 'มิ.ย.',
            'ก.ค.', 'ส.ค.', 'ก.ย.', 'ต.ค.', 'พ.ย.', 'ธ.ค.',
        }
        
        if found_names:
            # Filter and validate names
            names = [
                m.strip() for m in found_names
                if m.strip() not in excluded
                and len(m.strip()) >= 2  # Min 2 chars
                and len(m.strip()) <= 40  # Max 40 chars (for group names)
                and not m.strip()[0] in ['์', 'ิ', 'ี', 'ึ', 'ื', 'ุ', 'ู', '่', '้', '๊', '๋']
                and '.' not in m  # Exclude abbreviations with dots
            ]
            # Remove duplicates while preserving order
            seen = set()
            unique_names = []
            for name in names:
                if name not in seen and name:  # Also check not empty
                    seen.add(name)
                    unique_names.append(name)
            
            if unique_names:
                slots['attendees'] = ', '.join(unique_names[:2])  # Max 2 names
    
    # STEP 5.5: Pattern-based GENERIC PERSON detection (if no specific names found)
    if not slots['attendees']:
        generic_people = []
        
        # Pattern 1: "กับ" + generic person term
        generic_person_pattern = r'กับ\s*(เพื่อน|แฟน|พี่|น้อง|พ่อ|แม่|ลูก|สามี|ภรรยา|เจ้านาย|หัวหน้า|ทีม|เพื่อนร่วมงาน|คนรัก|แฟนสาว|แฟนหนุ่ม)'
        matches = re.findall(generic_person_pattern, normalized_text)
        generic_people.extend(matches)
        
        # Pattern 2: generic person + action verbs (พบ, เจอ, etc.)
        person_action_pattern = r'(เพื่อน|แฟน|พี่|น้อง)\s*(?:ไป|มา|พบ|เจอ|นัด)'
        matches = re.findall(person_action_pattern, normalized_text)
        generic_people.extend(matches)
        
        if generic_people:
            # Remove duplicates while preserving order
            unique_people = list(dict.fromkeys(generic_people))
            slots['attendees'] = ', '.join(unique_people[:2])  # Max 2
    
    # STEP 6: Pattern-based LOCATION detection  
    if not slots['location']:
        # Common location keywords - match more conservatively
        location_keywords = [
            'ตึก', 'อาคาร', 'ห้อง', 'ชั้น', 'ลาน',
            'โรงพยาบาล', 'โรงเรียน', 'มหาวิทยาลัย',
            'ศูนย์', 'คณะ', 'สำนักงาน'
        ]
        
        for keyword in location_keywords:
            pattern = keyword + r'\s*([ก-ฮา-ูเ-ไ0-9\s]{0,20})(?:\s|ที่|ตอน|เวลา|$)'
            match = re.search(pattern, normalized_text)
            if match:
                # Preserve spacing between keyword and content
                content = match.group(1).strip()
                if content:
                    # Add space between keyword and number if missing
                    if content and content[0].isdigit():
                        location = keyword + ' ' + content
                    else:
                        location = keyword + content
                else:
                    location = keyword
                
                # Validate: should be 3-30 chars and not just the keyword
                if 3 <= len(location) <= 30 and location != keyword:
                    slots['location'] = location[:30]
                    break
        
        # Specific location patterns
        if not slots['location']:
            location_patterns = [
                (r'ที่\s*([ก-ฮ][ก-ฮา-ูเ-ไ\s]{2,25})(?:ตอน|เวลา|ชั้น|$)', 1),  # "ที่" + location
                (r'(zoom|google\s*meet|teams|online)', 0),  # Online platforms
            ]
            
            for pattern, group_idx in location_patterns:
                match = re.search(pattern, normalized_text, re.IGNORECASE)
                if match:
                    location_text = match.group(group_idx) if group_idx > 0 else match.group()
                    if any(word in location_text.lower() for word in ['zoom', 'meet', 'teams', 'online']):
                        slots['location'] = 'ออนไลน์'
                    else:
                        slots['location'] = location_text.strip()[:30]
                    break
    
    # Convert attendees list to string (only if it's a list)
    if slots['attendees']:
        if isinstance(slots['attendees'], list):
            slots['attendees'] = ', '.join(slots['attendees'])
        # If it's already a string (from generic person detection), leave it as is
    else:
        slots['attendees'] = None
    
    return slots


def create_event(slots: Dict[str, any], event_id: Optional[str] = None) -> Dict:
    """Create a structured event from slots"""
    if event_id is None:
        event_id = f"evt_{uuid.uuid4().hex[:8]}"
    
    event = {
        'id': event_id,
        'date': slots.get('date'),
        'time': slots.get('time'),
        'description': slots.get('description'),
        'attendees': slots.get('attendees'),
        'location': slots.get('location'),
        'raw_text': slots.get('raw_text', ''),
        'created_at': get_current_datetime().isoformat()
    }
    
    return event


def load_events(filepath: str = EVENTS_FILE) -> List[Dict]:
    """Load events from JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('events', [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_events(events: List[Dict], filepath: str = EVENTS_FILE):
    """Save events to JSON file"""
    print(f"DEBUG: Saving {len(events)} events to {filepath}")  # Debug
    try:
        data = {'events': events}
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"DEBUG: Successfully saved to {filepath}")  # Debug
    except Exception as e:
        print(f"DEBUG: Error saving events: {e}")  # Debug
        raise


def add_event(event: Dict, filepath: str = EVENTS_FILE):
    """Add a new event to the file"""
    events = load_events(filepath)
    events.append(event)
    save_events(events, filepath)
    return event


def delete_event(event_id: str, filepath: str = EVENTS_FILE):
    """Delete an event by ID"""
    events = load_events(filepath)
    events = [e for e in events if e['id'] != event_id]
    save_events(events, filepath)


def update_event(event_id: str, updated_data: Dict, filepath: str = EVENTS_FILE):
    """
    Update an existing event by ID
    
    Args:
        event_id: ID of event to update
        updated_data: Dictionary with updated fields
        filepath: Path to events file
    """
    events = load_events(filepath)
    for i, event in enumerate(events):
        if event['id'] == event_id:
            # Preserve original metadata
            original_created_at = event.get('created_at')
            original_raw_text = event.get('raw_text')
            
            # Update fields
            events[i].update(updated_data)
            
            # Ensure critical fields are preserved
            events[i]['id'] = event_id
            if original_created_at:
                events[i]['created_at'] = original_created_at
            if original_raw_text:
                events[i]['raw_text'] = original_raw_text
            events[i]['updated_at'] = get_current_datetime().isoformat()
            
            save_events(events, filepath)
            return events[i]
    return None


def process_text_to_event(text: str, nlp_model=None, save_to_file: bool = False) -> Dict:
    """
    Complete pipeline: text → slots → validation → event
    
    Returns event dict with additional validation metadata:
    - 'is_valid': bool
    - 'missing_fields': list of critical missing fields
    - 'auto_filled': dict of fields that were auto-filled
    """
    from validation import validate_event_data, apply_safe_defaults
    
    # Extract slots
    slots = extract_slots(text, nlp_model)
    
    # Validate and get safe defaults
    is_valid, missing_fields, safe_defaults = validate_event_data(slots)
    
    # Apply safe defaults
    slots_with_defaults = apply_safe_defaults(slots, safe_defaults)
    
    # Create event (without saving yet)
    event = create_event(slots_with_defaults)
    
    # Add validation metadata
    event['is_valid'] = is_valid
    event['missing_fields'] = missing_fields
    event['auto_filled'] = safe_defaults
    
    # Only save if explicitly requested AND validation passes
    if save_to_file and is_valid:
        add_event(event)
    
    return event
