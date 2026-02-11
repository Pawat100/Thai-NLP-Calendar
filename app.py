"""
Thai NLP Chatbot with Calendar Integration
Streamlit App for Event Extraction
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import calendar as cal
import uuid
from nlp_utils import (
    load_ner_model,
    normalize_thai_text,
    extract_slots,
    extract_multiple_events,
    create_event,
    load_events,
    save_events,
    add_event,
    delete_event,
    update_event,
    get_current_datetime,
    process_text_to_event
)

# Generate unique session ID for this user
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4().hex[:12])

# Session-specific events file
SESSION_EVENTS_FILE = f"events_{st.session_state.session_id}.json"

# Page configuration
st.set_page_config(
    page_title="Thai NLP Calendar Chatbot",
    page_icon="📅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state FIRST
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'nlp_model' not in st.session_state:
    with st.spinner('Loading NLP model...'):
        st.session_state.nlp_model = load_ner_model()

if 'current_month' not in st.session_state:
    now = get_current_datetime()
    st.session_state.current_month = now.month
    st.session_state.current_year = now.year

if 'enlarged_view' not in st.session_state:
    st.session_state.enlarged_view = False

if 'selected_event' not in st.session_state:
    st.session_state.selected_event = None

if 'editing_event_id' not in st.session_state:
    st.session_state.editing_event_id = None

if 'editing_in_modal' not in st.session_state:
    st.session_state.editing_in_modal = None

# Custom CSS - with larger sidebar and modern calendar design
st.markdown("""
<style>
    /* Larger sidebar text */
    [data-testid="stSidebar"] {
        font-size: 1.1rem;
    }
    [data-testid="stSidebar"] .stButton button {
        font-size: 1.1rem;
    }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        font-size: 1.4rem;
    }
    
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .event-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #5a67d8;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .event-card h4 {
        color: white !important;
        margin-bottom: 1rem;
    }
    .event-card p strong {
        color: #ffd700;
    }
    
    /* Modern Calendar Styling */
    .calendar-container {
        background: white;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .calendar-day {
        border: 1px solid #e2e8f0;
        padding: 0.75rem;
        min-height: 120px;
        background: white;
        border-radius: 0.5rem;
        margin: 2px;
        transition: all 0.2s;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .calendar-day:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    .calendar-day-header {
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.75rem;
        font-size: 1.1rem;
        border-radius: 0.5rem;
        margin: 2px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .day-number {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2d3748;
        margin-bottom: 0.5rem;
    }
    .today-badge {
        background: #fbbf24;
        color: #78350f;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 700;
        margin-left: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("📅 Thai Calendar Bot")
    st.markdown("---")
    
    # Calendar navigation
    st.subheader("Calendar Navigation")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("◀"):
            st.session_state.current_month -= 1
            if st.session_state.current_month < 1:
                st.session_state.current_month = 12
                st.session_state.current_year -= 1
            st.rerun()
    
    with col2:
        month_year = f"{cal.month_name[st.session_state.current_month]} {st.session_state.current_year}"
        st.markdown(f"**{month_year}**")
    
    with col3:
        if st.button("▶"):
            st.session_state.current_month += 1
            if st.session_state.current_month > 12:
                st.session_state.current_month = 1
                st.session_state.current_year += 1
            st.rerun()
    
    # Today button
    if st.button("📍 Today"):
        now = get_current_datetime()
        st.session_state.current_month = now.month
        st.session_state.current_year = now.year
        st.rerun()
    
    st.markdown("---")
    
    # All events list - Each event collapsible
    st.subheader("📋 All Events")
    events = load_events(SESSION_EVENTS_FILE)
    
    if events:
        total_text = f"Total: {len(events)} events"
        st.markdown(f"**{total_text}**")
        st.markdown("---")
        
        
        for idx, event in enumerate(sorted(events, key=lambda x: x.get('date') or '9999-99-99')):
            desc = event.get('description') or 'No description'
            event_date = event.get('date', 'No date')
            event_time = event.get('time', '')
            
            # Each event in its own expander
            with st.expander(f"📅 {event_date} - {desc[:25]}...", expanded=False):
                # Show details inside the expander
                st.markdown(f"**Time:** {event_time}")
                st.markdown(f"**Event:** {desc}")
                if event.get('attendees') and event.get('attendees') != '-':
                    st.markdown(f"**Attendees:** {event.get('attendees')}")
                if event.get('location') and event.get('location') != '-':
                    st.markdown(f"**Location:** {event.get('location')}")
                
            st.markdown("---")
            
            # Buttons to view full details or delete
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔍 View", key=f"view_{idx}_{event['id']}", use_container_width=True):
                    st.session_state.selected_event = event
                    st.session_state.editing_in_modal = None  # Reset edit mode
                    st.rerun()
            with col2:
                if st.button("🗑️ Delete", key=f"del_{idx}_{event['id']}", use_container_width=True, type="secondary"):
                    delete_event(event['id'], SESSION_EVENTS_FILE)
                    st.success("Event deleted!")
                    st.rerun()
    else:
        st.info("No events yet. Start chatting to add events!")
    
    st.markdown("---")
    
    # Export options
    st.subheader("Export")
    if st.button("📥 Export as JSON"):
        st.download_button(
            label="Download events.json",
            data=open(SESSION_EVENTS_FILE, 'r', encoding='utf-8').read(),
            file_name='events.json',
            mime='application/json'
        )
    
    # Clear data
    if st.button("🗑️ Clear All Data"):
        if st.checkbox("Confirm delete all"):
            save_events([], SESSION_EVENTS_FILE)
            st.session_state.messages = []
            st.success("All data cleared!")
            st.rerun()

# Main content
st.title("💬 Thai NLP Calendar Chatbot")
st.caption("Chat in Thai to create calendar events automatically!")

# Tab layout
tab1, tab2 = st.tabs(["💬 Chat", "📅 Calendar"])

with tab1:
    # Chat interface
    st.subheader("Chat")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "event" in message:
                event = message["event"]
                
                # Always show all fields
                st.markdown(f"""
                <div class="event-card">
                    <h4>📅 Event Created</h4>
                    <p><strong>Date:</strong> {event.get('date', 'N/A')}</p>
                    <p><strong>Time:</strong> {event.get('time', 'N/A')}</p>
                    <p><strong>Event:</strong> {event.get('description', 'N/A')}</p>
                    <p><strong>Attendees:</strong> {event.get('attendees', '-')}</p>
                    <p><strong>Where:</strong> {event.get('location', '-')}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("พิมพ์ข้อความภาษาไทย... (เช่น พรุ่งนี้มีประชุมกับบีมตอน 10 โมง)"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process with NLP (DO NOT SAVE YET)
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                try:
                    # Extract MULTIPLE events using new separator logic
                    events = extract_multiple_events(
                        prompt,
                        nlp_model=st.session_state.nlp_model
                    )
                    
                    # Process each event - validate and create
                    from validation import validate_event_data, apply_safe_defaults
                    valid_events = []
                    for slots in events:
                        # Validate and get safe defaults
                        is_valid, missing_fields, safe_defaults = validate_event_data(slots)
                        
                        # Apply safe defaults
                        slots_with_defaults = apply_safe_defaults(slots, safe_defaults)
                        
                        # Create event (without saving)
                        event = create_event(slots_with_defaults)
                        
                        # Add validation metadata
                        event['is_valid'] = is_valid
                        event['missing_fields'] = missing_fields
                        event['auto_filled'] = safe_defaults
                        
                        valid_events.append(event)
                    
                    # Check if we have multiple events
                    if len(valid_events) > 1:
                        st.success(f"✨ พบ {len(valid_events)} กิจกรรม!")
                    
                    # Display each event
                    for idx, event in enumerate(valid_events, 1):
                        # Check validation status
                        if not event.get('is_valid'):
                            # CRITICAL DATA MISSING - Ask user
                            from validation import format_missing_fields_message
                            missing_msg = format_missing_fields_message(event.get('missing_fields', []))
                            
                            st.warning(f"⚠️ กิจกรรมที่ {idx}: ข้อมูลไม่ครบ")
                            st.markdown(missing_msg)
                            
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": f"ข้อมูลไม่ครบ: {missing_msg}"
                            })
                        
                        else:
                            # DATA IS VALID - Show confirmation UI
                            if len(valid_events) > 1:
                                response = f"✅ กิจกรรมที่ {idx}:"
                            else:
                                response = "✅ ฉันพบข้อมูลนี้จากข้อความของคุณ:"
                            st.markdown(response)
                        
                            # Show auto-filled info if any
                            if event.get('auto_filled'):
                                auto_fill_msg = ", ".join([f"{k}: {v}" for k, v in event['auto_filled'].items()])
                                st.info(f"ℹ️ ตั้งค่าอัตโนมัติ: {auto_fill_msg}")
                            
                            # Display extracted event data
                            
                            # Build conditional fields
                            attendees_html = f"<p><strong>ผู้เข้าร่วม:</strong> {event['attendees']}</p>" if event.get('attendees') and event['attendees'] != '-' else ""
                            location_html = f"<p><strong>สถานที่:</strong> {event['location']}</p>" if event.get('location') and event['location'] != '-' else ""
                            
                            st.markdown(f"""
                            <div class="event-card">
                                <h4>📅 ข้อมูลที่ตรวจพบ</h4>
                                <p><strong>วันที่:</strong> {event.get('date', 'N/A')}</p>
                                <p><strong>เวลา:</strong> {event.get('time', 'N/A')}</p>
                                <p><strong>กิจกรรม:</strong> {event.get('description', 'N/A')}</p>
                                {attendees_html}
                                {location_html}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            #Store pending event(s) in session state
                            # Reset pending_events for new input (prevents duplicates)
                            st.session_state.pending_events = []
                            st.session_state.pending_events.append(event)
                            
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response,
                                "event": event,
                                "needs_confirmation": True
                            })
                    
                except Exception as e:
                    error_msg = f"เกิดข้อผิดพลาด: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
        
        st.rerun()
    
    # Show confirmation buttons if there are pending events
    pending_events = st.session_state.get('pending_events', [])
    if pending_events:
        st.markdown("---")
        if len(pending_events) > 1:
            st.subheader(f"🔍 ยืนยันการบันทึก ({len(pending_events)} กิจกรรม)")
        else:
            st.subheader("🔍 ยืนยันการบันทึก")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("✅ บันทึกทั้งหมด", key="confirm_save_all_btn", use_container_width=True, type="primary"):
                try:
                    from validation import is_event_saveable
                    from nlp_utils import add_event
                    
                    saved_count = 0
                    failed_events = []
                    
                    # Save all valid events
                    for idx, pending in enumerate(pending_events, 1):
                        # CRITICAL: Clean FIRST, then validate (matches edit flow)
                        clean_event = {k: v for k, v in pending.items() 
                                     if k not in ['is_valid', 'missing_fields', 'auto_filled']}
                        
                        # Replace placeholder "-" with None for validation
                        for key in ['date', 'time', 'description', 'attendees', 'location']:
                            if clean_event.get(key) == '-':
                                clean_event[key] = None
                        
                        # Now validate the clean data
                        if is_event_saveable(clean_event):
                            add_event(clean_event, SESSION_EVENTS_FILE)
                            saved_count += 1
                        else:
                            # Track which fields are missing
                            missing = []
                            if not clean_event.get('date'):
                                missing.append('วันที่')
                            if not clean_event.get('time'):
                                missing.append('เวลา')
                            if not clean_event.get('description'):
                                missing.append('กิจกรรม')
                            failed_events.append((idx, missing))
                    
                    # Show detailed results
                    if saved_count == len(pending_events):
                        st.success(f"✅ บันทึกสำเร็จ {saved_count} กิจกรรม!")
                    elif saved_count > 0:
                        st.warning(f"⚠️ บันทึกสำเร็จ {saved_count}/{len(pending_events)} กิจกรรม")
                        for event_num, missing_fields in failed_events:
                            st.error(f"❌ กิจกรรมที่ {event_num}: ขาด {', '.join(missing_fields)}")
                    else:
                        st.error("❌ ไม่สามารถบันทึกได้ - ข้อมูลไม่ครบ")
                        for event_num, missing_fields in failed_events:
                            st.error(f"📌 กิจกรรมที่ {event_num}: ขาด {', '.join(missing_fields)}")
                    
                    # Clear all pending states
                    st.session_state.pending_events = []
                    st.session_state.pending_event = None
                    st.session_state.show_edit_form = False
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ เกิดข้อผิดพลาดในการบันทึก: {str(e)}")
        
        
        
        with col2:
            if st.button("✏️ แก้ไข", key="confirm_edit_btn", use_container_width=True):
                st.session_state.show_edit_form = True
                st.session_state.pending_event = pending_events[0]  # Edit first event
        
        with col3:
            if st.button("❌ ยกเลิก", key="confirm_cancel_btn", use_container_width=True):
                st.session_state.pending_events = []
                st.session_state.pending_event = None
                st.rerun()
        
        # Show edit form if requested
        if st.session_state.get('show_edit_form'):
            st.markdown("### ✏️ แก้ไขข้อมูล")
            
            with st.form("edit_event_form"):
                from datetime import datetime
                
                # Get the pending event from session state
                pending = st.session_state.get('pending_event', {})
                
                # Parse date for date_input
                try:
                    date_val = datetime.strptime(pending.get('date', ''), '%Y-%m-%d').date() if pending.get('date') else None
                except:
                    date_val = None
                
                # Parse time for time_input
                try:
                    time_val = datetime.strptime(pending.get('time', '09:00'), '%H:%M').time() if pending.get('time') else None
                except:
                    time_val = None
                
                new_date = st.date_input("📅 วันที่", value=date_val)
                new_time = st.time_input("🕐 เวลา", value=time_val)
                new_desc = st.text_input("📝 กิจกรรม", value=pending.get('description', ''))
                new_attendees = st.text_input("👥 ผู้เข้าร่วม", value=pending.get('attendees', ''))
                new_location = st.text_input("📍 สถานที่", value=pending.get('location', ''))
                
                if st.form_submit_button("💾 บันทึกการแก้ไข", use_container_width=True, type="primary"):
                    from validation import is_event_saveable
                    
                    # Update pending event with new values
                    updated_event = {
                        'id': pending['id'],
                        'date': new_date.strftime('%Y-%m-%d') if new_date else None,
                        'time': new_time.strftime('%H:%M') if new_time else None,
                        'description': new_desc,
                        'attendees': new_attendees if new_attendees else None,
                        'location': new_location if new_location else None,
                        'raw_text': pending.get('raw_text', ''),
                        'created_at': pending.get('created_at', '')
                    }
                    
                    # Final validation
                    if is_event_saveable(updated_event):
                        add_event(updated_event, SESSION_EVENTS_FILE)  # Save to session file!
                        st.success("✅ บันทึกการแก้ไขสำเร็จ!")
                        # Clear all pending states
                        st.session_state.pending_events = []
                        st.session_state.pending_event = None
                        st.session_state.show_edit_form = False
                        st.rerun()
                    else:
                        st.error("❌ กรุณาระบุทั้งวันที่และกิจกรรม")

with tab2:
    # Calendar view
    st.subheader(f"📅 {cal.month_name[st.session_state.current_month]} {st.session_state.current_year}")
    
    # Load events for display
    events = load_events(SESSION_EVENTS_FILE)
    events_by_date = {}
    for event in events:
        if event.get('date'):
            if event['date'] not in events_by_date:
                events_by_date[event['date']] = []
            events_by_date[event['date']].append(event)
    
    # Create calendar
    month_cal = cal.monthcalendar(st.session_state.current_year, st.session_state.current_month)
    
    # Day headers
    cols = st.columns(7)
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for i, day_name in enumerate(day_names):
        with cols[i]:
            st.markdown(f'<div class="calendar-day-header">{day_name}</div>', unsafe_allow_html=True)
    
    # Calendar days
    for week in month_cal:
        cols = st.columns(7)
        for i, day in enumerate(week):
            with cols[i]:
                if day == 0:
                    # Empty day cell
                    st.markdown('<div class="calendar-day" style="background-color: #f7fafc; min-height: 120px;"></div>', unsafe_allow_html=True)
                else:
                    date_str = f"{st.session_state.current_year}-{st.session_state.current_month:02d}-{day:02d}"
                    
                    # Check if this is today
                    now = get_current_datetime()
                    is_today = (date_str == now.strftime('%Y-%m-%d'))
                    
                    # Get events for this day
                    day_events = events_by_date.get(date_str, [])
                    
                    # Build day cell HTML with modern styling
                    today_badge = '<span class="today-badge">TODAY</span>' if is_today else ''
                    bg_color = "#fef3c7" if is_today else "white"
                    
                    # Start container
                    st.markdown(
                        f'<div class="calendar-day" style="background-color: {bg_color};">'
                        f'<div class="day-number">{day}{today_badge}</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Limit events shown to prevent overlap (max 2)
                    MAX_EVENTS_SHOWN = 2
                    visible_events = day_events[:MAX_EVENTS_SHOWN]
                    hidden_count = len(day_events) - MAX_EVENTS_SHOWN
                    
                    # Create clickable buttons for visible events
                    for event_idx, event in enumerate(visible_events):
                        time_str = event.get('time', '')
                        desc = (event.get('description') or 'Event')[:15]
                        button_label = f"🔔 {time_str} {desc}"
                        
                        if st.button(button_label, key=f"cal_event_{event_idx}_{event['id']}_{date_str}", use_container_width=True, type="secondary"):
                            st.session_state.selected_event = event
                            st.rerun()
                    
                    # Show "+ X more" button if there are hidden events
                    if hidden_count > 0:
                        if st.button(f"+ {hidden_count} more", key=f"more_{date_str}", use_container_width=True, type="secondary"):
                            # Set first hidden event to show them all somehow
                            # For now, just show the first hidden one
                            st.session_state.selected_event = day_events[MAX_EVENTS_SHOWN]
                            st.rerun()
                    
                    # Close div
                    st.markdown('</div>', unsafe_allow_html=True)

# Event Detail Modal - Using st.dialog for floating popup
@st.dialog("📋 Event Details", width="large")
def show_event_details():
    event = st.session_state.selected_event
    
    # Check if we're in edit mode for this event
    if st.session_state.get('editing_in_modal') == event['id']:
        # EDIT MODE
        st.markdown("### ✏️ แก้ไขข้อมูล")
        
        with st.form(key=f"modal_edit_{event['id']}"):
            from datetime import datetime
            
            # Parse existing values
            try:
                date_val = datetime.strptime(event.get('date', ''), '%Y-%m-%d').date() if event.get('date') else None
            except:
                date_val = None
            
            try:
                time_val = datetime.strptime(event.get('time', '09:00'), '%H:%M').time() if event.get('time') else None
            except:
                time_val = None
            
            new_date = st.date_input("📅 วันที่", value=date_val)
            new_time = st.time_input("🕐 เวลา", value=time_val)
            new_desc = st.text_input("📝 กิจกรรม", value=event.get('description', ''))
            new_attendees = st.text_input("👥 ผู้เข้าร่วม", value=event.get('attendees', ''))
            new_location = st.text_input("📍 สถานที่", value=event.get('location', ''))
            
            col_save, col_cancel = st.columns(2)
            with col_save:
                if st.form_submit_button("💾 บันทึก", use_container_width=True, type="primary"):
                    updated_data = {
                        'date': new_date.strftime('%Y-%m-%d') if new_date else None,
                        'time': new_time.strftime('%H:%M') if new_time else None,
                        'description': new_desc,
                        'attendees': new_attendees if new_attendees else '-',
                        'location': new_location if new_location else '-'
                    }
                    update_event(event['id'], updated_data, SESSION_EVENTS_FILE)
                    st.session_state.editing_in_modal = None
                    st.session_state.selected_event = None
                    st.success("✅ บันทึกเรียบร้อย!")
                    st.rerun()
            with col_cancel:
                if st.form_submit_button("❌ ยกเลิก", use_container_width=True):
                    st.session_state.editing_in_modal = None
                    st.rerun()
    else:
        # VIEW MODE
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2.5rem; border-radius: 1rem; color: white; box-shadow: 0 20px 60px rgba(0,0,0,0.3);">
            <h1 style="color: white; margin: 0; font-size: 2.5rem;">📅 {event.get('description', 'Event')}</h1>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("")
        
        # Details in columns
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📆 Date")
            st.markdown(f"<p style='font-size: 1.5rem; font-weight: 600;'>{event.get('date', 'N/A')}</p>", unsafe_allow_html=True)
            
            st.markdown("### 👥 Attendees")
            st.markdown(f"<p style='font-size: 1.3rem;'>{event.get('attendees', '-')}</p>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 🕐 Time")
            st.markdown(f"<p style='font-size: 1.5rem; font-weight: 600;'>{event.get('time', 'N/A')}</p>", unsafe_allow_html=True)
            
            st.markdown("### 📍 Location")
            st.markdown(f"<p style='font-size: 1.3rem;'>{event.get('location', '-')}</p>", unsafe_allow_html=True)
        
        if event.get('raw_text'):
            st.markdown("---")
            st.markdown("### 📝 Original Input")
            st.markdown(f"<p style='font-size: 1.6rem; line-height: 1.8; padding: 1rem; background: #f8f9fa; border-radius: 0.5rem; color: #2d3748;'>{event.get('raw_text', '')}</p>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Edit button
        if st.button("✏️ แก้ไข", use_container_width=True, type="primary"):
            st.session_state.editing_in_modal = event['id']
            st.rerun()

# Show dialog if event is selected
if st.session_state.selected_event:
    show_event_details()

# Footer
st.markdown("---")
st.caption("💡 Tip: Try typing in Thai like 'พรุ่งนี้มีประชุมกับบีมตอน 10 โมง' or 'วันจันทร์นัดหมอ 3 โมง'")
