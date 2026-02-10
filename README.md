# Thai NLP Calendar Chatbot 🗓️

A Streamlit-based chatbot that extracts calendar events from Thai text using NLP.

## Features

✨ **Multi-Event Detection** - Detects multiple events in a single message
- Example: "ประชุมวันจันทร์ 10 โมง และส่งเอกสารพรุ่งนี้" → 2 events

🎯 **Smart Entity Extraction**
- Dates (วันนี้, พรุ่งนี้, จันทร์, etc.)
- Times (10 โมง, 14:00, บ่ายสอง)
- Activities (ประชุม, ส่งงาน, นัด)
- People (รศ.ดร. ศิรวิชญ์, บีม, อาจารย์สาขาวิชาวิทยาการคอมฯ)
- Locations (ห้อง 301, มอ, คณะวิทย์)

📅 **Interactive Calendar** - Visual monthly calendar with event markers

✏️ **Event Management** - Add, edit, and delete events with confirmation

## Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd COM
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
streamlit run app.py
```

## Usage

1. **Type a message** in Thai (or English):
   - "พรุ่งนี้มีประชุมกับบีมตอน 10 โมง"
   - "วันจันทร์นัดหมอ 3 โมง"
   - "มีนัดประชุมจันทร์ 10 โมง และส่งเอกสารพรุ่งนี้"

2. **Review** the extracted information

3. **Confirm or Edit** before saving

4. **View** events in the calendar

## File Structure

```
COM/
├── app.py              # Main Streamlit application
├── nlp_utils.py        # NLP extraction logic
├── validation.py       # Event validation
├── train_model.py      # spaCy NER training data
├── requirements.txt    # Python dependencies
├── packages.txt        # System packages for Streamlit Cloud
├── .gitignore         # Git ignore rules
├── events.json        # Event storage (auto-generated)
└── README.md          # This file
```

## Technologies

- **Streamlit** - Web interface
- **spaCy** - NER (Named Entity Recognition)
- **pythainlp** - Thai text normalization
- **dateparser** - Date/time parsing

## Deployment

### Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy from your repository
4. Done! ✅

## License

MIT License

## Author

Created for Thai NLP Calendar Event Extraction Project
