

# 🏠 Intelligent Real Estate Assistant
A powerful AI-driven assistant for helping users with **property maintenance** and **tenancy law** questions. It intelligently switches between two specialized agents for more accurate and context-aware support.

![Real Estate Assistant Banner](https://images.unsplash.com/photo-1560518883-ce09059eeffa?ixlib=rb-1.2.1&auto=format&fit=crop&w=950&q=80)






## **🔗 Live link:** [realestate-assistant.streamlit.app](https://propertypulse-ai-ekpxbxbsspygzkjedwpf6g.streamlit.app/)

---

## 🎥 Demo Video

📽️ [Watch the Demo]

https://github.com/user-attachments/assets/2b9f9c27-75b4-413d-9ea6-7b1090aeb4e5


---

## 🚀 Features

- 🔧 **Property Issue Expert**: Diagnoses common issues like leaks, mold, pests, etc. from images + text.
- 📜 **Tenancy Law Specialist**: Provides rental law guidance tailored to your location.
- 🤖 **AI Agent Switching**: Classifies queries and routes them to the most relevant agent.
- 📸 **Image Support**: Analyze photos of property issues with Gemini 1.5 Flash.
- 🌍 **Location-Aware Responses**: Delivers advice based on detected or selected regions.
- 💬 **Memory & Context**: Remembers past inputs for consistent and intelligent conversation.

---

## 🧠 How It Works

### 🧭 Smart Agent Routing

1. **Query Classification**  
   Automatically determines if a message relates to:
   - 🔧 Maintenance → routed to **Property Issue Expert**
   - 🏘️ Tenancy Law → routed to **Tenancy FAQ Agent**

2. **Agent Switching**  
   Dynamically switches agents mid-conversation based on query content.

3. **Manual Override**  
   Users can choose which expert to speak to via UI buttons.

---

### 🖼️ Image Issue Detection

Upload a photo of a property issue, and the assistant will:

- Detect visual signs of problems (e.g., mold, water damage, pests)
- Combine analysis with your query for full context
- Return:
  - Problem description
  - Probable cause
  - Recommended fix
  - Urgency level

---

### 📍 Location-Based Tenancy Law Help

- Automatically detects location from text or manual input
- Retrieves up-to-date regional laws using DuckDuckGo Search API
- Maintains location memory for contextually accurate responses

---

## 🧪 Use Cases

### 🔧 Property Issues
- **Mold on ceiling** → Identifies health risks + gives cleaning tips  
- **Hot power outlet** → Flags fire hazard + urges electrician contact  

### 🏘️ Tenancy Law
- **Security deposit in California** → Returns legal refund timeframes  
- **Rent increase in the UK** → Explains limits + required notice period  

---

## 🛠️ Tech Stack

| Category           | Tool / Framework |
|--------------------|------------------|
| UI & Hosting       | Streamlit        |
| AI Models          | Gemini 2.0 & 1.5 Flash |
| Orchestration      | LangChain        |
| Search Integration | DuckDuckGo API   |
| Memory             | LangChain Memory |
| Styling            | Custom CSS       |

---

## 🧑‍💻 Getting Started

### 🐳 Docker Setup (Recommended)

**Build & Run:**
```bash
docker build -t real-estate-assistant .
docker run -p 8501:8501 real-estate-assistant
```

---

### 🔧 Manual Setup

1. **Clone Repo:**
   ```bash
   git clone https://github.com/yourusername/real-estate-assistant.git
   cd real-estate-assistant
   ```

2. **Install Requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set API Key:**
   Create `.env`:
   ```
   GOOGLE_API_KEY = your_gemini_api_key
   ```

4. **Run App:**
   ```bash
   streamlit run app.py
   ```


![Screenshot 2025-04-27 005204](https://github.com/user-attachments/assets/7f2ca86f-4e69-4610-80fa-df82f20d7dfa)
---


> ⚠️ *Disclaimer: This tool is for informational use only. Always consult professionals for legal or urgent maintenance issues.*

