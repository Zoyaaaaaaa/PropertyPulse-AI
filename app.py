import streamlit as st
import os
from PIL import Image
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool, Tool
from langchain.agents import AgentType, initialize_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from dotenv import load_dotenv

load_dotenv()
# Setup page configuration
st.set_page_config(
    page_title="Real Estate Assistant",
    page_icon="🏠",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_agent" not in st.session_state:
    st.session_state.active_agent = None
if "issue_detection_memory" not in st.session_state:
    st.session_state.issue_detection_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
if "tenancy_faq_memory" not in st.session_state:
    st.session_state.tenancy_faq_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

# API key setup
api_key=os.environ.get('GOOGLE_API_KEY')
print(api_key)
genai.configure(api_key=api_key)

# Initialize DuckDuckGo search wrapper
ddg_search = DuckDuckGoSearchAPIWrapper()

# Function to process images for Gemini
def process_image(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data
            }
        ]
        return image_parts
    return None

# Define tools for the agents
@tool
def search_tenancy_laws(location: str) -> str:
    """Search for tenancy laws based on location."""
    # This would ideally connect to a database or API
    # For demo purposes, returning static information
    laws = {
        "us": "In the US, tenancy laws vary by state. Generally, landlords must provide written notice before eviction.",
        "uk": "In the UK, landlords typically need to provide 2 months notice with a Section 21 notice.",
        "canada": "In Canada, tenancy laws are provincial. In Ontario, landlords need proper grounds and notice for eviction.",
        "australia": "In Australia, notice periods vary by state/territory, typically 14-30 days for breaches."
    }
    
    location = location.lower()
    for key in laws:
        if key in location:
            return laws[key]
    
    return "I don't have specific information for that location. Generally, landlords must provide written notice before eviction, but regulations vary widely by jurisdiction."

@tool
def identify_property_issue(description: str) -> str:
    """Identify common property issues based on description."""
    # Simple keyword matching for demo purposes
    issues = {
        "mold": "Mold issues often require professional remediation and addressing the moisture source.",
        "leak": "Water leaks should be addressed immediately to prevent structural damage.",
        "crack": "Cracks may indicate structural issues and should be evaluated by a professional.",
        "paint": "Peeling paint may indicate moisture problems or poor adhesion.",
        "electric": "Electrical problems should always be handled by a licensed electrician."
    }
    
    description = description.lower()
    for key in issues:
        if key in description:
            return issues[key]
    
    return "Based on the description, I can't identify a specific common issue. Consider consulting with a property inspector for an in-person assessment."

@tool
def web_search(query: str) -> str:
    """Search the web for tenancy information using DuckDuckGo."""
    try:
        search_results = ddg_search.run(f"tenancy laws rental agreement {query}")
        return search_results
    except Exception as e:
        return f"Search error: {str(e)}. Please try a more specific query or phrase your question differently."

# Setup LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)

# Create a classifier LLM for determining which agent to use
classifier_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)

# Define system prompts for each agent
issue_detection_prompt = """You are a Property Issue Detection & Troubleshooting Agent.

Your responsibilities:
- Analyze descriptions of properties to identify issues
- Detect problems like water damage, mold, cracks, poor lighting, broken fixtures
- Provide practical troubleshooting suggestions
- Ask clarifying questions when needed

Always be helpful, practical, and focus on identifying property issues from descriptions.
If image analysis is provided, use that information to inform your response.
"""

tenancy_faq_prompt = """You are a Tenancy FAQ Agent.

Your responsibilities:
- Answer questions about tenancy laws, agreements, landlord/tenant responsibilities
- Provide location-specific guidance when possible
- Help with common tenancy issues like notice periods, rent increases, deposits, etc.
- Request location information when needed for more specific advice
- Use web search when needed to provide accurate and up-to-date information

Always be helpful and factual about tenancy regulations and best practices.
"""

# Create the tools for each agent
issue_detection_tools = [search_tenancy_laws, identify_property_issue]
tenancy_faq_tools = [
    search_tenancy_laws,
    web_search
]

# Create agent executors with LangChain memory
def get_issue_detection_agent():
    return initialize_agent(
        issue_detection_tools,
        llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        memory=st.session_state.issue_detection_memory,
        system_message=issue_detection_prompt
    )

def get_tenancy_faq_agent():
    return initialize_agent(
        tenancy_faq_tools,
        llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        memory=st.session_state.tenancy_faq_memory,
        system_message=tenancy_faq_prompt
    )

# Function to determine which agent to use
def classify_query(query):
    classification_prompt = f"""
    Determine if this user query is about:
    1) Property issues, damage, maintenance, or troubleshooting (respond with "ISSUE_DETECTION")
    OR
    2) Tenancy laws, rental agreements, tenant rights, or landlord responsibilities (respond with "TENANCY_FAQ")
    
    Respond with ONLY "ISSUE_DETECTION" or "TENANCY_FAQ" based on the most appropriate category.
    
    Query: {query}
    """
    
    response = classifier_llm.invoke(classification_prompt)
    response_text = response.content.strip()
    
    if "ISSUE_DETECTION" in response_text:
        return "issue_detection"
    elif "TENANCY_FAQ" in response_text:
        return "tenancy_faq"
    else:
        # Default to tenancy FAQ if classification is unclear
        return "tenancy_faq"

# Function to process Gemini image analysis
def analyze_image_with_gemini(image_data, prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    image_analysis_prompt = """
    You are a property issue detection expert. Analyze this image and identify any visible problems 
    with the property such as:
    - Water damage or leaks
    - Mold or mildew
    - Cracks in walls, floors, ceilings
    - Electrical issues
    - Structural damage
    - Poor lighting or ventilation
    - Broken fixtures or appliances
    
    Describe what you see, identify the specific issues, and suggest possible causes and solutions.
    """
    
    try:
        response = model.generate_content([image_analysis_prompt, image_data[0], prompt])
        return response.text
    except Exception as e:
        return f"Error analyzing image: {str(e)}. Please try again with a different image."

# Streamlit UI
st.title("🏠 Intelligent Real Estate Assistant")

# Agent selection (manual override)
col1, col2 = st.columns(2)
with col1:
    if st.button("Issue Detection & Troubleshooting 🔍", 
                use_container_width=True, 
                type="primary" if st.session_state.active_agent == "issue_detection" else "secondary"):
        st.session_state.active_agent = "issue_detection"
        
with col2:
    if st.button("Tenancy FAQ Agent 📝", 
                use_container_width=True,
                type="primary" if st.session_state.active_agent == "tenancy_faq" else "secondary"):
        st.session_state.active_agent = "tenancy_faq"

# Display current agent if manually selected
if st.session_state.active_agent == "issue_detection":
    st.info("💬 You're now talking to the **Issue Detection & Troubleshooting Agent**. You can upload images of property issues and ask questions about them.")
    
    # Add image upload option for issue detection agent
    uploaded_file = st.file_uploader("Upload an image of the property issue (optional)", type=["jpg", "jpeg", "png"], key="main_uploader")
    
    if uploaded_file is not None:
        # Save to session state so it persists between reruns
        st.session_state.uploaded_image = uploaded_file
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=400)
        
elif st.session_state.active_agent == "tenancy_faq":
    st.info("💬 You're now talking to the **Tenancy FAQ Agent**. Ask questions about rental agreements, tenant rights, and landlord responsibilities.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask your question here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Auto-determine which agent to use if not already set
    if st.session_state.active_agent is None:
        st.session_state.active_agent = classify_query(prompt)
        if st.session_state.active_agent == "issue_detection":
            st.info("💬 Based on your question, I'm connecting you with the **Issue Detection & Troubleshooting Agent**.")
        else:
            st.info("💬 Based on your question, I'm connecting you with the **Tenancy FAQ Agent**.")
    
    # Process with appropriate agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if st.session_state.active_agent == "issue_detection":
                    # Get the agent instance
                    issue_agent = get_issue_detection_agent()
                    
                    # Process with image if available
                    if st.session_state.uploaded_image is not None:
                        image_data = process_image(st.session_state.uploaded_image)
                        image_analysis = analyze_image_with_gemini(image_data, prompt)
                        
                        # Combine image analysis with user query for the agent
                        combined_prompt = f"User query: {prompt}\n\nImage analysis: {image_analysis}"
                        
                        response = issue_agent.run(input=combined_prompt)
                    else:
                        response = issue_agent.run(input=prompt)
                else:  # tenancy_faq agent
                    tenancy_agent = get_tenancy_faq_agent()
                    response = tenancy_agent.run(input=prompt)
                
                st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_message = f"I encountered an error: {str(e)}. Let me try to respond more generally."
                st.error(error_message)
                
                # Fallback to direct LLM response without tools
                fallback_prompt = f"""
                As a real estate assistant, please respond to this user query without using external tools:
                {prompt}
                
                Provide a helpful response based on general knowledge about {st.session_state.active_agent}.
                """
                
                fallback_response = llm.invoke(fallback_prompt).content
                st.markdown(fallback_response)
                
                # Add fallback response to chat history
                st.session_state.messages.append({"role": "assistant", "content": fallback_response})
            
            # Re-classify for the next query
            st.session_state.active_agent = None

# Add disclaimer at the bottom
st.sidebar.title("About")
# Remove the duplicate uploader in the sidebar
st.sidebar.info(
    """
    This Intelligent Real Estate Assistant consists of two specialized agents:
    
    **Issue Detection Agent** - Analyzes property images and descriptions to identify problems and suggest solutions.
    
    **Tenancy FAQ Agent** - Provides information on tenancy laws, rental agreements, and landlord/tenant rights using real-time web search capabilities.
    
    *The assistant will automatically select the appropriate agent based on your question, but you can also manually switch between them using the buttons above.*
    
    *Note: This is a demonstration tool. For legal or property matters, always consult with qualified professionals.*
    """
)