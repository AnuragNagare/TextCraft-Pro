import streamlit as st
import requests
import json
import time
import re
from datetime import datetime, timedelta
import hashlib
from textstat import flesch_reading_ease, flesch_kincaid_grade
import nltk
from collections import Counter, defaultdict
import io
import asyncio
import threading
from typing import Dict, List, Any
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass
import sqlite3
import pandas as pd

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Page config
st.set_page_config(
    page_title="ContentStudio AI Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for modern enterprise design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        font-family: 'Inter', sans-serif;
    }
    
    .enterprise-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .enterprise-subtitle {
        font-size: 1.2rem;
        font-weight: 300;
        opacity: 0.9;
    }
    
    .metric-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9ff 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    
    .streaming-content {
        background: linear-gradient(145deg, #f8f9fa 0%, #ffffff 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #e9ecef;
        margin: 1rem 0;
        min-height: 200px;
        font-family: 'Inter', sans-serif;
        line-height: 1.6;
        box-shadow: 0 5px 20px rgba(0,0,0,0.05);
    }
    
    .agent-status {
        display: flex;
        align-items: center;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .agent-active {
        background: linear-gradient(90deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        border-left: 4px solid #28a745;
    }
    
    .agent-completed {
        background: linear-gradient(90deg, #cce5ff 0%, #b3d9ff 100%);
        color: #004085;
        border-left: 4px solid #007bff;
    }
    
    .agent-waiting {
        background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%);
        color: #6c757d;
        border-left: 4px solid #6c757d;
    }
    
    .analytics-dashboard {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .success-message {
        padding: 1rem;
        background: linear-gradient(90deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        border: 1px solid #c3e6cb;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .error-message {
        padding: 1rem;
        background: linear-gradient(90deg, #f8d7da 0%, #f5c6cb 100%);
        color: #721c24;
        border: 1px solid #f5c6cb;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .quality-score {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        font-family: 'Inter', sans-serif;
    }
    
    .brand-upload {
        border: 3px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: linear-gradient(145deg, #f8f9ff 0%, #ffffff 100%);
        transition: all 0.3s ease;
    }
    
    .brand-upload:hover {
        border-color: #764ba2;
        background: linear-gradient(145deg, #f0f4ff 0%, #ffffff 100%);
    }
    
    .rate-limit-indicator {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 500;
    }
    
    .rate-limit-ok {
        background: linear-gradient(90deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
    }
    
    .rate-limit-warning {
        background: linear-gradient(90deg, #fff3cd 0%, #ffeaa7 100%);
        color: #856404;
    }
    
    .rate-limit-danger {
        background: linear-gradient(90deg, #f8d7da 0%, #f5c6cb 100%);
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Advanced Data Models
@dataclass
class AgentTask:
    name: str
    status: str  # 'waiting', 'active', 'completed', 'failed'
    result: str = ""
    start_time: datetime = None
    end_time: datetime = None

@dataclass
class GenerationMetrics:
    tokens_used: int
    response_time: float
    quality_score: float
    cost_estimate: float
    timestamp: datetime

# Initialize enhanced session state
def init_session_state():
    defaults = {
        'api_validated': False,
        'generated_content': "",
        'content_variations': [],
        'brand_voice_samples': [],
        'generation_history': [],
        'agent_tasks': [],
        'streaming_content': "",
        'generation_metrics': [],
        'rate_limit_status': {'requests': 0, 'reset_time': datetime.now() + timedelta(hours=1)},
        'user_analytics': defaultdict(int),
        'content_performance': [],
        'api_usage_stats': {'total_requests': 0, 'total_tokens': 0, 'total_cost': 0.0}
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Database Setup for Analytics
def init_database():
    """Initialize SQLite database for analytics"""
    conn = sqlite3.connect('content_analytics.db', check_same_thread=False)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS content_generations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME,
        content_type TEXT,
        topic TEXT,
        word_count INTEGER,
        quality_score REAL,
        tokens_used INTEGER,
        response_time REAL,
        user_session TEXT
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS api_usage (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME,
        endpoint TEXT,
        tokens_used INTEGER,
        cost REAL,
        user_session TEXT
    )
    ''')
    
    conn.commit()
    return conn

# Rate Limiting System
class RateLimiter:
    def __init__(self, max_requests=100, window_hours=1):
        self.max_requests = max_requests
        self.window_hours = window_hours
    
    def check_rate_limit(self) -> tuple[bool, str]:
        current_time = datetime.now()
        
        # Reset if window expired
        if current_time > st.session_state.rate_limit_status['reset_time']:
            st.session_state.rate_limit_status = {
                'requests': 0,
                'reset_time': current_time + timedelta(hours=self.window_hours)
            }
        
        requests_made = st.session_state.rate_limit_status['requests']
        
        if requests_made >= self.max_requests:
            time_left = st.session_state.rate_limit_status['reset_time'] - current_time
            return False, f"Rate limit exceeded. Reset in {time_left}"
        
        return True, f"Requests: {requests_made}/{self.max_requests}"
    
    def increment_usage(self):
        st.session_state.rate_limit_status['requests'] += 1

rate_limiter = RateLimiter()

# Multi-Agent System
class ContentAgent:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
    
    async def execute_task(self, prompt: str, api_key: str) -> str:
        """Execute agent-specific task"""
        agent_prompts = {
            "Researcher": f"Research and gather key information about: {prompt}. Provide facts, statistics, and current trends.",
            "Writer": f"Create engaging, well-structured content about: {prompt}. Focus on clarity and reader engagement.",
            "Editor": f"Review and improve this content for clarity, flow, and impact: {prompt}",
            "SEO_Optimizer": f"Optimize this content for SEO with relevant keywords and structure: {prompt}"
        }
        
        specialized_prompt = agent_prompts.get(self.name, prompt)
        return await generate_with_llama_async(specialized_prompt, api_key)

# Enhanced API Functions
def validate_api_key_alternative(api_key):
    """Alternative API key validation using models endpoint"""
    try:
        api_key = api_key.strip()
        headers = {"Authorization": f"Bearer {api_key}"}
        
        response = requests.get(
            "https://api.together.xyz/v1/models",
            headers=headers,
            timeout=10
        )
        
        return response.status_code == 200
        
    except Exception as e:
        st.error(f"Validation error: {str(e)}")
        return False

def validate_api_key(api_key):
    """Enhanced API key validation with detailed feedback"""
    try:
        api_key = api_key.strip()
        
        # First try the simpler models endpoint
        if validate_api_key_alternative(api_key):
            return True
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        test_data = {
            "model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 5,
            "temperature": 0.1
        }
        
        response = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers=headers,
            json=test_data,
            timeout=15
        )
        
        return response.status_code == 200
    except Exception as e:
        st.error(f"Validation exception: {str(e)}")
        return False

def generate_with_llama_streaming(prompt, api_key, temperature=0.7, max_tokens=800, placeholder=None):
    """Enhanced generation with streaming simulation and metrics tracking"""
    start_time = time.time()
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "messages": [
                {"role": "system", "content": "You are an expert content creator. Generate high-quality, engaging content based on the user's requirements."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "stop": ["<|eot_id|>"]
        }
        
        response = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content'].strip()
            
            # Simulate streaming effect
            if placeholder:
                words = content.split()
                streamed_text = ""
                for i, word in enumerate(words):
                    streamed_text += word + " "
                    placeholder.markdown(f'<div class="streaming-content">{streamed_text}<span style="opacity:0.5">▋</span></div>', unsafe_allow_html=True)
                    time.sleep(0.05)  # Streaming delay
                
                # Final content without cursor
                placeholder.markdown(f'<div class="streaming-content">{content}</div>', unsafe_allow_html=True)
            
            # Track metrics
            end_time = time.time()
            response_time = end_time - start_time
            
            metrics = GenerationMetrics(
                tokens_used=len(content.split()) * 1.3,  # Rough estimation
                response_time=response_time,
                quality_score=analyze_content_quality(content)['readability_score'],
                cost_estimate=len(content.split()) * 0.0001,  # Rough cost estimation
                timestamp=datetime.now()
            )
            
            st.session_state.generation_metrics.append(metrics)
            rate_limiter.increment_usage()
            
            # Update API usage stats
            st.session_state.api_usage_stats['total_requests'] += 1
            st.session_state.api_usage_stats['total_tokens'] += int(metrics.tokens_used)
            st.session_state.api_usage_stats['total_cost'] += metrics.cost_estimate
            
            return content
        else:
            return f"Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"Error generating content: {str(e)}"

async def generate_with_llama_async(prompt, api_key, temperature=0.7, max_tokens=400):
    """Async version for multi-agent processing"""
    # Simulate async call (in real implementation, use aiohttp)
    return generate_with_llama_streaming(prompt, api_key, temperature, max_tokens)

def multi_agent_content_generation(prompt, api_key, use_agents=['Researcher', 'Writer', 'Editor']):
    """Orchestrate multiple AI agents for content creation"""
    agents = [ContentAgent(name, name) for name in use_agents]
    st.session_state.agent_tasks = []
    
    # Initialize agent tasks
    for agent in agents:
        task = AgentTask(name=agent.name, status='waiting')
        st.session_state.agent_tasks.append(task)
    
    results = {}
    
    # Execute agents sequentially (in real app, use asyncio.gather for parallel)
    for i, agent in enumerate(agents):
        # Update status
        st.session_state.agent_tasks[i].status = 'active'
        st.session_state.agent_tasks[i].start_time = datetime.now()
        
        # Generate content
        if agent.name == 'Researcher':
            agent_prompt = f"Research and provide key information, statistics, and insights about: {prompt}"
        elif agent.name == 'Writer':
            research_context = results.get('Researcher', '')
            agent_prompt = f"Using this research context: {research_context}\n\nWrite comprehensive, engaging content about: {prompt}"
        elif agent.name == 'Editor':
            writer_content = results.get('Writer', prompt)
            agent_prompt = f"Edit and improve this content for clarity, engagement, and flow: {writer_content}"
        else:
            agent_prompt = prompt
        
        result = generate_with_llama_streaming(agent_prompt, api_key, max_tokens=600)
        results[agent.name] = result
        
        # Update completion
        st.session_state.agent_tasks[i].status = 'completed'
        st.session_state.agent_tasks[i].result = result[:100] + "..."
        st.session_state.agent_tasks[i].end_time = datetime.now()
    
    # Return final result (Editor's output or Writer's if no Editor)
    return results.get('Editor', results.get('Writer', results.get('Researcher', '')))

def analyze_content_quality(text):
    """Enhanced content analysis with more metrics"""
    if not text or len(text.strip()) == 0:
        return {
            'readability_score': 0,
            'grade_level': 0,
            'word_count': 0,
            'sentence_count': 0,
            'seo_score': 0,
            'sentiment': 'neutral',
            'keyword_density': 0,
            'engagement_score': 0
        }
    
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Readability
    try:
        readability = flesch_reading_ease(text)
        grade_level = flesch_kincaid_grade(text)
    except:
        readability = 50
        grade_level = 8
    
    # Enhanced SEO score
    seo_factors = [
        len(words) > 300,  # Good length
        len(sentences) > 5,  # Multiple sentences
        any(word.isupper() for word in words[:10]),  # Headers/emphasis
        text.count('?') > 0,  # Questions for engagement
        text.count('!') > 0   # Exclamations for engagement
    ]
    seo_score = min(100, sum(seo_factors) * 20 + len(words) * 0.1)
    
    # Engagement score (based on questions, calls to action, etc.)
    engagement_indicators = ['?', '!', 'you', 'your', 'how', 'why', 'what', 'discover', 'learn']
    engagement_score = min(100, sum(text.lower().count(indicator) for indicator in engagement_indicators) * 5)
    
    # Simple sentiment analysis
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'best', 'success']
    negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disappointing', 'fail']
    
    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count > neg_count:
        sentiment = 'positive'
    elif neg_count > pos_count:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    
    return {
        'readability_score': max(0, min(100, readability)),
        'grade_level': max(1, min(20, grade_level)),
        'word_count': len(words),
        'sentence_count': len(sentences),
        'seo_score': min(100, seo_score),
        'sentiment': sentiment,
        'keyword_density': len(set(words)) / len(words) * 100 if words else 0,
        'engagement_score': engagement_score
    }

def get_content_templates():
    """Enhanced content framework templates"""
    return {
        "AIDA": "Write using the AIDA framework:\n- Attention: Hook the reader with a compelling opening\n- Interest: Build interest with relevant information\n- Desire: Create desire by highlighting benefits\n- Action: Clear call to action\n\nTopic: {topic}",
        "PAS": "Write using the PAS framework:\n- Problem: Clearly identify the problem\n- Agitation: Emphasize the pain points\n- Solution: Present your solution\n\nTopic: {topic}",
        "Before-After-Bridge": "Write using the Before-After-Bridge framework:\n- Before: Current problematic situation\n- After: Desired ideal outcome\n- Bridge: Your solution as the path\n\nTopic: {topic}",
        "Problem-Solution": "Write about {topic} using:\n- Clear problem identification\n- Comprehensive solution explanation\n- Benefits and positive outcomes",
        "Storytelling": "Write a compelling story about {topic} with:\n- Engaging opening hook\n- Character development\n- Conflict and resolution\n- Clear message or lesson",
        "STAR": "Write using the STAR method:\n- Situation: Set the context\n- Task: Describe what needed to be done\n- Action: Explain the actions taken\n- Result: Share the outcomes\n\nTopic: {topic}"
    }

def generate_analytics_dashboard():
    """Create advanced analytics dashboard"""
    if not st.session_state.generation_metrics:
        st.info("📊 Generate some content to see analytics")
        return
    
    # Create metrics dataframe
    metrics_data = []
    for metric in st.session_state.generation_metrics[-20:]:  # Last 20 generations
        metrics_data.append({
            'timestamp': metric.timestamp,
            'quality_score': metric.quality_score,
            'response_time': metric.response_time,
            'tokens_used': metric.tokens_used,
            'cost_estimate': metric.cost_estimate
        })
    
    df = pd.DataFrame(metrics_data)
    
    if not df.empty:
        # Quality trend chart
        fig_quality = px.line(df, x='timestamp', y='quality_score', 
                            title='Content Quality Trend',
                            labels={'quality_score': 'Quality Score', 'timestamp': 'Time'})
        fig_quality.update_layout(height=300)
        st.plotly_chart(fig_quality, use_container_width=True)
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_quality = df['quality_score'].mean()
            st.metric("Avg Quality", f"{avg_quality:.1f}/100", 
                     delta=f"{avg_quality - 70:.1f}" if len(df) > 1 else None)
        
        with col2:
            avg_response_time = df['response_time'].mean()
            st.metric("Avg Response Time", f"{avg_response_time:.1f}s",
                     delta=f"{avg_response_time - 5:.1f}s" if len(df) > 1 else None)
        
        with col3:
            total_tokens = df['tokens_used'].sum()
            st.metric("Total Tokens", f"{int(total_tokens):,}")
        
        with col4:
            total_cost = df['cost_estimate'].sum()
            st.metric("Estimated Cost", f"${total_cost:.4f}")

# Main Application
def main():
    # Enhanced Header
    st.markdown("""
    <div class="main-header">
        <div class="enterprise-header">🚀 ContentStudio AI Pro</div>
        <div class="enterprise-subtitle">Enterprise-Grade Content Creation Platform • Multi-Agent AI • Real-time Analytics</div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("🔑 API Configuration")
        
        # API Key input
        import os
        default_api_key = os.getenv("TOGETHER_API_KEY", "")
        
        api_key = st.text_input(
            "Together AI API Key:", 
            value=default_api_key,
            type="password",
            help="Enter your Together AI API key to get started"
        )
        
        # Rate limiting status
        can_proceed, rate_status = rate_limiter.check_rate_limit()
        rate_class = "rate-limit-ok" if can_proceed else "rate-limit-danger"
        
        st.markdown(f'<div class="{rate_class}">🚦 {rate_status}</div>', unsafe_allow_html=True)
        
        # API validation
        if api_key and st.button("🔍 Validate API Key"):
            with st.spinner("Validating API key..."):
                if validate_api_key(api_key):
                    st.session_state.api_validated = True
                    st.session_state.api_key = api_key
                    st.success("✅ API Key validated successfully!")
                else:
                    st.error("❌ API key validation failed.")
        
        # Auto-validation
        if api_key and not st.session_state.api_validated and len(api_key) > 10:
            with st.spinner("Validating API key..."):
                if validate_api_key(api_key):
                    st.session_state.api_validated = True
                    st.session_state.api_key = api_key
                    st.success("✅ API Key validated!")
        
        if st.session_state.api_validated:
            st.success("🟢 API Connected")
            
            # Enhanced Content Settings
            st.header("🎯 Content Configuration")
            
            # Content type selection
            content_types = {
                "Blog Post": "Write a comprehensive, SEO-optimized blog post about",
                "Social Media": "Create viral-worthy social media content about",
                "Email Campaign": "Write a high-converting email campaign about",
                "Ad Copy": "Create persuasive advertisement copy for",
                "Product Description": "Write a compelling product description for",
                "Press Release": "Write a professional press release about",
                "Technical Article": "Write an in-depth technical article about",
                "Case Study": "Create a detailed case study about",
                "White Paper": "Write a comprehensive white paper on",
                "Sales Copy": "Create high-converting sales copy for"
            }
            
            selected_type = st.selectbox("Content Type:", list(content_types.keys()))
            
            # Enhanced framework selection
            st.subheader("📋 Content Framework")
            frameworks = list(get_content_templates().keys()) + ["Custom"]
            selected_framework = st.selectbox("Framework:", frameworks)
            
            # Multi-agent selection
            st.subheader("🤖 AI Agent Configuration")
            available_agents = ["Researcher", "Writer", "Editor", "SEO_Optimizer"]
            selected_agents = st.multiselect(
                "Select AI Agents:", 
                available_agents,
                default=["Writer", "Editor"]
            )
            
            # Advanced settings
            with st.expander("⚙️ Advanced Settings"):
                temperature = st.slider("Creativity Level:", 0.1, 1.5, 0.7, 0.1)
                max_tokens = st.slider("Content Length:", 400, 2000, 1000, 100)
                use_streaming = st.checkbox("Real-time Streaming", value=True)
                generate_variations = st.checkbox("Generate Multiple Variations", value=True)
            
            # Brand voice section
            st.header("🎯 Brand Voice Training")
            uploaded_files = st.file_uploader(
                "Upload brand voice samples:",
                type=['txt'],
                accept_multiple_files=True,
                help="Upload examples of your writing style"
            )
            
            if uploaded_files:
                for file in uploaded_files:
                    content = file.read().decode('utf-8')
                    if content not in st.session_state.brand_voice_samples:
                        st.session_state.brand_voice_samples.append(content)
                st.success(f"📚 {len(st.session_state.brand_voice_samples)} brand samples loaded")

    # Main content area
    if not st.session_state.api_validated:
        st.warning("🔑 Please enter your Together AI API key in the sidebar to get started")
        st.info("""
        **🚀 Enterprise Features:**
        - **Multi-Agent AI System** - Researcher, Writer, Editor, SEO agents working together
        - **Real-time Streaming Generation** - See content appear word by word
        - **Advanced Analytics Dashboard** - Track performance, costs, and quality metrics
        - **Rate Limiting & Usage Analytics** - Enterprise-grade API management
        - **Brand Voice Training** - Upload samples to maintain consistent voice
        """)
    else:
        if not can_proceed:
            st.error("🚫 Rate limit exceeded. Please wait before making more requests.")
            return
        
        # Main interface with enhanced layout
        tab1, tab2, tab3 = st.tabs(["🎨 Content Creation", "📊 Analytics Dashboard", "🔧 Agent Status"])
        
        with tab1:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.header("💭 Intelligent Content Creation")
                
                # Smart prompt builder
                user_topic = st.text_area(
                    "What would you like to create content about?",
                    height=120,
                    placeholder="Enter your topic, keywords, or detailed requirements...\n\nExample: 'Create a blog post about AI in healthcare focusing on diagnostic applications, include recent statistics and case studies'"
                )
                
                # Enhanced smart suggestions
                if user_topic:
                    st.subheader("💡 AI-Powered Suggestions")
                    suggestions = [
                        f"Latest trends and innovations in {user_topic}",
                        f"Complete beginner's guide to {user_topic}",
                        f"Expert strategies for {user_topic}",
                        f"Common challenges and solutions in {user_topic}",
                        f"Future predictions for {user_topic}",
                        f"ROI and business impact of {user_topic}"
                    ]
                    
                    cols = st.columns(3)
                    for i, suggestion in enumerate(suggestions[:6]):
                        if cols[i % 3].button(f"💡 {suggestion.split()[-2:][0].title()}", key=f"suggestion_{i}"):
                            user_topic = suggestion
                            st.experimental_rerun()
                
                # Generate button with enhanced features
                generate_col1, generate_col2 = st.columns([3, 1])
                
                with generate_col1:
                    if st.button("🚀 Generate Content with AI Agents", type="primary", disabled=not user_topic):
                        if user_topic:
                            # Build the enhanced prompt
                            base_prompt = content_types[selected_type] + " " + user_topic
                            
                            # Add framework if selected
                            if selected_framework != "Custom":
                                template = get_content_templates()[selected_framework]
                                base_prompt = template.format(topic=user_topic)
                            
                            # Add tone and advanced instructions
                            base_prompt += f"\n\nRequirements:\n- Tone: Professional and engaging\n- Include relevant statistics and examples\n- Optimize for SEO\n- Make it actionable and valuable"
                            
                            # Add brand voice context if available
                            if st.session_state.brand_voice_samples:
                                brand_context = "\n\nBrand voice reference:\n" + "\n---\n".join(st.session_state.brand_voice_samples[:2])
                                base_prompt += brand_context + "\n\nMaintain this writing style and voice."
                            
                            # Use multi-agent generation or single generation
                            if len(selected_agents) > 1:
                                with st.spinner("🤖 AI Agents working on your content..."):
                                    main_content = multi_agent_content_generation(
                                        base_prompt, 
                                        st.session_state.api_key,
                                        selected_agents
                                    )
                            else:
                                # Streaming placeholder
                                streaming_placeholder = st.empty()
                                main_content = generate_with_llama_streaming(
                                    base_prompt, 
                                    st.session_state.api_key,
                                    temperature=temperature,
                                    max_tokens=max_tokens,
                                    placeholder=streaming_placeholder if use_streaming else None
                                )
                            
                            st.session_state.generated_content = main_content
                            
                            # Generate variations if requested
                            if generate_variations:
                                with st.spinner("✨ Creating content variations..."):
                                    variations = []
                                    variation_styles = [
                                        ("Professional", 0.6, "professional and authoritative"),
                                        ("Conversational", 0.8, "casual and conversational"), 
                                        ("Persuasive", 0.7, "persuasive and compelling")
                                    ]
                                    
                                    for style_name, temp, style_desc in variation_styles:
                                        variation_prompt = f"Rewrite this content in a {style_desc} style, maintaining the key information: {main_content[:500]}..."
                                        variation_content = generate_with_llama_streaming(
                                            variation_prompt, 
                                            st.session_state.api_key, 
                                            temperature=temp,
                                            max_tokens=max_tokens//2
                                        )
                                        variations.append({
                                            'content': variation_content,
                                            'style': style_name,
                                            'temperature': temp
                                        })
                                    
                                    st.session_state.content_variations = variations
                            
                            # Add to enhanced history
                            st.session_state.generation_history.append({
                                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'topic': user_topic,
                                'type': selected_type,
                                'agents_used': selected_agents,
                                'content': main_content[:150] + "...",
                                'quality_score': analyze_content_quality(main_content)['readability_score']
                            })
                
                with generate_col2:
                    if st.button("🔄 Quick Regenerate"):
                        if st.session_state.generated_content:
                            regenerate_prompt = f"Regenerate and improve this content: {user_topic}"
                            with st.spinner("Regenerating..."):
                                new_content = generate_with_llama_streaming(
                                    regenerate_prompt,
                                    st.session_state.api_key,
                                    temperature=temperature + 0.2
                                )
                                st.session_state.generated_content = new_content
                                st.experimental_rerun()
                
                # Display generated content with enhanced UI
                if st.session_state.generated_content:
                    st.subheader("📄 Generated Content")
                    
                    # Content quality indicator
                    quality_analysis = analyze_content_quality(st.session_state.generated_content)
                    quality_score = int((quality_analysis['readability_score'] + quality_analysis['seo_score'] + quality_analysis['engagement_score']) / 3)
                    
                    quality_color = 'green' if quality_score > 70 else 'orange' if quality_score > 50 else 'red'
                    st.markdown(f"""
                    <div style="background: linear-gradient(90deg, rgba({quality_color=='green' and '40,167,69' or quality_color=='orange' and '255,193,7' or '220,53,69'}, 0.1) 0%, rgba(255,255,255,0.1) 100%); 
                                padding: 1rem; border-radius: 10px; margin: 1rem 0; border-left: 4px solid {quality_color};">
                        <strong>📊 Quality Score: {quality_score}/100</strong> | 
                        📝 {quality_analysis['word_count']} words | 
                        📈 SEO: {quality_analysis['seo_score']:.0f}/100 | 
                        🎯 Engagement: {quality_analysis['engagement_score']:.0f}/100
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Content tabs with variations
                    if st.session_state.content_variations:
                        tabs = st.tabs(["🎯 Main Content"] + [f"✨ {var['style']}" for var in st.session_state.content_variations])
                        
                        with tabs[0]:
                            st.markdown(f'<div class="streaming-content">{st.session_state.generated_content}</div>', unsafe_allow_html=True)
                        
                        for i, variation in enumerate(st.session_state.content_variations):
                            with tabs[i+1]:
                                st.markdown(f'<div class="streaming-content">{variation["content"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="streaming-content">{st.session_state.generated_content}</div>', unsafe_allow_html=True)
                    
                    # Enhanced action buttons
                    st.subheader("🛠️ Content Actions")
                    action_col1, action_col2, action_col3, action_col4, action_col5 = st.columns(5)
                    
                    with action_col1:
                        if st.button("✨ AI Improve"):
                            improve_prompt = f"Significantly improve this content by enhancing clarity, engagement, and adding more valuable insights: {st.session_state.generated_content}"
                            with st.spinner("AI improving content..."):
                                improved = generate_with_llama_streaming(improve_prompt, st.session_state.api_key, temperature=0.6)
                                st.session_state.generated_content = improved
                                st.experimental_rerun()
                    
                    with action_col2:
                        if st.button("📏 Optimize Length"):
                            current_length = len(st.session_state.generated_content.split())
                            target = "shorter and more concise" if current_length > 500 else "more detailed and comprehensive"
                            optimize_prompt = f"Make this content {target} while maintaining all key points: {st.session_state.generated_content}"
                            with st.spinner("Optimizing length..."):
                                optimized = generate_with_llama_streaming(optimize_prompt, st.session_state.api_key)
                                st.session_state.generated_content = optimized
                                st.experimental_rerun()
                    
                    with action_col3:
                        if st.button("🔍 SEO Boost"):
                            seo_prompt = f"Optimize this content for SEO by adding relevant keywords, improving structure, and enhancing readability: {st.session_state.generated_content}"
                            with st.spinner("SEO optimizing..."):
                                seo_optimized = generate_with_llama_streaming(seo_prompt, st.session_state.api_key, temperature=0.5)
                                st.session_state.generated_content = seo_optimized
                                st.experimental_rerun()
                    
                    with action_col4:
                        if st.button("🎨 Change Tone"):
                            tone_options = ["more professional", "more casual", "more persuasive", "more technical", "more creative"]
                            selected_tone_change = st.selectbox("", tone_options, key="tone_change")
                            tone_prompt = f"Rewrite this content to be {selected_tone_change}: {st.session_state.generated_content}"
                            with st.spinner("Changing tone..."):
                                tone_changed = generate_with_llama_streaming(tone_prompt, st.session_state.api_key, temperature=0.7)
                                st.session_state.generated_content = tone_changed
                                st.experimental_rerun()
                    
                    with action_col5:
                        if st.button("📱 Repurpose"):
                            repurpose_options = ["social media posts", "email subject lines", "bullet points", "executive summary"]
                            selected_repurpose = st.selectbox("", repurpose_options, key="repurpose")
                            repurpose_prompt = f"Convert this content into {selected_repurpose}: {st.session_state.generated_content}"
                            with st.spinner("Repurposing content..."):
                                repurposed = generate_with_llama_streaming(repurpose_prompt, st.session_state.api_key, temperature=0.6)
                                st.info(f"**Repurposed as {selected_repurpose}:**\n\n{repurposed}")
            
            with col2:
                st.header("📊 Real-time Analytics")
                
                if st.session_state.generated_content:
                    analysis = analyze_content_quality(st.session_state.generated_content)
                    
                    # Enhanced quality dashboard
                    overall_score = int((analysis['readability_score'] + analysis['seo_score'] + analysis['engagement_score']) / 3)
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="quality-score" style="color: {'#28a745' if overall_score > 70 else '#ffc107' if overall_score > 50 else '#dc3545'}">
                            {overall_score}/100
                        </div>
                        <div style="text-align: center; margin-top: 0.5rem; font-weight: 600;">Overall Quality Score</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Detailed metrics with progress bars
                    st.subheader("📈 Detailed Metrics")
                    
                    metrics = [
                        ("📖 Readability", analysis['readability_score'], 100),
                        ("🔍 SEO Score", analysis['seo_score'], 100),
                        ("🎯 Engagement", analysis['engagement_score'], 100),
                        ("📝 Word Density", analysis['keyword_density'], 50)
                    ]
                    
                    for metric_name, value, max_val in metrics:
                        progress = min(value / max_val, 1.0)
                        color = '#28a745' if progress > 0.7 else '#ffc107' if progress > 0.5 else '#dc3545'
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <strong>{metric_name}</strong><br>
                            <div style="background: #e9ecef; border-radius: 10px; height: 10px; margin: 0.5rem 0;">
                                <div style="background: {color}; width: {progress*100}%; height: 100%; border-radius: 10px;"></div>
                            </div>
                            <span style="color: {color}; font-size: 1.2rem; font-weight: 600;">{value:.1f}/{max_val}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Content statistics
                    st.markdown(f"""
                    <div class="metric-card">
                        <strong>📊 Content Statistics</strong><br>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
                            <div>Words: <strong>{analysis['word_count']}</strong></div>
                            <div>Sentences: <strong>{analysis['sentence_count']}</strong></div>
                            <div>Grade Level: <strong>{analysis['grade_level']:.1f}</strong></div>
                            <div>Sentiment: <strong>{analysis['sentiment'].title()}</strong></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # API Usage Stats
                    st.subheader("⚡ API Usage")
                    usage_stats = st.session_state.api_usage_stats
                    
                    usage_col1, usage_col2 = st.columns(2)
                    with usage_col1:
                        st.metric("Total Requests", usage_stats['total_requests'])
                        st.metric("Total Tokens", f"{usage_stats['total_tokens']:,}")
                    with usage_col2:
                        st.metric("Estimated Cost", f"${usage_stats['total_cost']:.4f}")
                        st.metric("Avg Quality", f"{sum(m.quality_score for m in st.session_state.generation_metrics[-5:]) / min(5, len(st.session_state.generation_metrics)):.1f}/100" if st.session_state.generation_metrics else "N/A")
                    
                    # Export options
                    st.subheader("📤 Export Options")
                    
                    export_col1, export_col2 = st.columns(2)
                    
                    with export_col1:
                        if st.button("📋 Copy to Clipboard"):
                            st.code(st.session_state.generated_content, language=None)
                            st.success("Content ready to copy!")
                    
                    with export_col2:
                        # Enhanced download options
                        export_formats = ["TXT", "MD", "HTML"]
                        export_format = st.selectbox("Format:", export_formats)
                        
                        if export_format == "MD":
                            export_content = f"# Generated Content\n\n{st.session_state.generated_content}\n\n---\n*Generated by ContentStudio AI Pro*"
                            mime_type = "text/markdown"
                            file_ext = "md"
                        elif export_format == "HTML":
                            export_content = f"<html><body><h1>Generated Content</h1><p>{st.session_state.generated_content.replace(chr(10), '</p><p>')}</p></body></html>"
                            mime_type = "text/html"
                            file_ext = "html"
                        else:
                            export_content = st.session_state.generated_content
                            mime_type = "text/plain"
                            file_ext = "txt"
                        
                        if st.download_button(
                            label=f"💾 Download as {export_format}",
                            data=export_content,
                            file_name=f"content_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_ext}",
                            mime=mime_type
                        ):
                            st.success("File downloaded!")
                
                else:
                    st.info("📊 Analytics will appear here after content generation")
                
                # Generation history with enhanced display
                if st.session_state.generation_history:
                    st.subheader("🕒 Recent Generations")
                    for i, item in enumerate(reversed(st.session_state.generation_history[-5:])):
                        with st.expander(f"{item['timestamp']} - {item['type']} (Quality: {item.get('quality_score', 'N/A')})"):
                            st.write(f"**Topic:** {item['topic']}")
                            st.write(f"**Agents Used:** {', '.join(item.get('agents_used', ['Standard']))}")
                            st.write(f"**Preview:** {item['content']}")
        
        with tab2:
            st.header("📊 Advanced Analytics Dashboard")
            generate_analytics_dashboard()
            
            # Enhanced usage analytics
            if st.session_state.generation_metrics:
                st.subheader("📈 Performance Trends")
                
                # Create comprehensive charts
                df_metrics = pd.DataFrame([{
                    'timestamp': m.timestamp,
                    'quality_score': m.quality_score,
                    'response_time': m.response_time,
                    'tokens_used': m.tokens_used,
                    'cost_estimate': m.cost_estimate
                } for m in st.session_state.generation_metrics])
                
                # Response time trend
                fig_response = px.line(df_metrics, x='timestamp', y='response_time',
                                     title='Response Time Trend',
                                     labels={'response_time': 'Response Time (s)', 'timestamp': 'Time'})
                st.plotly_chart(fig_response, use_container_width=True)
                
                # Token usage over time
                fig_tokens = px.bar(df_metrics, x='timestamp', y='tokens_used',
                                  title='Token Usage Per Generation',
                                  labels={'tokens_used': 'Tokens Used', 'timestamp': 'Time'})
                st.plotly_chart(fig_tokens, use_container_width=True)
                
                # Cost analysis
                total_cost = df_metrics['cost_estimate'].sum()
                avg_cost_per_generation = df_metrics['cost_estimate'].mean()
                
                cost_col1, cost_col2, cost_col3 = st.columns(3)
                with cost_col1:
                    st.metric("Total Cost", f"${total_cost:.4f}")
                with cost_col2:
                    st.metric("Avg Cost/Generation", f"${avg_cost_per_generation:.4f}")
                with cost_col3:
                    projected_monthly = avg_cost_per_generation * 30 * st.session_state.api_usage_stats['total_requests']
                    st.metric("Projected Monthly", f"${projected_monthly:.2f}")
        
        with tab3:
            st.header("🤖 AI Agent Status")
            
            if st.session_state.agent_tasks:
                st.subheader("🔄 Current Agent Pipeline")
                
                for task in st.session_state.agent_tasks:
                    if task.status == 'active':
                        status_class = "agent-active"
                        status_icon = "🔄"
                    elif task.status == 'completed':
                        status_class = "agent-completed"
                        status_icon = "✅"
                    elif task.status == 'failed':
                        status_class = "agent-error"
                        status_icon = "❌"
                    else:
                        status_class = "agent-waiting"
                        status_icon = "⏳"
                    
                    duration = ""
                    if task.start_time and task.end_time:
                        duration = f" ({(task.end_time - task.start_time).total_seconds():.1f}s)"
                    elif task.start_time:
                        duration = f" ({(datetime.now() - task.start_time).total_seconds():.1f}s)"
                    
                    st.markdown(f"""
                    <div class="agent-status {status_class}">
                        {status_icon} <strong>{task.name} Agent</strong> - {task.status.title()}{duration}
                        {f'<br><small>{task.result}</small>' if task.result else ''}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("🤖 No agents currently active. Start a generation to see agent status.")
            
            # Agent performance metrics
            if st.session_state.generation_history:
                st.subheader("📊 Agent Performance Analysis")
                
                # Analyze which agent combinations work best
                agent_performance = defaultdict(list)
                for item in st.session_state.generation_history:
                    agents_key = ", ".join(sorted(item.get('agents_used', ['Standard'])))
                    quality = item.get('quality_score', 0)
                    agent_performance[agents_key].append(quality)
                
                if agent_performance:
                    performance_data = []
                    for agents, scores in agent_performance.items():
                        performance_data.append({
                            'agents': agents,
                            'avg_quality': sum(scores) / len(scores),
                            'generations': len(scores)
                        })
                    
                    df_performance = pd.DataFrame(performance_data)
                    
                    fig_agents = px.bar(df_performance, x='agents', y='avg_quality',
                                      title='Average Quality Score by Agent Configuration',
                                      labels={'avg_quality': 'Average Quality Score', 'agents': 'Agent Configuration'})
                    st.plotly_chart(fig_agents, use_container_width=True)

# Footer with enhanced branding
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 10px; margin-top: 2rem;">
    <div style="font-size: 1.5rem; font-weight: 700; margin-bottom: 0.5rem;">🚀 ContentStudio AI Pro</div>
    <div style="font-size: 1rem; margin-bottom: 1rem;">Enterprise-Grade Content Creation Platform</div>
    <div style="font-size: 0.9rem; opacity: 0.8;">
        Powered by Llama 3.1 🦙 | Multi-Agent AI | Real-time Analytics | Enterprise Ready
    </div>
    <div style="font-size: 0.8rem; margin-top: 1rem; opacity: 0.6;">
        © 2025 ContentStudio AI Pro - Advanced AI Content Generation Platform
    </div>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()