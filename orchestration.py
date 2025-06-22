import json
import asyncio
from datetime import datetime, timedelta
import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel
from enum import Enum
import argparse
import numpy as np
import random
import math
from dataclasses import dataclass
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.utils.configuration import Settings, set_global

# =============================================================================
# WEAV AGENT CONTROL FLAGS
# =============================================================================
ENABLE_PROFILE_SUMMARIZER_AGENT = True
ENABLE_CONTEXT_MONITOR_AGENT = True
ENABLE_INTEREST_MATCHER_AGENT = True
ENABLE_PROXIMITY_VALIDATOR_AGENT = True
ENABLE_TIMING_AGENT = True
ENABLE_CONTEXT_BUILDER_AGENT = True
ENABLE_INTRO_WRITER_AGENT = True
ENABLE_FOLLOWUP_AGENT = False  # Optional for MVP

# Configure Langroid to use OpenAI
def setup_langroid():
    """Configure Langroid with OpenAI settings"""
    settings = Settings()
    set_global(settings)
    
    llm_config = OpenAIGPTConfig(
        chat_model="gpt-4",
        chat_context_length=8000,  
        max_output_tokens=1000,    
        timeout=45,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    return llm_config

# -------------------------------------
# Data Models for Weav
# -------------------------------------
@dataclass
class UserProfile:
    user_id: str
    name: str
    linkedin_title: str
    company: str
    bio: str
    work_passion: str
    interests: List[str]
    location: Dict[str, Any]
    phone_behavior: Dict[str, Any]
    schedule: Dict[str, Any]
    interaction_history: Dict[str, Any]
    communication_style: str = "professional"
    
@dataclass
class ConferenceContext:
    name: str
    current_time: str
    current_session: str
    venue_zones: List[str]
    crowd_density: str
    noise_level: str
    social_appropriateness: str

class MessageType(Enum):
    ALERT = "alert"
    CONSULTATION_REQUEST = "consultation_request"
    RECOMMENDATION = "recommendation"
    STATUS_UPDATE = "status_update"
    DECISION_INPUT = "decision_input"

class AgentMessage(BaseModel):
    sender_agent: str
    recipient_agent: str
    message_type: MessageType
    content: Dict[str, Any]
    priority: int
    timestamp: datetime
    requires_response: bool = False

# -------------------------------------
# Weav-specific Collaboration Hub
# -------------------------------------
class WeavCollaborationHub:
    def __init__(self):
        self.message_queue: List[AgentMessage] = []
        self.agent_states: Dict[str, Dict] = {}
        self.consultation_history: List[Dict] = []
        self.active_agents: List[str] = []
        self.decision_sequence: List[Dict] = []
        self.introduction_opportunities: List[Dict] = []

    async def broadcast_message(self, message: AgentMessage):
        """Send message to all relevant agents"""
        self.message_queue.append(message)

    async def request_consultation(self, requesting_agent: str, consultation_type: str, 
                                 context: Dict, target_agents: List[str] = None):
        """Agent requests input from other agents"""
        consultation = {
            "requesting_agent": requesting_agent,
            "consultation_type": consultation_type,
            "context": context,
            "target_agents": target_agents or ["ALL"],
            "timestamp": datetime.utcnow(),
            "responses": {}
        }
        self.consultation_history.append(consultation)
        return consultation

    def get_messages_for_agent(self, agent_name: str) -> List[AgentMessage]:
        """Get all messages for a specific agent"""
        return [msg for msg in self.message_queue 
                if msg.recipient_agent == agent_name or msg.recipient_agent == "ALL"]

    def update_agent_state(self, agent_name: str, state: Dict):
        """Update agent's current state"""
        self.agent_states[agent_name] = state

    def log_introduction_opportunity(self, opportunity: Dict):
        """Log a potential introduction opportunity"""
        self.introduction_opportunities.append({
            **opportunity,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })

# -------------------------------------
# Base Weav Agent
# -------------------------------------
class WeavAgent:
    def __init__(self, agent_name: str, collaboration_hub: WeavCollaborationHub):
        self.agent_name = agent_name
        self.collaboration_hub = collaboration_hub
        self.current_state = {}

    async def process_messages(self):
        """Process incoming messages from other agents"""
        messages = self.collaboration_hub.get_messages_for_agent(self.agent_name)
        for message in messages:
            await self.handle_message(message)

    async def handle_message(self, message: AgentMessage):
        """Handle incoming message - to be implemented by specific agents"""
        pass

    async def send_message(self, recipient: str, message_type: MessageType, 
                          content: Dict, priority: int = 2):
        """Send message to another agent"""
        message = AgentMessage(
            sender_agent=self.agent_name,
            recipient_agent=recipient,
            message_type=message_type,
            content=content,
            priority=priority,
            timestamp=datetime.utcnow()
        )
        await self.collaboration_hub.broadcast_message(message)

    async def run_task_async(self, prompt: str, turns: int = 1):
        """Run a blocking task.run() call in thread pool"""
        def blocking_task():
            return self.task.run(prompt, turns)
        
        return await asyncio.get_event_loop().run_in_executor(None, blocking_task)

# -------------------------------------
# Weav-specific Agents
# -------------------------------------
class ProfileSummarizerAgent(WeavAgent):
    def __init__(self, collaboration_hub: WeavCollaborationHub):
        super().__init__("profile_summarizer_agent", collaboration_hub)
        self.llm_config = setup_langroid()
        config = ChatAgentConfig(
            name="ProfileSummarizerAgent",
            system_message="""
            You are a profile analysis specialist for intelligent networking. 
            Parse raw profile information and extract structured, meaningful insights.
            
            Extract: professional context, personal interests, communication style, networking goals, experience level
            
            Focus on actionable insights that enable high-quality introductions.
            """,
            llm=self.llm_config
        )
        self.agent = ChatAgent(config)
        self.task = Task(self.agent, interactive=False)

    async def process_profile(self, user_profile: UserProfile) -> Dict:
        """Process and structure a user profile"""
        await self.process_messages()
        
        try:
            # Structure the analysis without LLM for speed
            structured_profile = {
                "user_id": user_profile.user_id,
                "professional_summary": self._extract_professional_context(user_profile),
                "personal_interests": self._extract_personal_interests(user_profile),
                "communication_style": self._infer_communication_style(user_profile),
                "networking_goals": self._infer_networking_goals(user_profile),
                "experience_level": self._assess_experience_level(user_profile),
                "key_topics": self._extract_key_topics(user_profile),
                "profile_completeness": self._assess_completeness(user_profile),
                "confidence": 0.9 if self._assess_completeness(user_profile) > 0.7 else 0.6
            }
            
            self.current_state = structured_profile
            self.collaboration_hub.update_agent_state(self.agent_name, structured_profile)
            
            return structured_profile
            
        except Exception as e:
            return {"error": str(e), "confidence": 0.0}

    def _extract_professional_context(self, profile: UserProfile) -> Dict:
        return {
            "title": profile.linkedin_title,
            "company": profile.company,
            "industry": self._infer_industry(profile.linkedin_title, profile.company),
            "role_type": self._categorize_role(profile.linkedin_title)
        }
    
    def _extract_personal_interests(self, profile: UserProfile) -> List[str]:
        bio_interests = []
        bio_lower = profile.bio.lower()
        
        # Simple keyword extraction
        interest_keywords = {
            "outdoor": ["hiking", "outdoor", "nature", "climbing"],
            "fitness": ["running", "fitness", "gym", "exercise"],
            "tech": ["ai", "technology", "coding", "programming"],
            "travel": ["travel", "exploring", "adventure"],
            "family": ["parent", "family", "children"],
            "arts": ["music", "art", "creative", "design"]
        }
        
        for category, keywords in interest_keywords.items():
            if any(keyword in bio_lower for keyword in keywords):
                bio_interests.append(category)
        
        return bio_interests
    
    def _infer_communication_style(self, profile: UserProfile) -> str:
        bio = profile.bio.lower()
        if "enthusiast" in bio or "passionate" in bio:
            return "enthusiastic"
        elif "professional" in bio or profile.linkedin_title.startswith(("Senior", "VP", "Director")):
            return "professional"
        else:
            return "approachable"
    
    def _infer_networking_goals(self, profile: UserProfile) -> List[str]:
        goals = []
        title_lower = profile.linkedin_title.lower()
        
        if "founder" in title_lower or "ceo" in title_lower:
            goals.extend(["fundraising", "partnerships", "talent"])
        elif "product" in title_lower:
            goals.extend(["product_insights", "user_research", "industry_trends"])
        elif "engineer" in title_lower or "developer" in title_lower:
            goals.extend(["technical_learning", "collaboration", "career_growth"])
        else:
            goals.append("professional_growth")
            
        return goals
    
    def _assess_experience_level(self, profile: UserProfile) -> str:
        title = profile.linkedin_title.lower()
        if any(word in title for word in ["senior", "lead", "principal", "director", "vp", "founder"]):
            return "senior"
        elif any(word in title for word in ["junior", "associate", "intern"]):
            return "junior"
        else:
            return "mid-level"
    
    def _extract_key_topics(self, profile: UserProfile) -> List[str]:
        topics = []
        combined_text = f"{profile.linkedin_title} {profile.bio} {profile.work_passion}".lower()
        
        topic_keywords = {
            "ai_ml": ["ai", "artificial intelligence", "machine learning", "ml"],
            "healthtech": ["health", "medical", "wellness", "fitness"],
            "fintech": ["finance", "banking", "payments", "crypto"],
            "climate": ["climate", "sustainability", "green", "carbon"],
            "enterprise": ["enterprise", "b2b", "saas", "business"],
            "consumer": ["consumer", "b2c", "mobile", "app", "product"]
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in combined_text for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def _assess_completeness(self, profile: UserProfile) -> float:
        completeness = 0.0
        if profile.linkedin_title: completeness += 0.3
        if profile.bio: completeness += 0.3
        if profile.work_passion: completeness += 0.4
        return completeness
    
    def _infer_industry(self, title: str, company: str) -> str:
        combined = f"{title} {company}".lower()
        if any(word in combined for word in ["health", "medical", "bio", "wellness", "fitness"]):
            return "healthcare"
        elif any(word in combined for word in ["tech", "software", "ai", "engineer"]):
            return "technology"
        elif any(word in combined for word in ["finance", "bank", "invest"]):
            return "finance"
        else:
            return "other"
    
    def _categorize_role(self, title: str) -> str:
        title_lower = title.lower()
        if any(word in title_lower for word in ["product", "pm"]):
            return "product"
        elif any(word in title_lower for word in ["engineer", "developer", "tech"]):
            return "engineering"
        elif any(word in title_lower for word in ["founder", "ceo", "cto"]):
            return "leadership"
        elif any(word in title_lower for word in ["sales", "business", "bd"]):
            return "business"
        else:
            return "other"

class ContextMonitorAgent(WeavAgent):
    def __init__(self, collaboration_hub: WeavCollaborationHub):
        super().__init__("context_monitor_agent", collaboration_hub)
        self.monitoring_active = False
        self.user_states = {}
        self.last_monitoring_time = {}

    async def monitor_user_context(self, users: List[UserProfile], context: ConferenceContext) -> Dict:
        """Monitor real-time context for all users"""
        await self.process_messages()
        
        # Rate limiting - don't monitor the same users too frequently
        user_ids = [user.user_id for user in users]
        cache_key = ":".join(sorted(user_ids))
        
        if cache_key in self.last_monitoring_time:
            last_time = self.last_monitoring_time[cache_key]
            if (datetime.utcnow() - last_time).total_seconds() < 30:  # 30 second cooldown
                return {"cached": True, "confidence": 0.8}
        
        self.last_monitoring_time[cache_key] = datetime.utcnow()
        
        context_analysis = {
            "monitoring_timestamp": datetime.utcnow().isoformat() + "Z",
            "user_availability": {},
            "environmental_factors": {},
            "opportunity_signals": [],
            "confidence": 0.0
        }
        
        # Analyze each user's current state
        for user in users:
            availability = self._assess_user_availability(user)
            context_analysis["user_availability"][user.user_id] = availability
        
        # Analyze environmental context
        context_analysis["environmental_factors"] = self._assess_environmental_context(context)
        
        # Detect opportunity signals
        context_analysis["opportunity_signals"] = self._detect_opportunity_signals(users, context)
        
        # Calculate overall confidence
        context_analysis["confidence"] = self._calculate_monitoring_confidence(context_analysis)
        
        # Only alert if significant opportunities detected
        if len(context_analysis["opportunity_signals"]) >= 2:  # Need multiple positive signals
            await self.send_message(
                "ALL",
                MessageType.ALERT,
                {
                    "context_update": context_analysis,
                    "opportunity_detected": True,
                    "user_pairs": self._identify_potential_pairs(users)
                },
                priority=1
            )
        
        self.current_state = context_analysis
        self.collaboration_hub.update_agent_state(self.agent_name, context_analysis)
        
        return context_analysis

    def _assess_user_availability(self, user: UserProfile) -> Dict:
        """Assess individual user's availability for introductions"""
        phone = user.phone_behavior
        schedule = user.schedule
        interaction = user.interaction_history
        
        availability_score = 0.0
        availability_indicators = []
        
        # Phone behavior analysis
        if phone.get("last_action") in ["scrolling_feed", "browsing_mode"]:
            availability_score += 0.3
            availability_indicators.append("browsing_mode")
        
        if phone.get("last_app") in ["linkedin", "instagram", "twitter"]:
            availability_score += 0.2
            availability_indicators.append("social_apps")
        
        if not phone.get("typing_active", True):
            availability_score += 0.2
            availability_indicators.append("not_typing")
        
        # Schedule analysis
        if schedule.get("current_session") in ["networking_break", "coffee_break", "lunch_break"]:
            availability_score += 0.4
            availability_indicators.append("in_break")
        
        if schedule.get("time_until_next", 0) > 10:
            availability_score += 0.2
            availability_indicators.append("sufficient_time")
        
        # Interaction history
        time_since_last = interaction.get("last_intro", 60)
        if time_since_last > 30:
            availability_score += 0.2
            availability_indicators.append("good_spacing")
        
        return {
            "availability_score": min(availability_score, 1.0),
            "indicators": availability_indicators,
            "receptivity_level": self._determine_receptivity_level(availability_score),
            "last_activity": phone.get("last_action"),
            "energy_level": interaction.get("energy_level", "medium")
        }
    
    def _assess_environmental_context(self, context: ConferenceContext) -> Dict:
        """Assess environmental factors for introductions"""
        return {
            "conference_phase": self._determine_conference_phase(context.current_time),
            "social_appropriateness": context.social_appropriateness,
            "crowd_density": context.crowd_density,
            "noise_level": context.noise_level,
            "networking_friendly": context.current_session in ["networking_break", "coffee_break", "expo_time"]
        }
    
    def _detect_opportunity_signals(self, users: List[UserProfile], context: ConferenceContext) -> List[str]:
        """Detect signals that indicate good introduction opportunities"""
        signals = []
        
        # Multiple users in networking mode
        available_users = [u for u in users if self._assess_user_availability(u)["availability_score"] > 0.6]
        if len(available_users) >= 2:
            signals.append(f"multiple_users_available_{len(available_users)}")
        
        # Optimal timing
        if context.current_session in ["networking_break", "coffee_break"]:
            signals.append("optimal_timing_break")
        
        # Good environmental conditions
        if context.social_appropriateness == "high" and context.crowd_density == "moderate":
            signals.append("favorable_environment")
        
        return signals
    
    def _identify_potential_pairs(self, users: List[UserProfile]) -> List[Tuple[str, str]]:
        """Identify potential user pairs for introduction"""
        pairs = []
        available_users = [u for u in users if self._assess_user_availability(u)["availability_score"] > 0.6]
        
        for i in range(len(available_users)):
            for j in range(i + 1, len(available_users)):
                pairs.append((available_users[i].user_id, available_users[j].user_id))
        
        return pairs
    
    def _determine_receptivity_level(self, score: float) -> str:
        if score >= 0.8:
            return "high"
        elif score >= 0.5:
            return "medium"
        else:
            return "low"
    
    def _determine_conference_phase(self, current_time: str) -> str:
        try:
            hour = int(current_time.split(":")[0])
            if hour < 10:
                return "early_morning"
            elif hour < 12:
                return "morning"
            elif hour < 14:
                return "midday"
            elif hour < 17:
                return "afternoon"
            else:
                return "evening"
        except:
            return "unknown"
    
    def _calculate_monitoring_confidence(self, analysis: Dict) -> float:
        confidence = 0.0
        
        if analysis["user_availability"]:
            confidence += 0.4
        
        if analysis["environmental_factors"]:
            confidence += 0.3
        
        if analysis["opportunity_signals"]:
            confidence += 0.3
        
        return confidence

class InterestMatcherAgent(WeavAgent):
    def __init__(self, collaboration_hub: WeavCollaborationHub):
        super().__init__("interest_matcher_agent", collaboration_hub)
        self.match_cache = {}

    async def calculate_match_score(self, profile_a: Dict, profile_b: Dict) -> Dict:
        """Calculate compatibility score between two users"""
        await self.process_messages()
        
        # Create cache key
        cache_key = f"{profile_a.get('user_id')}:{profile_b.get('user_id')}"
        reverse_key = f"{profile_b.get('user_id')}:{profile_a.get('user_id')}"
        
        # Check cache first
        if cache_key in self.match_cache:
            return self.match_cache[cache_key]
        if reverse_key in self.match_cache:
            cached = self.match_cache[reverse_key]
            result = cached.copy()
            result["user_a_id"] = profile_a.get("user_id")
            result["user_b_id"] = profile_b.get("user_id")
            return result
        
        # Calculate quantitative scores
        match_analysis = {
            "user_a_id": profile_a.get("user_id"),
            "user_b_id": profile_b.get("user_id"),
            "professional_score": self._calculate_professional_compatibility(profile_a, profile_b),
            "personal_score": self._calculate_personal_compatibility(profile_a, profile_b),
            "shared_interests": self._find_shared_interests(profile_a, profile_b),
            "complementary_skills": self._find_complementary_skills(profile_a, profile_b),
            "conversation_starters": self._generate_conversation_starters(profile_a, profile_b),
            "mutual_value": self._assess_mutual_value(profile_a, profile_b),
            "confidence": 0.85
        }
        
        # Calculate overall score
        match_analysis["overall_score"] = (
            match_analysis["professional_score"] * 0.7 + 
            match_analysis["personal_score"] * 0.3
        )
        
        # Determine recommendation
        if match_analysis["overall_score"] >= 70:
            match_analysis["recommendation"] = "strong_match"
        elif match_analysis["overall_score"] >= 50:
            match_analysis["recommendation"] = "good_match"
        else:
            match_analysis["recommendation"] = "weak_match"
        
        # Simple LLM analysis for good matches
        match_analysis["llm_analysis"] = f"Score: {match_analysis['overall_score']}/100 - {match_analysis['recommendation']}"
        
        # Cache the result
        self.match_cache[cache_key] = match_analysis
        
        self.current_state = match_analysis
        self.collaboration_hub.update_agent_state(self.agent_name, match_analysis)
        
        return match_analysis

    def _calculate_professional_compatibility(self, profile_a: Dict, profile_b: Dict) -> float:
        """Calculate professional compatibility score"""
        score = 0.0
        
        # Industry alignment
        industry_a = profile_a.get('professional_summary', {}).get('industry', '')
        industry_b = profile_b.get('professional_summary', {}).get('industry', '')
        if industry_a == industry_b:
            score += 30
        elif industry_a and industry_b:
            score += 15
        
        # Topic overlap
        topics_a = set(profile_a.get('key_topics', []))
        topics_b = set(profile_b.get('key_topics', []))
        topic_overlap = len(topics_a.intersection(topics_b))
        score += min(topic_overlap * 10, 40)
        
        # Goal compatibility
        goals_a = set(profile_a.get('networking_goals', []))
        goals_b = set(profile_b.get('networking_goals', []))
        if goals_a.intersection(goals_b):
            score += 20
        
        return min(score, 100)
    
    def _calculate_personal_compatibility(self, profile_a: Dict, profile_b: Dict) -> float:
        """Calculate personal interest compatibility"""
        interests_a = set(profile_a.get('personal_interests', []))
        interests_b = set(profile_b.get('personal_interests', []))
        
        if not interests_a or not interests_b:
            return 50  # Neutral score if no data
        
        overlap = len(interests_a.intersection(interests_b))
        total_unique = len(interests_a.union(interests_b))
        
        if total_unique == 0:
            return 50
        
        similarity_ratio = overlap / total_unique
        return min(similarity_ratio * 100, 100)
    
    def _find_shared_interests(self, profile_a: Dict, profile_b: Dict) -> List[str]:
        """Find shared interests between users"""
        interests_a = set(profile_a.get('personal_interests', []))
        interests_b = set(profile_b.get('personal_interests', []))
        topics_a = set(profile_a.get('key_topics', []))
        topics_b = set(profile_b.get('key_topics', []))
        
        shared = list(interests_a.intersection(interests_b))
        shared.extend(list(topics_a.intersection(topics_b)))
        
        return list(set(shared))  # Remove duplicates
    
    def _find_complementary_skills(self, profile_a: Dict, profile_b: Dict) -> Dict:
        """Identify complementary skills and expertise"""
        role_a = profile_a.get('professional_summary', {}).get('role_type', '')
        role_b = profile_b.get('professional_summary', {}).get('role_type', '')
        
        complementary_pairs = {
            ("product", "engineering"): "Product-Engineering collaboration",
            ("business", "engineering"): "Business-Technical partnership", 
            ("leadership", "product"): "Strategic-Execution alignment",
            ("sales", "product"): "Market-Product development"
        }
        
        for (role1, role2), description in complementary_pairs.items():
            if (role_a == role1 and role_b == role2) or (role_a == role2 and role_b == role1):
                return {
                    "type": description,
                    "user_a_offers": f"{role_a} expertise",
                    "user_b_offers": f"{role_b} expertise"
                }
        
        return {"type": "general_collaboration", "user_a_offers": "experience", "user_b_offers": "perspective"}
    
    def _generate_conversation_starters(self, profile_a: Dict, profile_b: Dict) -> List[str]:
        """Generate natural conversation starter topics"""
        starters = []
        
        # Shared topics
        shared = self._find_shared_interests(profile_a, profile_b)
        for topic in shared[:3]:  # Top 3
            starters.append(f"Shared interest in {topic}")
        
        # Industry trends
        industry_a = profile_a.get('professional_summary', {}).get('industry', '')
        if industry_a:
            starters.append(f"Trends in {industry_a}")
        
        # Complementary expertise
        comp_skills = self._find_complementary_skills(profile_a, profile_b)
        if comp_skills.get("type") != "general_collaboration":
            starters.append(comp_skills["type"])
        
        return starters[:4]  # Limit to 4 starters
    
    def _assess_mutual_value(self, profile_a: Dict, profile_b: Dict) -> Dict:
        """Assess potential mutual value of introduction"""
        comp_skills = self._find_complementary_skills(profile_a, profile_b)
        
        return {
            "value_to_a": f"Could benefit from {comp_skills['user_b_offers']}",
            "value_to_b": f"Could benefit from {comp_skills['user_a_offers']}",
            "mutual_opportunities": ["knowledge_sharing", "potential_collaboration"],
            "networking_value": "medium_to_high"
        }

class ProximityValidatorAgent(WeavAgent):
    def __init__(self, collaboration_hub: WeavCollaborationHub):
        super().__init__("proximity_validator_agent", collaboration_hub)

    async def validate_proximity(self, user_a: UserProfile, user_b: UserProfile, context: ConferenceContext) -> Dict:
        """Validate that users are in proximity and appropriate location"""
        await self.process_messages()
        
        proximity_analysis = {
            "user_a_id": user_a.user_id,
            "user_b_id": user_b.user_id,
            "distance_meters": self._calculate_distance(user_a.location, user_b.location),
            "same_zone": user_a.location.get("zone") == user_b.location.get("zone"),
            "zone_type": user_a.location.get("zone"),
            "introduction_appropriate": False,
            "environmental_assessment": {},
            "confidence": 0.0
        }
        
        # Distance check
        distance = proximity_analysis["distance_meters"]
        if distance <= 50:
            proximity_status = "very_close"
            proximity_analysis["introduction_appropriate"] = True
        elif distance <= 100:
            proximity_status = "close"
            proximity_analysis["introduction_appropriate"] = True
        elif distance <= 200:
            proximity_status = "nearby"
            proximity_analysis["introduction_appropriate"] = True
        else:
            proximity_status = "too_far"
            proximity_analysis["introduction_appropriate"] = False
        
        proximity_analysis["proximity_status"] = proximity_status
        
        # Zone appropriateness check
        zone = user_a.location.get("zone", "unknown")
        appropriate_zones = ["coffee_area", "networking_lounge", "expo_hall", "break_area"]
        inappropriate_zones = ["presentation_room", "quiet_study", "bathroom", "speaker_green_room"]
        
        if zone in appropriate_zones:
            zone_appropriate = True
        elif zone in inappropriate_zones:
            zone_appropriate = False
        else:
            zone_appropriate = True  # Default to appropriate for unknown zones
        
        # Environmental assessment
        proximity_analysis["environmental_assessment"] = {
            "zone_appropriate": zone_appropriate,
            "crowd_density": context.crowd_density,
            "noise_level": context.noise_level,
            "social_context": self._assess_social_context(zone, context)
        }
        
        # Overall appropriateness
        proximity_analysis["introduction_appropriate"] = (
            proximity_analysis["introduction_appropriate"] and 
            zone_appropriate and
            context.social_appropriateness == "high"
        )
        
        # Calculate confidence
        confidence_factors = []
        if distance <= 100:
            confidence_factors.append(0.4)
        if proximity_analysis["same_zone"]:
            confidence_factors.append(0.3)
        if zone_appropriate:
            confidence_factors.append(0.3)
        
        proximity_analysis["confidence"] = sum(confidence_factors)
        
        self.current_state = proximity_analysis
        self.collaboration_hub.update_agent_state(self.agent_name, proximity_analysis)
        
        return proximity_analysis

    def _calculate_distance(self, location_a: Dict, location_b: Dict) -> float:
        """Calculate distance between two GPS coordinates in meters"""
        lat1, lon1 = location_a.get("lat", 0), location_a.get("lng", 0)
        lat2, lon2 = location_b.get("lat", 0), location_b.get("lng", 0)
        
        # Simple distance calculation (good enough for conference venues)
        lat_diff = (lat2 - lat1) * 111000  # Rough meters per degree latitude
        lng_diff = (lon2 - lon1) * 111000 * math.cos(math.radians((lat1 + lat2) / 2))
        
        return math.sqrt(lat_diff**2 + lng_diff**2)
    
    def _assess_social_context(self, zone: str, context: ConferenceContext) -> str:
        """Assess if the social context is appropriate for introductions"""
        if zone in ["coffee_area", "networking_lounge"] and context.current_session.endswith("break"):
            return "ideal_networking"
        elif zone == "expo_hall":
            return "active_networking"
        elif zone in ["main_hall", "auditorium"] and not context.current_session.endswith("break"):
            return "presentation_mode"
        else:
            return "neutral"

class TimingAgent(WeavAgent):
    def __init__(self, collaboration_hub: WeavCollaborationHub):
        super().__init__("timing_agent", collaboration_hub)

    async def assess_timing(self, user_a: UserProfile, user_b: UserProfile, context: ConferenceContext) -> Dict:
        """Assess if timing is optimal for introduction"""
        await self.process_messages()
        
        timing_analysis = {
            "user_a_id": user_a.user_id,
            "user_b_id": user_b.user_id,
            "schedule_alignment": self._check_schedule_alignment(user_a, user_b),
            "behavioral_timing": self._assess_behavioral_timing(user_a, user_b),
            "interaction_frequency": self._check_interaction_frequency(user_a, user_b),
            "time_window": self._calculate_time_window(user_a, user_b),
            "overall_timing": "unknown",
            "confidence": 0.0
        }
        
        # Schedule compatibility
        schedule_score = 0
        if timing_analysis["schedule_alignment"]["both_available"]:
            schedule_score = 0.4
        
        # Behavioral readiness
        behavioral_score = 0
        if timing_analysis["behavioral_timing"]["both_receptive"]:
            behavioral_score = 0.3
        
        # Frequency appropriateness
        frequency_score = 0
        if timing_analysis["interaction_frequency"]["spacing_appropriate"]:
            frequency_score = 0.2
        
        # Time window sufficiency
        window_score = 0
        if timing_analysis["time_window"]["sufficient_time"]:
            window_score = 0.1
        
        total_score = schedule_score + behavioral_score + frequency_score + window_score
        
        # Determine overall timing
        if total_score >= 0.8:
            timing_analysis["overall_timing"] = "optimal"
        elif total_score >= 0.6:
            timing_analysis["overall_timing"] = "good"
        elif total_score >= 0.4:
            timing_analysis["overall_timing"] = "fair"
        else:
            timing_analysis["overall_timing"] = "poor"
        
        timing_analysis["confidence"] = total_score
        
        self.current_state = timing_analysis
        self.collaboration_hub.update_agent_state(self.agent_name, timing_analysis)
        
        return timing_analysis

    def _check_schedule_alignment(self, user_a: UserProfile, user_b: UserProfile) -> Dict:
        """Check if both users' schedules align for introduction"""
        schedule_a = user_a.schedule
        schedule_b = user_b.schedule
        
        both_in_break = (
            schedule_a.get("current_session", "").endswith("break") and
            schedule_b.get("current_session", "").endswith("break")
        )
        
        time_until_next_a = schedule_a.get("time_until_next", 0)
        time_until_next_b = schedule_b.get("time_until_next", 0)
        
        return {
            "both_available": both_in_break,
            "user_a_session": schedule_a.get("current_session"),
            "user_b_session": schedule_b.get("current_session"),
            "time_remaining_a": time_until_next_a,
            "time_remaining_b": time_until_next_b,
            "minimum_time_available": min(time_until_next_a, time_until_next_b)
        }
    
    def _assess_behavioral_timing(self, user_a: UserProfile, user_b: UserProfile) -> Dict:
        """Assess behavioral readiness for introduction"""
        def assess_user_readiness(user):
            phone = user.phone_behavior
            interaction = user.interaction_history
            
            # Positive indicators
            receptive_actions = ["scrolling_feed", "browsing_mode", "just_closed_slack"]
            receptive_apps = ["linkedin", "instagram", "twitter"]
            
            readiness_score = 0
            indicators = []
            
            if phone.get("last_action") in receptive_actions:
                readiness_score += 0.3
                indicators.append("receptive_activity")
            
            if phone.get("last_app") in receptive_apps:
                readiness_score += 0.2
                indicators.append("social_apps")
            
            if not phone.get("typing_active", True):
                readiness_score += 0.2
                indicators.append("not_busy_typing")
            
            if interaction.get("energy_level") in ["medium", "high"]:
                readiness_score += 0.3
                indicators.append("good_energy")
            
            return {
                "readiness_score": readiness_score,
                "indicators": indicators,
                "ready": readiness_score >= 0.5
            }
        
        readiness_a = assess_user_readiness(user_a)
        readiness_b = assess_user_readiness(user_b)
        
        return {
            "user_a_readiness": readiness_a,
            "user_b_readiness": readiness_b,
            "both_receptive": readiness_a["ready"] and readiness_b["ready"],
            "combined_readiness_score": (readiness_a["readiness_score"] + readiness_b["readiness_score"]) / 2
        }
    
    def _check_interaction_frequency(self, user_a: UserProfile, user_b: UserProfile) -> Dict:
        """Check if introduction frequency is appropriate"""
        history_a = user_a.interaction_history
        history_b = user_b.interaction_history
        
        last_intro_a = history_a.get("last_intro", 60)  # Minutes since last intro
        last_intro_b = history_b.get("last_intro", 60)
        
        total_intros_a = history_a.get("total_intros_today", 0)
        total_intros_b = history_b.get("total_intros_today", 0)
        
        # Good spacing is 30+ minutes since last intro
        good_spacing = last_intro_a >= 30 and last_intro_b >= 30
        
        # Not overwhelmed (less than 5 intros today)
        not_overwhelmed = total_intros_a < 5 and total_intros_b < 5
        
        return {
            "last_intro_a_minutes": last_intro_a,
            "last_intro_b_minutes": last_intro_b,
            "total_intros_today_a": total_intros_a,
            "total_intros_today_b": total_intros_b,
            "spacing_appropriate": good_spacing,
            "volume_appropriate": not_overwhelmed,
            "frequency_ok": good_spacing and not_overwhelmed
        }
    
    def _calculate_time_window(self, user_a: UserProfile, user_b: UserProfile) -> Dict:
        """Calculate available time window for introduction"""
        time_a = user_a.schedule.get("time_until_next", 0)
        time_b = user_b.schedule.get("time_until_next", 0)
        
        available_time = min(time_a, time_b)
        
        return {
            "available_minutes": available_time,
            "sufficient_time": available_time >= 10,  # Need at least 10 minutes
            "time_pressure": "low" if available_time >= 20 else "medium" if available_time >= 10 else "high"
        }

class ContextBuilderAgent(WeavAgent):
    def __init__(self, collaboration_hub: WeavCollaborationHub):
        super().__init__("context_builder_agent", collaboration_hub)

    async def build_introduction_context(self, profile_a: Dict, profile_b: Dict, 
                                       match_analysis: Dict, proximity_data: Dict, 
                                       timing_data: Dict, conference_context: ConferenceContext) -> Dict:
        """Build compelling context for the introduction"""
        await self.process_messages()
        
        try:
            context_data = {
                "user_a_id": profile_a.get("user_id"),
                "user_b_id": profile_b.get("user_id"),
                "value_propositions": self._build_value_propositions(profile_a, profile_b, match_analysis),
                "situational_context": self._build_situational_context(proximity_data, timing_data, conference_context),
                "conversation_opportunities": self._identify_conversation_opportunities(match_analysis, conference_context),
                "suggested_next_steps": self._suggest_next_steps(profile_a, profile_b, match_analysis),
                "introduction_hook": self._create_introduction_hook(match_analysis, proximity_data),
                "confidence": 0.88
            }
            
            self.current_state = context_data
            self.collaboration_hub.update_agent_state(self.agent_name, context_data)
            
            return context_data
            
        except Exception as e:
            return {"error": str(e), "confidence": 0.0}

    def _build_value_propositions(self, profile_a: Dict, profile_b: Dict, match_analysis: Dict) -> Dict:
        """Build specific value propositions for each user"""
        comp_skills = match_analysis.get("complementary_skills", {})
        shared_interests = match_analysis.get("shared_interests", [])
        
        # What B offers to A
        value_to_a = []
        if comp_skills.get("user_b_offers"):
            value_to_a.append(f"Expertise in {comp_skills['user_b_offers']}")
        
        role_b = profile_b.get('professional_summary', {}).get('role_type', '')
        if role_b == "leadership" and profile_a.get('professional_summary', {}).get('role_type') != "leadership":
            value_to_a.append("Leadership and strategic perspective")
        
        # What A offers to B  
        value_to_b = []
        if comp_skills.get("user_a_offers"):
            value_to_b.append(f"Expertise in {comp_skills['user_a_offers']}")
        
        role_a = profile_a.get('professional_summary', {}).get('role_type', '')
        if role_a == "leadership" and profile_b.get('professional_summary', {}).get('role_type') != "leadership":
            value_to_b.append("Leadership and strategic perspective")
        
        return {
            "for_user_a": value_to_a,
            "for_user_b": value_to_b,
            "mutual_benefits": shared_interests,
            "collaboration_potential": comp_skills.get("type", "general collaboration")
        }
    
    def _build_situational_context(self, proximity_data: Dict, timing_data: Dict, conference_context: ConferenceContext) -> Dict:
        """Build context about the current situation"""
        return {
            "location_context": f"Both in {proximity_data.get('zone_type', 'conference area')}",
            "timing_context": f"{timing_data.get('time_window', {}).get('available_minutes', 'limited')} minutes available",
            "conference_context": f"At {conference_context.name}",
            "session_context": conference_context.current_session,
            "environmental_note": f"Good conditions for networking ({proximity_data.get('environmental_assessment', {}).get('social_context', 'appropriate')})"
        }
    
    def _identify_conversation_opportunities(self, match_analysis: Dict, conference_context: ConferenceContext) -> List[str]:
        """Identify specific conversation opportunities"""
        opportunities = []
        
        # From shared interests
        shared = match_analysis.get("shared_interests", [])
        for interest in shared[:2]:  # Top 2
            opportunities.append(f"Discuss {interest} applications")
        
        # From conference context
        opportunities.append(f"Compare experiences at {conference_context.name}")
        
        # From complementary skills
        comp_skills = match_analysis.get("complementary_skills", {})
        if comp_skills.get("type") != "general collaboration":
            opportunities.append(f"Explore {comp_skills['type']} opportunities")
        
        return opportunities[:4]  # Limit to 4
    
    def _suggest_next_steps(self, profile_a: Dict, profile_b: Dict, match_analysis: Dict) -> List[str]:
        """Suggest specific next steps they could take"""
        next_steps = []
        
        # Based on their roles and goals
        goals_a = profile_a.get('networking_goals', [])
        goals_b = profile_b.get('networking_goals', [])
        
        if "collaboration" in goals_a or "collaboration" in goals_b:
            next_steps.append("Explore potential collaboration opportunities")
        
        if "learning" in str(goals_a) or "learning" in str(goals_b):
            next_steps.append("Share insights and best practices")
        
        # Generic valuable next steps
        next_steps.extend([
            "Exchange contact information",
            "Schedule a follow-up coffee meeting",
            "Connect on LinkedIn"
        ])
        
        return next_steps[:3]  # Top 3 most relevant
    
    def _create_introduction_hook(self, match_analysis: Dict, proximity_data: Dict) -> str:
        """Create a compelling hook for the introduction"""
        shared_interests = match_analysis.get("shared_interests", [])
        location = proximity_data.get("zone_type", "here")
        
        if shared_interests:
            primary_interest = shared_interests[0]
            return f"Both working on {primary_interest} and {location.replace('_', ' ')}"
        else:
            comp_skills = match_analysis.get("complementary_skills", {})
            if comp_skills.get("type") != "general collaboration":
                return f"Complementary expertise in {comp_skills['type']} and both {location.replace('_', ' ')}"
            else:
                return f"Similar professional interests and both {location.replace('_', ' ')}"

class IntroWriterAgent(WeavAgent):
    def __init__(self, collaboration_hub: WeavCollaborationHub):
        super().__init__("intro_writer_agent", collaboration_hub)

    async def write_introduction_messages(self, profile_a: Dict, profile_b: Dict, 
                                        context_data: Dict) -> Dict:
        """Write personalized introduction messages for both users"""
        await self.process_messages()
        
        try:
            # Simple message generation without LLM for speed
            value_props = context_data.get("value_propositions", {})
            situational = context_data.get("situational_context", {})
            
            name_a = profile_a.get("user_id", "User A")
            name_b = profile_b.get("user_id", "User B")
            
            title_a = profile_a.get('professional_summary', {}).get('title', 'Professional')
            title_b = profile_b.get('professional_summary', {}).get('title', 'Professional')
            
            location = situational.get('location_context', 'nearby')
            timing = situational.get('timing_context', 'now')
            
            # Generate simple but effective messages
            message_to_a = f"Hi! I noticed you're both {location} with {timing}. Meet {name_b}, a {title_b} with complementary expertise. Perfect for a quick networking chat!"
            
            message_to_b = f"Hi! I noticed you're both {location} with {timing}. Meet {name_a}, a {title_a} with relevant experience. Great opportunity to connect!"
            
            intro_messages = {
                "user_a_id": profile_a.get("user_id"),
                "user_b_id": profile_b.get("user_id"),
                "message_to_a": message_to_a,
                "message_to_b": message_to_b,
                "message_style": "professional_friendly",
                "confidence": 0.91
            }
            
            self.current_state = intro_messages
            self.collaboration_hub.update_agent_state(self.agent_name, intro_messages)
            
            return intro_messages
            
        except Exception as e:
            return {"error": str(e), "confidence": 0.0}

# -------------------------------------
# Main Weav Orchestrator
# -------------------------------------
class WeavOrchestrator:
    def __init__(self):
        self.collaboration_hub = WeavCollaborationHub()
        
        # Initialize agents based on flags
        self.profile_summarizer = ProfileSummarizerAgent(self.collaboration_hub) if ENABLE_PROFILE_SUMMARIZER_AGENT else None
        self.context_monitor = ContextMonitorAgent(self.collaboration_hub) if ENABLE_CONTEXT_MONITOR_AGENT else None
        self.interest_matcher = InterestMatcherAgent(self.collaboration_hub) if ENABLE_INTEREST_MATCHER_AGENT else None
        self.proximity_validator = ProximityValidatorAgent(self.collaboration_hub) if ENABLE_PROXIMITY_VALIDATOR_AGENT else None
        self.timing_agent = TimingAgent(self.collaboration_hub) if ENABLE_TIMING_AGENT else None
        self.context_builder = ContextBuilderAgent(self.collaboration_hub) if ENABLE_CONTEXT_BUILDER_AGENT else None
        self.intro_writer = IntroWriterAgent(self.collaboration_hub) if ENABLE_INTRO_WRITER_AGENT else None
        
        self.introduction_database = IntroductionDatabase()

    async def process_introduction_opportunity(self, users: List[UserProfile], conference_context: ConferenceContext) -> Dict:
        """Process a potential introduction opportunity"""
        
        # Suppress verbose output - only show final results
        import sys
        from io import StringIO
        
        # Capture stdout to suppress agent chatter
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        try:
            results = {
                "opportunity_id": f"intro_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                "users": [user.user_id for user in users],
                "agent_results": {},
                "final_decision": None,
                "introduction_sent": False
            }
            
            # Clear previous states
            self.collaboration_hub.active_agents = []
            self.collaboration_hub.message_queue = []
            self.collaboration_hub.agent_states = {}
            
            # Phase 1: Profile Analysis
            if self.profile_summarizer:
                profile_results = {}
                for user in users:
                    profile_analysis = await self.profile_summarizer.process_profile(user)
                    profile_results[user.user_id] = profile_analysis
                results["agent_results"]["profile_analysis"] = profile_results
                self.collaboration_hub.active_agents.append("profile_summarizer")
            
            # Phase 2: Context Monitoring
            if self.context_monitor:
                context_analysis = await self.context_monitor.monitor_user_context(users, conference_context)
                results["agent_results"]["context_monitoring"] = context_analysis
                self.collaboration_hub.active_agents.append("context_monitor")
            
            # Early exit if users not available
            if context_analysis and not context_analysis.get("opportunity_signals"):
                results["final_decision"] = {"proceed": False, "reason": "users_not_available"}
                return results
            
            # Phase 3: Pairwise Analysis
            if len(users) >= 2:
                user_a, user_b = users[0], users[1]
                profile_a = profile_results.get(user_a.user_id, {})
                profile_b = profile_results.get(user_b.user_id, {})
                
                # Interest Matching
                if self.interest_matcher:
                    match_analysis = await self.interest_matcher.calculate_match_score(profile_a, profile_b)
                    results["agent_results"]["interest_matching"] = match_analysis
                    self.collaboration_hub.active_agents.append("interest_matcher")
                    
                    # Early exit if poor match
                    if match_analysis.get("overall_score", 0) < 30:  # Reasonable threshold
                        results["final_decision"] = {"proceed": False, "reason": "poor_interest_match"}
                        return results
                
                # Proximity Validation
                if self.proximity_validator:
                    proximity_analysis = await self.proximity_validator.validate_proximity(user_a, user_b, conference_context)
                    results["agent_results"]["proximity_validation"] = proximity_analysis
                    self.collaboration_hub.active_agents.append("proximity_validator")
                    
                    # Early exit if too far or inappropriate location
                    if not proximity_analysis.get("introduction_appropriate"):
                        results["final_decision"] = {"proceed": False, "reason": "proximity_inappropriate"}
                        return results
                
                # Timing Assessment
                if self.timing_agent:
                    timing_analysis = await self.timing_agent.assess_timing(user_a, user_b, conference_context)
                    results["agent_results"]["timing_assessment"] = timing_analysis
                    self.collaboration_hub.active_agents.append("timing_agent")
                    
                    # Early exit if poor timing
                    if timing_analysis.get("overall_timing") == "poor":
                        results["final_decision"] = {"proceed": False, "reason": "poor_timing"}
                        return results
                
                # Context Building
                if self.context_builder:
                    context_data = await self.context_builder.build_introduction_context(
                        profile_a, profile_b, match_analysis, proximity_analysis, timing_analysis, conference_context
                    )
                    results["agent_results"]["context_building"] = context_data
                    self.collaboration_hub.active_agents.append("context_builder")
                
                # Introduction Writing
                if self.intro_writer:
                    intro_messages = await self.intro_writer.write_introduction_messages(
                        profile_a, profile_b, context_data
                    )
                    results["agent_results"]["intro_writing"] = intro_messages
                    self.collaboration_hub.active_agents.append("intro_writer")
                
                # Final Decision
                all_agents_success = all(
                    result.get("confidence", 0) > 0.5 
                    for result in results["agent_results"].values()
                    if isinstance(result, dict) and "confidence" in result
                )
                
                if all_agents_success:
                    # Log the introduction
                    intro_record = {
                        "opportunity_id": results["opportunity_id"],
                        "user_a": user_a.user_id,
                        "user_b": user_b.user_id,
                        "match_score": match_analysis.get("overall_score", 0),
                        "message_to_a": intro_messages.get("message_to_a"),
                        "message_to_b": intro_messages.get("message_to_b"),
                        "context": conference_context.name,
                        "timestamp": datetime.utcnow().isoformat() + "Z"
                    }
                    
                    self.introduction_database.save_introduction(intro_record)
                    
                    results["final_decision"] = {
                        "proceed": True,
                        "confidence": 0.9,
                        "reason": "all_agents_approve",
                        "introduction_messages": intro_messages
                    }
                    results["introduction_sent"] = True
                    
                else:
                    results["final_decision"] = {"proceed": False, "reason": "agent_concerns"}
            
            else:
                results["final_decision"] = {"proceed": False, "reason": "insufficient_users"}
            
            # Log the opportunity
            self.collaboration_hub.log_introduction_opportunity(results)
            
            return results
            
        except Exception as e:
            results["final_decision"] = {"proceed": False, "reason": f"error: {str(e)}"}
            return results
        
        finally:
            # Restore stdout
            sys.stdout = old_stdout
            
            # Only print final result
            if results.get("introduction_sent"):
                user_names = [user.name for user in users[:2]]
                print(f"✅ INTRODUCTION SENT: {user_names[0]} ↔ {user_names[1]}")
            else:
                reason = results.get("final_decision", {}).get("reason", "unknown")
                print(f"⚠️ No introduction: {reason}")

    def get_system_stats(self) -> Dict:
        """Get system statistics"""
        return {
            "enabled_agents": [
                agent for agent, enabled in [
                    ("profile_summarizer", ENABLE_PROFILE_SUMMARIZER_AGENT),
                    ("context_monitor", ENABLE_CONTEXT_MONITOR_AGENT),
                    ("interest_matcher", ENABLE_INTEREST_MATCHER_AGENT),
                    ("proximity_validator", ENABLE_PROXIMITY_VALIDATOR_AGENT),
                    ("timing_agent", ENABLE_TIMING_AGENT),
                    ("context_builder", ENABLE_CONTEXT_BUILDER_AGENT),
                    ("intro_writer", ENABLE_INTRO_WRITER_AGENT)
                ] if enabled
            ],
            "introduction_opportunities": len(self.collaboration_hub.introduction_opportunities),
            "total_messages": len(self.collaboration_hub.message_queue),
            "database_stats": self.introduction_database.get_stats()
        }

# -------------------------------------
# Introduction Database
# -------------------------------------
class IntroductionDatabase:
    def __init__(self, db_path="data/weav_introductions.json"):
        self.db_path = db_path
        self.ensure_db_exists()
    
    def ensure_db_exists(self):
        """Ensure database file exists"""
        directory = os.path.dirname(self.db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        if not os.path.exists(self.db_path) or os.path.getsize(self.db_path) == 0:
            with open(self.db_path, 'w') as f:
                json.dump({"introductions": [], "users": {}, "analytics": {}}, f, indent=2)
    
    def save_introduction(self, intro_record: Dict):
        """Save an introduction record"""
        try:
            self.ensure_db_exists()
            
            with open(self.db_path, 'r') as f:
                data = json.load(f)
            
            if "introductions" not in data:
                data["introductions"] = []
            
            data["introductions"].append(intro_record)
            
            # Update user stats
            if "users" not in data:
                data["users"] = {}
            
            for user_id in [intro_record["user_a"], intro_record["user_b"]]:
                if user_id not in data["users"]:
                    data["users"][user_id] = {"total_introductions": 0, "successful_meetings": 0}
                data["users"][user_id]["total_introductions"] += 1
            
            with open(self.db_path, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
        except Exception as e:
            print(f"❌ Error saving introduction: {e}")
    
    def get_user_history(self, user_id: str) -> List[Dict]:
        """Get introduction history for a user"""
        try:
            with open(self.db_path, 'r') as f:
                data = json.load(f)
            
            return [
                intro for intro in data.get("introductions", [])
                if intro.get("user_a") == user_id or intro.get("user_b") == user_id
            ]
        except Exception as e:
            return []
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        try:
            with open(self.db_path, 'r') as f:
                data = json.load(f)
            
            intros = data.get("introductions", [])
            
            return {
                "total_introductions": len(intros),
                "unique_users": len(data.get("users", {})),
                "average_match_score": np.mean([intro.get("match_score", 0) for intro in intros]) if intros else 0,
                "latest_introduction": intros[-1].get("opportunity_id") if intros else None
            }
        except Exception as e:
            return {"error": str(e)}

async def main():
    print("🎯 Weav Agent Orchestration System Ready")

if __name__ == "__main__":
    asyncio.run(main())