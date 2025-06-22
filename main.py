from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import asyncio
import json
import os
from datetime import datetime
import logging

# Import your existing Weav system
from orchestration import WeavOrchestrator, UserProfile, ConferenceContext

app = FastAPI(title="Weav API", version="1.0.0")

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://localhost:8080",  # Alternative ports
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Global state
orchestrator = WeavOrchestrator()
active_connections: Dict[str, WebSocket] = {}
user_contexts: Dict[str, Dict] = {}

# JSON Database for persistence
DB_FILE = "weav_demo_db.json"

def load_database():
    """Load database from JSON file"""
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {
        "users": {},
        "introductions": [],
        "user_contexts": {},
        "introduction_history": {}  # Track who has been introduced to whom
    }

def save_database(data):
    """Save database to JSON file"""
    try:
        with open(DB_FILE, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as e:
        print(f"Error saving database: {e}")

# Load initial data
db_data = load_database()
user_contexts = db_data.get("user_contexts", {})

# Pydantic models for API
class UserProfileCreate(BaseModel):
    name: str
    title: str
    company: str
    bio: str
    linkedin: Optional[str] = None
    interests: List[str]
    intent: str

class LocationUpdate(BaseModel):
    lat: float
    lng: float
    zone: str

class PhoneBehaviorUpdate(BaseModel):
    last_app: str
    screen_time_minutes: int
    app_switches_last_hour: int
    typing_active: bool
    last_action: str

class ScheduleUpdate(BaseModel):
    current_session: str
    next_session: str
    time_until_next: int

class IntroductionResponse(BaseModel):
    intro_id: str
    action: str  # "accept" or "dismiss"

# ============================================================================
# Basic Routes
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Weav AI Networking API is running!",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "stats": "/api/system/stats",
            "database": "/api/debug/database"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "active_users": len(user_contexts),
        "active_connections": len(active_connections)
    }

@app.options("/api/users/profile")
async def options_create_user_profile():
    """Handle CORS preflight for profile creation"""
    return {"status": "ok"}

# ============================================================================
# User Profile Management
# ============================================================================

@app.post("/api/users/profile")
async def create_user_profile(profile_data: UserProfileCreate):
    """Create user profile from onboarding"""
    try:
        print(f"📝 Creating profile for: {profile_data.name}")
        
        # Generate user ID (in production, use proper auth)
        db_data = load_database()
        user_id = f"user_{len(db_data['users']) + 1}"
        
        # Save to persistent database
        db_data["users"][user_id] = {
            **profile_data.model_dump(),
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Initialize user context
        user_contexts[user_id] = {
            "profile": profile_data.model_dump(),
            "location": {"lat": 0, "lng": 0, "zone": "unknown"},
            "phone_behavior": {
                "last_app": "weav",
                "screen_time_minutes": 0,
                "app_switches_last_hour": 0,
                "typing_active": False,
                "last_action": "app_opened"
            },
            "schedule": {
                "current_session": "networking_break",
                "next_session": "unknown",
                "time_until_next": 30
            },
            "interaction_history": {
                "last_intro": 120,  # Start with longer gap to avoid immediate matching
                "total_intros_today": 0,
                "energy_level": "high"
            },
            "last_opportunity_check": datetime.utcnow().isoformat()
        }
        
        db_data["user_contexts"] = user_contexts
        save_database(db_data)
        
        print(f"✅ Created user profile with ID: {user_id}")
        return {"user_id": user_id, "status": "created"}
        
    except Exception as e:
        print(f"❌ Error creating profile: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/users/{user_id}/profile")
async def get_user_profile(user_id: str):
    """Get user profile"""
    if user_id not in user_contexts:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user_contexts[user_id]["profile"]

# ============================================================================
# Real-time Context Updates
# ============================================================================

@app.post("/api/users/{user_id}/location")
async def update_location(user_id: str, location: LocationUpdate):
    """Update user location"""
    if user_id not in user_contexts:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_contexts[user_id]["location"] = location.model_dump()
    
    # Only trigger proximity checking occasionally to avoid spam
    await maybe_check_introduction_opportunities(user_id)
    
    return {"status": "updated"}

@app.post("/api/users/{user_id}/phone-behavior")
async def update_phone_behavior(user_id: str, behavior: PhoneBehaviorUpdate):
    """Update phone behavior for context monitoring"""
    if user_id not in user_contexts:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_contexts[user_id]["phone_behavior"] = behavior.model_dump()
    return {"status": "updated"}

@app.post("/api/users/{user_id}/schedule")
async def update_schedule(user_id: str, schedule: ScheduleUpdate):
    """Update user schedule"""
    if user_id not in user_contexts:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_contexts[user_id]["schedule"] = schedule.model_dump()
    return {"status": "updated"}

@app.get("/api/users/{user_id}/context")
async def get_user_context(user_id: str):
    """Get current user context for debugging"""
    if user_id not in user_contexts:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user_contexts[user_id]

# ============================================================================
# WebSocket for Real-time Introductions
# ============================================================================

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await websocket.accept()
    active_connections[user_id] = websocket
    print(f"🔗 User {user_id} connected via WebSocket")
    
    try:
        while True:
            # Keep connection alive and listen for client messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message.get("type") == "heartbeat":
                await websocket.send_text(json.dumps({"type": "heartbeat_ack"}))
            elif message.get("type") == "context_update":
                # Update user context from frontend
                await update_user_context_from_frontend(user_id, message.get("data", {}))
                
    except WebSocketDisconnect:
        if user_id in active_connections:
            del active_connections[user_id]
        print(f"🔌 User {user_id} disconnected")

async def send_introduction_to_user(user_id: str, introduction_data: Dict):
    """Send introduction to user via WebSocket"""
    if user_id in active_connections:
        try:
            await active_connections[user_id].send_text(json.dumps({
                "type": "introduction",
                "data": introduction_data
            }))
            print(f"📱 Sent introduction to {user_id}")
        except Exception as e:
            print(f"❌ Error sending introduction to {user_id}: {e}")

# ============================================================================
# Introduction Logic - FIXED
# ============================================================================

async def maybe_check_introduction_opportunities(triggering_user_id: str):
    """Check for introduction opportunities with rate limiting"""
    try:
        # Rate limiting - only check every 2 minutes per user
        last_check_str = user_contexts[triggering_user_id].get("last_opportunity_check")
        if last_check_str:
            last_check = datetime.fromisoformat(last_check_str.replace('Z', ''))
            if (datetime.utcnow() - last_check).total_seconds() < 120:  # 2 minutes
                return
        
        user_contexts[triggering_user_id]["last_opportunity_check"] = datetime.utcnow().isoformat()
        
        # Find nearby users (EXCLUDING SELF)
        nearby_users = []
        triggering_context = user_contexts.get(triggering_user_id)
        
        if not triggering_context:
            return
        
        for user_id, context in user_contexts.items():
            if user_id == triggering_user_id:  # Skip self!
                continue
                
            # Simple proximity check
            distance = calculate_simple_distance(
                triggering_context["location"], 
                context["location"]
            )
            
            if distance < 200:  # 200 meters
                # Check if already introduced recently
                if not recently_introduced(triggering_user_id, user_id):
                    nearby_users.append(user_id)
        
        if not nearby_users:
            print(f"🔍 No nearby users found for {triggering_user_id}")
            return
        
        print(f"🎯 Found {len(nearby_users)} potential matches for {triggering_user_id}")
        
        # Take just the first nearby user to avoid overwhelming
        target_user_id = nearby_users[0]
        
        # Convert to UserProfile objects for orchestrator
        users_for_analysis = []
        for uid in [triggering_user_id, target_user_id]:
            context = user_contexts[uid]
            profile = context["profile"]
            
            user_profile = UserProfile(
                user_id=uid,
                name=profile["name"],
                linkedin_title=profile["title"],
                company=profile["company"],
                bio=profile["bio"],
                work_passion=profile.get("bio", ""),
                interests=profile["interests"],
                location=context["location"],
                phone_behavior=context["phone_behavior"],
                schedule=context["schedule"],
                interaction_history=context["interaction_history"],
                communication_style="professional"
            )
            users_for_analysis.append(user_profile)
        
        # Create conference context
        conference_context = ConferenceContext(
            name="TechConf 2024",
            current_time=datetime.now().strftime("%I:%M %p"),
            current_session="networking_break",
            venue_zones=["main_hall", "coffee_area", "expo_hall"],
            crowd_density="moderate",
            noise_level="conversation_friendly",
            social_appropriateness="high"
        )
        
        print(f"🤖 Running AI agents for {triggering_user_id} + {target_user_id}...")
        
        # Run orchestrator (suppress verbose output)
        result = await orchestrator.process_introduction_opportunity(
            users_for_analysis, conference_context
        )
        
        # If introduction approved, send to both users
        if result.get("introduction_sent"):
            intro_messages = result["final_decision"]["introduction_messages"]
            
            # Save to database
            db_data = load_database()
            intro_record = {
                "id": result["opportunity_id"],
                "user_a": triggering_user_id,
                "user_b": target_user_id,
                "match_score": result.get("agent_results", {}).get("interest_matching", {}).get("overall_score", 0),
                "message_to_a": intro_messages.get("message_to_a", ""),
                "message_to_b": intro_messages.get("message_to_b", ""),
                "timestamp": datetime.utcnow().isoformat(),
                "context": conference_context.name
            }
            
            db_data["introductions"].append(intro_record)
            
            # Mark as introduced
            if "introduction_history" not in db_data:
                db_data["introduction_history"] = {}
            
            intro_key = f"{triggering_user_id}:{target_user_id}"
            db_data["introduction_history"][intro_key] = datetime.utcnow().isoformat()
            intro_key_reverse = f"{target_user_id}:{triggering_user_id}"
            db_data["introduction_history"][intro_key_reverse] = datetime.utcnow().isoformat()
            
            save_database(db_data)
            
            print(f"✅ MATCH MADE! Sending introductions between {triggering_user_id} and {target_user_id}")
            
            # Send to both users
            for i, user_profile in enumerate(users_for_analysis):
                message_key = f"message_to_{'a' if i == 0 else 'b'}"
                other_user = users_for_analysis[1 if i == 0 else 0]
                
                introduction_data = {
                    "id": result["opportunity_id"],
                    "match_user": {
                        "name": other_user.name,
                        "title": other_user.linkedin_title,
                        "company": other_user.company,
                        "location": other_user.location.get("zone", "nearby"),
                        "avatar": "👤"
                    },
                    "intro_message": intro_messages.get(message_key, "Great match found!"),
                    "why": f"Match score: {result.get('agent_results', {}).get('interest_matching', {}).get('overall_score', 0)}/100",
                    "timing_context": "Perfect timing for a quick introduction!"
                }
                
                await send_introduction_to_user(user_profile.user_id, introduction_data)
                
                # Update interaction history
                user_contexts[user_profile.user_id]["interaction_history"]["last_intro"] = 0
                user_contexts[user_profile.user_id]["interaction_history"]["total_intros_today"] += 1
        else:
            reason = result.get("final_decision", {}).get("reason", "unknown")
            print(f"⚠️ No introduction: {reason}")
            
    except Exception as e:
        print(f"❌ Error checking introduction opportunities: {e}")

def recently_introduced(user_a: str, user_b: str) -> bool:
    """Check if two users were recently introduced"""
    db_data = load_database()
    intro_history = db_data.get("introduction_history", {})
    
    # Check both directions
    intro_key = f"{user_a}:{user_b}"
    intro_key_reverse = f"{user_b}:{user_a}"
    
    for key in [intro_key, intro_key_reverse]:
        if key in intro_history:
            intro_time = datetime.fromisoformat(intro_history[key])
            # Don't reintroduce within 24 hours
            if (datetime.utcnow() - intro_time).total_seconds() < 86400:
                return True
    
    return False

def calculate_simple_distance(loc1: Dict, loc2: Dict) -> float:
    """Simple distance calculation"""
    import math
    
    lat1, lng1 = loc1.get("lat", 0), loc1.get("lng", 0)
    lat2, lng2 = loc2.get("lat", 0), loc2.get("lng", 0)
    
    # Simple distance in meters (rough approximation)
    lat_diff = (lat2 - lat1) * 111000
    lng_diff = (lng2 - lng1) * 111000 * math.cos(math.radians((lat1 + lat2) / 2))
    
    return math.sqrt(lat_diff**2 + lng_diff**2)

async def update_user_context_from_frontend(user_id: str, data: Dict):
    """Update user context from frontend data"""
    if user_id in user_contexts:
        context = user_contexts[user_id]
        
        # Update relevant fields
        if "phone_behavior" in data:
            context["phone_behavior"].update(data["phone_behavior"])
        if "location" in data:
            context["location"].update(data["location"])
        if "schedule" in data:
            context["schedule"].update(data["schedule"])

# ============================================================================
# Introduction Response Handling
# ============================================================================

@app.post("/api/introductions/{intro_id}/respond")
async def respond_to_introduction(intro_id: str, response: IntroductionResponse):
    """Handle user response to introduction"""
    db_data = load_database()
    
    # Find the introduction
    intro = None
    for introduction in db_data["introductions"]:
        if introduction["id"] == intro_id:
            intro = introduction
            break
    
    if intro:
        intro["response"] = {
            "action": response.action,
            "timestamp": datetime.utcnow().isoformat()
        }
        save_database(db_data)
        print(f"📝 User responded to introduction {intro_id}: {response.action}")
    
    return {"status": "recorded"}

# ============================================================================
# System Status and Debug Endpoints
# ============================================================================

@app.get("/api/system/stats")
async def get_system_stats():
    """Get system statistics"""
    db_data = load_database()
    return {
        "active_users": len(user_contexts),
        "active_connections": len(active_connections),
        "total_users_in_db": len(db_data.get("users", {})),
        "total_introductions": len(db_data.get("introductions", [])),
        "orchestrator_stats": orchestrator.get_system_stats()
    }

@app.get("/api/debug/database")
async def debug_database():
    """Debug endpoint to see database contents"""
    db_data = load_database()
    return {
        "users": list(db_data.get("users", {}).keys()),
        "introductions_count": len(db_data.get("introductions", [])),
        "recent_introductions": db_data.get("introductions", [])[-5:],  # Last 5
        "active_contexts": list(user_contexts.keys())
    }

@app.get("/api/debug/users")
async def debug_users():
    """Debug endpoint to see all users"""
    return {
        "users": list(user_contexts.keys()),
        "contexts": {uid: {k: v for k, v in ctx.items() if k != "profile"} for uid, ctx in user_contexts.items()}
    }

# ============================================================================
# Database Management
# ============================================================================

@app.post("/api/debug/reset-database")
async def reset_database():
    """Reset the database for testing"""
    global user_contexts
    user_contexts = {}
    
    # Clear active connections
    for ws in active_connections.values():
        try:
            await ws.close()
        except:
            pass
    active_connections.clear()
    
    # Reset file
    initial_data = {
        "users": {},
        "introductions": [],
        "user_contexts": {},
        "introduction_history": {}
    }
    save_database(initial_data)
    
    return {"status": "database_reset", "message": "All data cleared"}

# ============================================================================
# Background Tasks with Lifespan
# ============================================================================

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application lifespan events"""
    # Startup
    print("🚀 Starting Weav API...")
    print("📊 Orchestrator initialized")
    print(f"💾 Database: {DB_FILE}")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down Weav API...")
    # Save final state
    db_data = load_database()
    db_data["user_contexts"] = user_contexts
    save_database(db_data)

app.router.lifespan_context = lifespan

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)