#!/usr/bin/env python3
"""
Test script for Weav API endpoints
Run this to verify your API is working correctly
"""

import asyncio
import json
import requests
import websockets
from datetime import datetime

API_BASE = "http://localhost:8000"
WS_BASE = "ws://localhost:8000"

def test_basic_endpoints():
    """Test basic API endpoints"""
    print("🧪 Testing Basic Endpoints...")
    
    # Test root endpoint
    try:
        response = requests.get(f"{API_BASE}/")
        print(f"✅ Root: {response.status_code} - {response.json()['message']}")
    except Exception as e:
        print(f"❌ Root failed: {e}")
    
    # Test health check
    try:
        response = requests.get(f"{API_BASE}/health")
        health_data = response.json()
        print(f"✅ Health: {response.status_code} - {health_data['status']}")
    except Exception as e:
        print(f"❌ Health failed: {e}")
    
    # Test system stats
    try:
        response = requests.get(f"{API_BASE}/api/system/stats")
        stats = response.json()
        print(f"✅ Stats: {response.status_code} - {stats['active_users']} users")
    except Exception as e:
        print(f"❌ Stats failed: {e}")

def test_user_profile_creation():
    """Test user profile creation"""
    print("\n👤 Testing User Profile Creation...")
    
    test_profile = {
        "name": "Test User",
        "title": "AI Engineer",
        "company": "Test Corp",
        "bio": "Testing the Weav system",
        "interests": ["ai", "tech", "networking"],
        "intent": "Networking"
    }
    
    try:
        response = requests.post(
            f"{API_BASE}/api/users/profile",
            json=test_profile,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            user_data = response.json()
            user_id = user_data['user_id']
            print(f"✅ Profile created: {user_id}")
            return user_id
        else:
            print(f"❌ Profile creation failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Profile creation error: {e}")
        return None

def test_context_updates(user_id):
    """Test context update endpoints"""
    print(f"\n📍 Testing Context Updates for {user_id}...")
    
    # Test location update
    try:
        location_data = {
            "lat": 30.2672,
            "lng": -97.7431,
            "zone": "coffee_area"
        }
        response = requests.post(
            f"{API_BASE}/api/users/{user_id}/location",
            json=location_data
        )
        print(f"✅ Location update: {response.status_code}")
    except Exception as e:
        print(f"❌ Location update failed: {e}")
    
    # Test phone behavior update
    try:
        behavior_data = {
            "last_app": "linkedin",
            "screen_time_minutes": 5,
            "app_switches_last_hour": 3,
            "typing_active": False,
            "last_action": "scrolling_feed"
        }
        response = requests.post(
            f"{API_BASE}/api/users/{user_id}/phone-behavior",
            json=behavior_data
        )
        print(f"✅ Phone behavior update: {response.status_code}")
    except Exception as e:
        print(f"❌ Phone behavior update failed: {e}")
    
    # Test schedule update
    try:
        schedule_data = {
            "current_session": "networking_break",
            "next_session": "AI Panel",
            "time_until_next": 20
        }
        response = requests.post(
            f"{API_BASE}/api/users/{user_id}/schedule",
            json=schedule_data
        )
        print(f"✅ Schedule update: {response.status_code}")
    except Exception as e:
        print(f"❌ Schedule update failed: {e}")

async def test_websocket_connection(user_id):
    """Test WebSocket connection"""
    print(f"\n🔗 Testing WebSocket connection for {user_id}...")
    
    try:
        async with websockets.connect(f"{WS_BASE}/ws/{user_id}") as websocket:
            print("✅ WebSocket connected")
            
            # Send heartbeat
            await websocket.send(json.dumps({"type": "heartbeat"}))
            
            # Wait for response (with timeout)
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                message = json.loads(response)
                print(f"✅ Heartbeat response: {message['type']}")
            except asyncio.TimeoutError:
                print("⚠️ No heartbeat response (but connection works)")
            
            print("✅ WebSocket test completed")
            
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")

def test_multi_user_scenario():
    """Test scenario with multiple users to trigger introductions"""
    print("\n👥 Testing Multi-User Introduction Scenario...")
    
    # Create two test users
    users = []
    for i in range(2):
        profile = {
            "name": f"Test User {i+1}",
            "title": "Product Manager" if i == 0 else "Software Engineer",
            "company": f"Company {i+1}",
            "bio": f"Interested in AI and technology, user {i+1}",
            "interests": ["ai", "tech", "networking"],
            "intent": "Networking"
        }
        
        try:
            response = requests.post(f"{API_BASE}/api/users/profile", json=profile)
            if response.status_code == 200:
                user_id = response.json()['user_id']
                users.append(user_id)
                print(f"✅ Created user {i+1}: {user_id}")
                
                # Set location near each other
                location_data = {
                    "lat": 30.2672 + (i * 0.0001),  # Very close
                    "lng": -97.7431 + (i * 0.0001),
                    "zone": "coffee_area"
                }
                requests.post(f"{API_BASE}/api/users/{user_id}/location", json=location_data)
                
                # Set good availability
                behavior_data = {
                    "last_app": "linkedin",
                    "screen_time_minutes": 3,
                    "app_switches_last_hour": 2,
                    "typing_active": False,
                    "last_action": "browsing_mode"
                }
                requests.post(f"{API_BASE}/api/users/{user_id}/phone-behavior", json=behavior_data)
                
        except Exception as e:
            print(f"❌ Failed to create user {i+1}: {e}")
    
    if len(users) == 2:
        print(f"✅ Two users created and positioned for potential introduction")
        print(f"⏳ Wait ~10 seconds for background monitoring to detect opportunity...")
        return users
    else:
        print("❌ Failed to create test users")
        return []

async def main():
    """Run all tests"""
    print("🚀 Weav API Test Suite")
    print("=" * 50)
    
    # Basic endpoint tests
    test_basic_endpoints()
    
    # User profile test
    user_id = test_user_profile_creation()
    
    if user_id:
        # Context update tests
        test_context_updates(user_id)
        
        # WebSocket test
        await test_websocket_connection(user_id)
    
    # Multi-user scenario
    test_users = test_multi_user_scenario()
    
    print("\n📊 Test Summary:")
    print("- Check http://localhost:8000/docs for interactive API docs")
    print("- Check http://localhost:8000/api/system/stats for system status")
    print("- Monitor your server logs for introduction opportunities")
    
    if test_users:
        print(f"- Created test users: {test_users}")
        print("- Watch server logs for potential introductions!")

if __name__ == "__main__":
    asyncio.run(main())