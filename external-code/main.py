import os
# Set OpenMP environment variable to resolve duplicate library error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Set FFmpeg path to ensure soxr resampler is available
os.environ["FFMPEG_BINARY"] = r"C:\Users\josep\scoop\shims\ffmpeg.exe"

import argparse
import time
import sys
import keyboard
import threading
import random
import platform
import subprocess
import langid
import json
import socket
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import httpx
from typing import Optional, List, Dict
from datetime import datetime
import asyncio

# Import local modules and configuration
import config

# Import service modules from their new locations
sys.path.append(os.path.join(os.getcwd(), "services"))

# Service imports
from services import touchdesigner_service
from utils import general_utils

# API client imports
from api_client import llm_client, tts_client, whisper_client

app = FastAPI(title="Spanish AI Assistant - Ollama API")

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

@app.websocket("/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    print("[DEBUG] WebSocket connection established")
    try:
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                action = message.get("action")
                
                if action == "start":
                    print("[DEBUG] Received start action")
                    # Send listening status
                    await manager.send_message(json.dumps({
                        "status": "listening",
                        "timestamp": datetime.now().isoformat()
                    }), websocket)
                    print("[DEBUG] Sent listening status")
                    
                    # Send voice detection status immediately
                    await manager.send_message(json.dumps({
                        "status": "voice_detected",
                        "timestamp": datetime.now().isoformat()
                    }), websocket)
                    print("[DEBUG] Sent voice detection status")
                    
                    # Send transcription text
                    await manager.send_message(json.dumps({
                        "type": "transcription",
                        "text": "¡Hola, esto es una prueba!",
                        "timestamp": datetime.now().isoformat()
                    }), websocket)
                    print("[DEBUG] Sent transcription text")
                    
                elif action == "stop":
                    print("[DEBUG] Received stop action")
                    # Send finished status
                    await manager.send_message(json.dumps({
                        "status": "finished",
                        "timestamp": datetime.now().isoformat()
                    }), websocket)
                    print("[DEBUG] Sent finished status")
                    # Don't break here, keep connection open
                    
            except json.JSONDecodeError:
                print("[ERROR] Invalid JSON message received")
                try:
                    await manager.send_message(json.dumps({
                        "error": "Invalid JSON message"
                    }), websocket)
                except:
                    pass
            except WebSocketDisconnect:
                print("[DEBUG] Client disconnected")
                break
            except Exception as e:
                print(f"[ERROR] Server error: {e}")
                try:
                    await manager.send_message(json.dumps({
                        "error": f"Server error: {str(e)}"
                    }), websocket)
                except:
                    pass
                # Don't break on error, keep connection open
                
    except WebSocketDisconnect:
        print("[DEBUG] WebSocket connection ended")
    except Exception as e:
        print(f"[ERROR] WebSocket error: {e}")
    finally:
        try:
            manager.disconnect(websocket)
            print("[DEBUG] WebSocket connection cleaned up")
        except:
            pass

class ChatRequest(BaseModel):
    model: str
    prompt: str
    stream: bool = False
    context: Optional[List[Dict]] = None

class ChatResponse(BaseModel):
    response: str
    context: Optional[List[Dict]] = None

class LLMService:
    def __init__(self):
        """Initialize the LLM service with necessary components."""
        self.connected = False
        self.last_health_check = None
        
    def ensure_llm_connection(self):
        """
        Check if the LLM service is connected and healthy.
        Returns:
            bool: True if service is healthy, False otherwise
        """
        try:
            # Example health check - replace with actual implementation
            # This could check API connectivity, model loading, etc.
            self.connected = True
            self.last_health_check = datetime.now()
            return True
        except Exception as e:
            self.connected = False
            print(f"LLM health check failed: {e}")
            return False

@app.get("/")
async def root():
    return {"message": "Spanish AI Assistant Ollama API"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://{config.LLM_CHAT_HOST}:{config.LLM_CHAT_PORT}/api/generate",
                json=request.dict()
            )
            response.raise_for_status()
            result = response.json()
            
            return ChatResponse(
                response=result.get("response", ""),
                context=result.get("context")
            )
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"LLM API service error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models")
async def list_models():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"http://{config.LLM_CHAT_HOST}:{config.LLM_CHAT_PORT}/api/tags")
            response.raise_for_status()
            return response.json()
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"LLM API service error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--audio_path", type=str, default=".", help="path to the directory containing reference voice files")
parser.add_argument("--keyboard_shortcut", action='store_true', default=True, help="enable keyboard shortcut (Ctrl+R) to start interaction")
parser.add_argument("--td-host", type=str, default=config.CONFIG_VALUES.get("TOUCHDESIGNER_HOST", "127.0.0.1"), help="TouchDesigner host IP address")
parser.add_argument("--td-port", type=int, default=int(config.CONFIG_VALUES.get("TOUCHDESIGNER_PORT", "7000")), help="TouchDesigner UDP port")
args = parser.parse_args()

# Update audio path in config
config.AUDIO_PATH = os.path.abspath(args.audio_path)
print(f"Using audio path: {config.AUDIO_PATH}")

# Enhanced terminal colors for better visibility
USER_COLOR = config.NEON_GREEN  # Brighter green for user text
ASSISTANT_COLOR = config.PINK   # Keep pink for assistant responses
SYSTEM_COLOR = '\033[94m'       # Blue for system messages

# Initialize the LLM service
llm_service = LLMService()

# Main conversation function
def user_chatbot_conversation():
    """
    Main user-chatbot conversation function.
    Handles speech recognition and chatbot communication.
    """
    if not llm_service.ensure_llm_connection():
        print("LLM service is not available. Please check the service and try again.")
        return
    
    # Send loading state to TouchDesigner
    touchdesigner_service.loading()
    
    # Initialize conversation history
    global conversation_history, system_message
    conversation_history = []
    
    # Set up the system message
    system_message = config.DEFAULT_SYSTEM_MESSAGE
    
    # Initialize the services
    print(f"\n{SYSTEM_COLOR}🔄 Initializing services...{config.RESET_COLOR}")
    
    # Check Whisper API availability
    if not whisper_client.is_available():
        print(f"{SYSTEM_COLOR}❌ Speech recognition service is not available - cannot continue{config.RESET_COLOR}")
        return
        
    # Check TTS service
    if not tts_client.is_available():
        print("TTS service is not available. Please start the TTS API server.")
        return
    
    # Send ready state to TouchDesigner
    touchdesigner_service.ready()
    
    # Check for activation options
    keyboard_shortcut_enabled = args.keyboard_shortcut
    
    if keyboard_shortcut_enabled:
        print(f"{SYSTEM_COLOR}🔵 Keyboard shortcut (Ctrl+R) enabled. Press Ctrl+R to start interaction.{config.RESET_COLOR}")
        try:
            # Setup the keyboard shortcut
            def keyboard_callback():
                """Synchronous wrapper for the keyboard handler"""
                print(f"\n{SYSTEM_COLOR}--- Ctrl+R detected --- {config.RESET_COLOR}")
                try:
                    # Create a new event loop for this specific operation
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    # Run the async handler in the new loop
                    loop.run_until_complete(keyboard_recording_handler(conversation_history, system_message))
                    
                    # Clean up the loop
                    loop.close()
                    
                    print(f"\n{SYSTEM_COLOR}⏳ Waiting for next Ctrl+R...{config.RESET_COLOR}")
                except Exception as handler_error:
                    print(f"\n{SYSTEM_COLOR}❌ Error in keyboard handler execution: {handler_error}{config.RESET_COLOR}")
                    import traceback
                    traceback.print_exc()
                    print(f"\n{SYSTEM_COLOR}⏳ Waiting for next Ctrl+R...{config.RESET_COLOR}")
            
            # Register the keyboard shortcut with error handling
            try:
                keyboard.add_hotkey('ctrl+r', keyboard_callback)
                print(f"{SYSTEM_COLOR}✅ Keyboard shortcut registered successfully{config.RESET_COLOR}")
            except Exception as kb_error:
                print(f"{SYSTEM_COLOR}❌ Error setting up keyboard shortcut: {kb_error}{config.RESET_COLOR}")
                print(f"{SYSTEM_COLOR}⚠️ Falling back to regular conversation mode{config.RESET_COLOR}")
                keyboard_shortcut_enabled = False
            
            # If successfully set up, keep program running
            if keyboard_shortcut_enabled:
                print(f"\n{SYSTEM_COLOR}⌨️  Press Ctrl+R anytime to start a conversation, or Ctrl+C to exit.{config.RESET_COLOR}")
                print(f"{SYSTEM_COLOR}⏳ Waiting for activation...{config.RESET_COLOR}")
                
                try:
                    # Keep the program running until Ctrl+C
                    while True:
                        time.sleep(1)  # Sleep to reduce CPU usage
                except KeyboardInterrupt:
                    print(f"\n{SYSTEM_COLOR}👋 Exiting by keyboard interrupt.{config.RESET_COLOR}")
                finally:
                    # Close TouchDesigner communication
                    touchdesigner_service.close_communication()
                    print(f"{SYSTEM_COLOR}Memory saved on exit.{config.RESET_COLOR}")
                    # Unhook hotkey (optional, good practice)
                    try:
                        keyboard.remove_hotkey('ctrl+r')
                    except:
                        pass 
                    return  # Exit the function
        except Exception as e:
            print(f"{SYSTEM_COLOR}❌ Error in keyboard shortcut mode setup: {e}{config.RESET_COLOR}")
            print(f"{SYSTEM_COLOR}⚠️ Falling back to regular conversation mode{config.RESET_COLOR}")
            keyboard_shortcut_enabled = False

async def keyboard_recording_handler(conversation_history, system_message):
    """
    Handle the full conversation loop initiated by Ctrl+R:
    1. Signal TouchDesigner (listening state)
    2. Wait for audio file from Whisper server
    3. Process transcription with Ollama LLM
    4. Generate speech with F5TTS
    5. Loop until end command received
    """
    # Notify TouchDesigner we're in listening state
    touchdesigner_service.listening()
    
    try:
        # Wait for audio file from Whisper server
        print(f"\n{SYSTEM_COLOR}🎙️ Waiting for audio from Whisper server...{config.RESET_COLOR}")
        
        # Once the audio file is available, transcribe it
        user_input, detected_language = await whisper_client.transcribe_audio(
            None,  # No audio file path needed, handled by whisper client
            language=None,  # Auto-detect language
            quiet=True  # Suppress detailed transcription progress
        )
        
        if not user_input:
            print(f"{SYSTEM_COLOR}❌ No speech detected. Waiting for next activation.{config.RESET_COLOR}")
            return
            
        print(f"{USER_COLOR}You:{config.RESET_COLOR} {user_input}")
        
        # Check for exit command
        if any(command in user_input.lower() for command in ["adios", "exit", "salir", "terminar", "quit"]):
            print(f"{SYSTEM_COLOR}👋 Goodbye!{config.RESET_COLOR}")
            return
        
        # Process with LLM (Ollama)
        print(f"{SYSTEM_COLOR}🤔 Processing via LLM: '{user_input}'{config.RESET_COLOR}")
        
        # Notify TouchDesigner we're in thinking state
        touchdesigner_service.thinking()
        
        # Get the transcription file path from the handler
        from api_client.tempfile_handler import transcription_handler
        transcription_file = transcription_handler.get_file_path()
        
        # Process transcription with LLM
        response, success = await llm_client.process_transcription(transcription_file)
        
        if not success or not response:
            print(f"{SYSTEM_COLOR}❌ No valid response from LLM. Waiting for next activation.{config.RESET_COLOR}")
            return
            
        # Add to conversation history
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": response})
        
        # Display the response
        print(f"{ASSISTANT_COLOR}Assistant:{config.RESET_COLOR} {response}")
        
        # Notify TouchDesigner we're in speaking state
        touchdesigner_service.speaking()
        
        # Generate speech using F5-TTS
        success, audio_length = await tts_client.synthesize_text(response, language_code=detected_language)
        if not success:
            print(f"{SYSTEM_COLOR}❌ Failed to generate speech{config.RESET_COLOR}")
        else:
            print(f"{SYSTEM_COLOR}✅ Generated speech (length: {audio_length:.2f}s){config.RESET_COLOR}")
        
        # Return to ready state after processing
        touchdesigner_service.ready()
        
    except Exception as e:
        print(f"{SYSTEM_COLOR}❌ Error in keyboard_recording_handler: {e}{config.RESET_COLOR}")
        import traceback
        traceback.print_exc()

# Entry point
if __name__ == "__main__":
    # Print system information
    print("=" * 50)
    print(f"{SYSTEM_COLOR}🎤 Speech-to-text system using OpenAI Whisper{config.RESET_COLOR}")
    print("=" * 50)
    
    # Print TouchDesigner connection details
    print(f"{SYSTEM_COLOR}🔄 TouchDesigner connection: UDP to {args.td_host}:{args.td_port}{config.RESET_COLOR}")
    print("=" * 50)
    
    # Print system information
    system_info = general_utils.get_system_info()
    
    if "device" in system_info and system_info["device"] == "gpu":
        print(f"{SYSTEM_COLOR}💻 CUDA available: {system_info.get('gpu_name', 'GPU')}{config.RESET_COLOR}")
    else:
        print(f"{SYSTEM_COLOR}💻 CUDA not available. Using CPU.{config.RESET_COLOR}")
        print(f"{SYSTEM_COLOR}   For better performance, consider using a system with NVIDIA GPU.{config.RESET_COLOR}")
    
    # CPU information
    print(f"\n{SYSTEM_COLOR}💻 CPU: {system_info.get('cpu', 'Unknown')}{config.RESET_COLOR}")
    print(f"{SYSTEM_COLOR}   Physical cores: {system_info.get('cpu_count', 'Unknown')}{config.RESET_COLOR}")
    print(f"{SYSTEM_COLOR}   Logical cores: {system_info.get('cpu_count_logical', 'Unknown')}{config.RESET_COLOR}")
    print(f"{SYSTEM_COLOR}   RAM: {system_info.get('total_ram_gb', 'Unknown')} GB{config.RESET_COLOR}")
    
    print(f"\n{SYSTEM_COLOR}💡 Available commands:{config.RESET_COLOR}")
    print(f"{SYSTEM_COLOR}   - 'exit', 'salir', or 'adios': End the conversation{config.RESET_COLOR}")
    
    if args.keyboard_shortcut:
        print(f"\n{SYSTEM_COLOR}⌨️  Keyboard shortcut (Ctrl+R) enabled. Press Ctrl+R anytime to start recording{config.RESET_COLOR}")
    else:
        print(f"\n{SYSTEM_COLOR}⌨️  To enable keyboard shortcuts, start with: python main.py --keyboard_shortcut{config.RESET_COLOR}")
    
    print("\n" + "=" * 50)
    
    # Start the conversation
    user_chatbot_conversation()

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4000) 