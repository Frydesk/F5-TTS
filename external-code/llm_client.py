"""
LLM Client for interacting with Ollama API
"""

import httpx
import config
import os
import asyncio
import websockets
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# Use the same temp directory as whisper client
TEMP_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "temp"
TEMP_DIR.mkdir(exist_ok=True)

class LLMClient:
    def __init__(self):
        self.base_url = f"http://{config.LLM_CHAT_HOST}:{config.LLM_CHAT_PORT}"
        self.ws_url = f"ws://{config.LLM_CHAT_HOST}:8100/airesponse"
        self.model = config.OLLAMA_MODEL
        self.temperature = config.OLLAMA_TEMPERATURE
        self.websocket = None
        self.connection_active = False
        self.response_received = asyncio.Event()
        self.llm_response = None
        self.temp_file = TEMP_DIR / "llm_response.txt"

    def is_available(self):
        """Check if LLM service is available"""
        try:
            response = httpx.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False

    def save_response(self, text):
        """Save LLM response to temporary file using atomic operations"""
        try:
            # Atomic write using temporary file
            temp_path = self.temp_file.with_suffix('.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(text)
            # Atomic rename
            shutil.move(temp_path, self.temp_file)
            print(f"[DEBUG] Saved LLM response to {self.temp_file}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to save LLM response: {e}")
            return False

    async def connect_websocket(self):
        """Establish WebSocket connection"""
        try:
            if self.websocket:
                try:
                    await self.websocket.close()
                except:
                    pass
                self.websocket = None
            print(f"[DEBUG] Connecting to LLM WebSocket at {self.ws_url}")
            self.websocket = await websockets.connect(self.ws_url)
            self.connection_active = True
            print("[DEBUG] LLM WebSocket connection established")
            return True
        except Exception as e:
            print(f"[ERROR] LLM WebSocket connection error: {e}")
            self.connection_active = False
            return False

    async def send_message(self, message):
        """Send message through WebSocket"""
        if not self.websocket or not self.connection_active:
            if not await self.connect_websocket():
                return False
        try:
            print(f"[DEBUG] Sending message to LLM: {message}")
            await self.websocket.send(json.dumps(message))
            return True
        except Exception as e:
            print(f"[ERROR] Error sending LLM WebSocket message: {e}")
            self.connection_active = False
            return False

    async def receive_messages(self):
        """Receive messages from WebSocket"""
        if not self.websocket:
            print("[ERROR] No WebSocket connection available")
            return
        
        try:
            while self.connection_active:
                try:
                    print("[DEBUG] Waiting for WebSocket message...")
                    message = await self.websocket.recv()
                    print(f"[DEBUG] Raw message received: {message}")
                    data = json.loads(message)
                    print(f"[DEBUG] Parsed message data: {data}")
                    
                    if data.get("type") == "answer":
                        answer = data.get("data", {}).get("answer", "")
                        sources = data.get("data", {}).get("sources", [])
                        print(f"\n[DEBUG] Answer received: {answer[:100]}...")  # Print first 100 chars
                        print(f"[DEBUG] Sources received: {sources}")
                        self.llm_response = answer
                        self.response_received.set()
                        # Save response to temporary file
                        self.save_response(self.llm_response)
                    elif data.get("error"):
                        print(f"\n❌ LLM WebSocket error: {data['error']}")
                        self.connection_active = False
                        break
                    else:
                        print(f"[DEBUG] Unhandled message type: {data.get('type')}")
                        
                except websockets.exceptions.ConnectionClosed:
                    print("\n❌ LLM WebSocket connection closed")
                    self.connection_active = False
                    break
                except json.JSONDecodeError as e:
                    print(f"\n[ERROR] Failed to parse message as JSON: {e}")
                    print(f"Raw message was: {message}")
                    continue
                except Exception as e:
                    print(f"\n❌ Error receiving LLM WebSocket message: {type(e).__name__}: {str(e)}")
                    self.connection_active = False
                    break
                    
        finally:
            self.connection_active = False
            print("[DEBUG] Receive messages loop ended")

    async def process_transcription(self, transcription_file):
        """
        Process transcription from temporary file and get LLM response
        
        Args:
            transcription_file (str): Path to the transcription file
            
        Returns:
            tuple: (response_text, success)
        """
        try:
            # Reset response state
            self.llm_response = None
            self.response_received.clear()
            
            # Read transcription from temporary file
            try:
                with open(transcription_file, 'r', encoding='utf-8') as f:
                    transcription = f.read().strip()
            except Exception as e:
                print(f"[ERROR] Failed to read transcription file: {e}")
                return None, False

            if not transcription:
                print("[ERROR] Empty transcription file")
                return None, False

            # Connect to WebSocket server
            if not await self.connect_websocket():
                return None, False

            print("[DEBUG] Sending 'start' message...")
            # Send raw "start" message as expected by the server
            try:
                await self.websocket.send("start")
            except Exception as e:
                print(f"[ERROR] Failed to send 'start' message: {e}")
                return None, False

            print("[DEBUG] Sending transcription message...")
            print(f"[DEBUG] Transcription content: {transcription[:100]}...")  # Print first 100 chars
            # Send the transcription message with proper JSON formatting
            transcription_message = {"type": "message", "content": transcription}
            if not await self.send_message(transcription_message):
                print("[ERROR] Failed to send transcription message")
                return None, False

            # Start receiving messages in a separate task
            print("[DEBUG] Starting receive_messages task...")
            receive_task = asyncio.create_task(self.receive_messages())
            
            # Wait for response
            print("[DEBUG] Waiting for LLM response...")
            await self.response_received.wait()
            print("[DEBUG] LLM response received")

            return self.llm_response, True
                
        except Exception as e:
            print(f"\n❌ Error in process_transcription: {e}")
            return None, False

# Create a singleton instance
llm_client = LLMClient() 