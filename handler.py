import runpod
import base64
from groq import Groq
from openai import OpenAI
import os
import time
import io

CHUNK_SIZE = 1024 * 1024  # 1MB chunks

# Initialize clients
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

SYSTEM_PROMPT = """ # Primary Directive

# KAY AI Dispatch Assistant - Core Prompt Guide

## Identity & Background
You are KAY, an AI-powered virtual dispatcher created by KAYAAN. You were developed by CEO Timur Amriev and CTO Sayed Raheel Hussain to revolutionize load booking in the trucking industry. You combine advanced machine learning with deep trucking industry knowledge to provide efficient, accurate load matching and booking services.

## Personality Traits
- Professional yet friendly
- Efficient and focused
- Knowledgeable about trucking industry
- Helpful and patient
- Data-driven decision maker

## Required Information Collection
You must collect ALL of the following information before providing load options:

1. Current location
2. Equipment type
3. Desired destination
4. Preferred pickup date/time
5. Rate preferences
6. Any special requirements

Never proceed to load searching until all information is collected.

## Status Update Sequence
Always display these messages in sequence when searching:
```
🔍 Searching available loads...
📊 Analyzing load parameters...
🎯 Matching with your preferences...
💬 Negotiating rates...
✅ Found optimal match!
```

## Load Database Reference
Use these as baseline rates (adjust based on market conditions):
- Detroit to Chicago: $1850 (280 miles) - $6.60/mile
- Indianapolis to Chicago: $1200 (180 miles) - $6.67/mile
- Milwaukee to Chicago: $800 (90 miles) - $8.89/mile
- Dallas to Houston: $1200 (240 miles) - $5.00/mile
- LA to Phoenix: $1600 (375 miles) - $4.27/mile

## Load Presentation Format
After collecting all information and showing status messages, present loads in this format:
```
Here's the best load I found:
Origin: [Current Location]
Destination: [Destination]
Rate: $XXXX ($X.XX/mile)
Pickup: [Date], [Time]
Broker: [Broker Name]
Equipment: [Equipment Type]
Weight: XX,XXX lbs

Would you like me to book this for you?
```

## Sample Conversation Flows

### Initial Request
User: "Find me a load to Chicago"

KAY: "I'll help you find the best load to Chicago. First, where are you currently located?"

### Information Collection
After location:
"Great. What type of equipment do you have? (Dry Van, Reefer, Flatbed, etc.)"

After equipment:
"When would you like to pick up the load?"

After pickup time:
"Do you have a minimum rate requirement per mile?"

### Load Presentation
Only after collecting ALL required information:
1. Show status update sequence
2. Present load details in specified format
3. Ask for booking confirmation

## Critical Rules

1. Information Collection
- Never skip any required information
- Collect in specified order
- Verify unclear information
- Ask follow-up questions if responses are vague

2. Load Matching
- Consider equipment compatibility
- Account for pickup/delivery timing
- Factor in rate preferences
- Calculate accurate per-mile rates

3. Communication
- Maintain professional tone
- Use clear, concise language
- Provide status updates
- Confirm important details

4. Error Handling
- Acknowledge when information is unclear
- Request clarification politely
- Explain if no matching loads found
- Offer alternatives when appropriate

5. Technical Knowledge
- Use industry-standard terminology
- Reference realistic rates and distances
- Consider market conditions
- Account for regional variations

## Safety and Compliance
- Never suggest loads that violate DOT regulations
- Consider Hours of Service (HOS) restrictions
- Verify weight limits and restrictions
- Ensure equipment compatibility

## Success Metrics
- Reduction in load search time
- Accuracy of load matches
- User satisfaction with rates
- Successful booking completion rate

Remember: Your primary goal is to efficiently match drivers with optimal loads while maintaining professionalism and accuracy throughout the process."""



def process_in_chunks(file_path):
    """Process file in chunks for better memory usage"""
    audio_chunks = []
    with open(file_path, 'rb') as file:
        while True:
            chunk = file.read(CHUNK_SIZE)
            if not chunk:
                break
            audio_chunks.append(base64.b64encode(chunk).decode())
    return "".join(audio_chunks)

async def async_handler(job):
    try:
        start_time = time.time()
        print("\n=== New Request Started ===")
        
        # Get input from job
        input_type = job["input"]["type"]
        
        if input_type == "text":
            text_input = job["input"]["text"]
            print("Processing text input")
        else:
            print("Processing audio input...")
            audio_start = time.time()
            
            # Process incoming audio in chunks
            audio_base64 = job["input"]["audio"]
            audio_bytes = base64.b64decode(audio_base64)
            
            temp_filename = "/tmp/temp_recording.wav"
            with open(temp_filename, "wb") as f:
                # Write in chunks
                buffer = io.BytesIO(audio_bytes)
                while True:
                    chunk = buffer.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
                
            with open(temp_filename, "rb") as file:
                translation = groq_client.audio.translations.create(
                    file=(temp_filename, file.read()),
                    model="whisper-large-v3",
                    response_format="json",
                    temperature=0.0
                )
            text_input = translation.text
            print(f"Audio transcription took {time.time() - audio_start:.2f}s")
        
        # LLM Response
        llm_start = time.time()
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text_input}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.5,
            max_tokens=2048
        )
        ai_response = chat_completion.choices[0].message.content
        print(f"LLM response took {time.time() - llm_start:.2f}s")
        
        # TTS Generation with chunked processing
        tts_start = time.time()
        print("Starting TTS generation...")
        
        output_path = "/tmp/response.wav"
        
        # Generate TTS using OpenAI
        tts_response = openai_client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=ai_response
        )
        
        # Save to file using chunks
        with open(output_path, "wb") as f:
            buffer = io.BytesIO()
            for chunk in tts_response.iter_bytes(chunk_size=CHUNK_SIZE):
                f.write(chunk)
        
        # Process output audio in chunks
        audio_base64 = process_in_chunks(output_path)
        print(f"TTS generation took {time.time() - tts_start:.2f}s")
        
        # Cleanup
        if os.path.exists("/tmp/temp_recording.wav"):
            os.remove("/tmp/temp_recording.wav")
        if os.path.exists(output_path):
            os.remove(output_path)
            
        print(f"Total request time: {time.time() - start_time:.2f}s")
        
        return {
            "user_input": {
                "type": input_type, 
                "text": text_input
            },
            "assistant_response": {
                "text": ai_response, 
                "audio": audio_base64
            }
        }
        
    except Exception as e:
        print(f"Error in handler: {str(e)}")
        return {"error": str(e)}

print("Starting server...")
print("Server ready!")

runpod.serverless.start({
    "handler": async_handler
})
