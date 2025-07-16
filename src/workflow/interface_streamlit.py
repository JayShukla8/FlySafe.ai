import streamlit as st
import sys
import os
import time
import asyncio
from streamlit_float import *
import base64
# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)
from src.utils.tools.videoUnderstanding import _vrun_flight_inspector

#function for creating transcript
def write_transcript(conversation: str) -> str:
    """Write the conversation to a transcript file."""
    path = "transcript.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(conversation)
    return path

#---------------------------------------------------------------------------------------------------
#code for amazon nova sonic
import os
import asyncio
import base64
import json
import uuid
import warnings
import pyaudio
import pytz
import datetime
import time
import inspect
from aws_sdk_bedrock_runtime.client import BedrockRuntimeClient, InvokeModelWithBidirectionalStreamOperationInput
from aws_sdk_bedrock_runtime.models import InvokeModelWithBidirectionalStreamInputChunk, BidirectionalInputPayloadPart
from aws_sdk_bedrock_runtime.config import Config, HTTPAuthSchemeResolver, SigV4AuthScheme
from smithy_aws_core.credentials_resolvers.environment import EnvironmentCredentialsResolver
from src.utils.tools.convert2pdf import convert_2_pdf
from src.utils.tools.sendemail import SendEmail

# Suppress warnings
warnings.filterwarnings("ignore")

# Audio configuration
INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 26000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK_SIZE = 1024  # Number of frames per buffer

# Debug mode flag
DEBUG = False

def debug_print(message):
    """Print only if debug mode is enabled"""
    if DEBUG:
        functionName = inspect.stack()[1].function
        if  functionName == 'time_it' or functionName == 'time_it_async':
            functionName = inspect.stack()[2].function
        print('{:%Y-%m-%d %H:%M:%S.%f}'.format(datetime.datetime.now())[:-3] + ' ' + functionName + ' ' + message)

def time_it(label, methodToRun):
    start_time = time.perf_counter()
    result = methodToRun()
    end_time = time.perf_counter()
    debug_print(f"Execution time for {label}: {end_time - start_time:.4f} seconds")
    return result

async def time_it_async(label, methodToRun):
    start_time = time.perf_counter()
    result = await methodToRun()
    end_time = time.perf_counter()
    debug_print(f"Execution time for {label}: {end_time - start_time:.4f} seconds")
    return result

class BedrockStreamManager:
    """Manages bidirectional streaming with AWS Bedrock using asyncio"""
    
    # Event templates
    START_SESSION_EVENT = '''{
        "event": {
            "sessionStart": {
            "inferenceConfiguration": {
                "maxTokens": 1024,
                "topP": 0.9,
                "temperature": 0.7
                }
            }
        }
    }'''

    CONTENT_START_EVENT = '''{
        "event": {
            "contentStart": {
            "promptName": "%s",
            "contentName": "%s",
            "type": "AUDIO",
            "interactive": true,
            "role": "USER",
            "audioInputConfiguration": {
                "mediaType": "audio/lpcm",
                "sampleRateHertz": 16000,
                "sampleSizeBits": 16,
                "channelCount": 1,
                "audioType": "SPEECH",
                "encoding": "base64"
                }
            }
        }
    }'''

    AUDIO_EVENT_TEMPLATE = '''{
        "event": {
            "audioInput": {
            "promptName": "%s",
            "contentName": "%s",
            "content": "%s"
            }
        }
    }'''

    TEXT_CONTENT_START_EVENT = '''{
        "event": {
            "contentStart": {
            "promptName": "%s",
            "contentName": "%s",
            "type": "TEXT",
            "role": "%s",
            "interactive": true,
                "textInputConfiguration": {
                    "mediaType": "text/plain"
                }
            }
        }
    }'''

    TEXT_INPUT_EVENT = '''{
        "event": {
            "textInput": {
            "promptName": "%s",
            "contentName": "%s",
            "content": "%s"
            }
        }
    }'''

    TOOL_CONTENT_START_EVENT = '''{
        "event": {
            "contentStart": {
                "promptName": "%s",
                "contentName": "%s",
                "interactive": false,
                "type": "TOOL",
                "role": "TOOL",
                "toolResultInputConfiguration": {
                    "toolUseId": "%s",
                    "type": "TEXT",
                    "textInputConfiguration": {
                        "mediaType": "text/plain"
                    }
                }
            }
        }
    }'''

    CONTENT_END_EVENT = '''{
        "event": {
            "contentEnd": {
            "promptName": "%s",
            "contentName": "%s"
            }
        }
    }'''

    PROMPT_END_EVENT = '''{
        "event": {
            "promptEnd": {
            "promptName": "%s"
            }
        }
    }'''

    SESSION_END_EVENT = '''{
        "event": {
            "sessionEnd": {}
        }
    }'''
    
    def start_prompt(self):
        """Create a promptStart event"""
        get_default_tool_schema = json.dumps({
            "type": "object",
            "properties": {},
            "required": []
        })

        get_search_tool_schema = json.dumps({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Query to search using SERPAPI tool"
                }
            },
            "required": ["query"]
        })

        generate_invoice_pdf_schema = json.dumps({
            "type": "object",
            "properties": {
                "option": {
                    "type": "string",
                    "description": "The option chosen by the user from the given options"
                }
            },
            "required": ["option"]
        })

        send_invoice_email = json.dumps({
            "type": "object",
            "properties": {
                "option": {
                    "type": "string",
                    "description": "The option chosen by the user from the given options"
                }
            },
            "required": ["confirmation","option"]
        })
        
        prompt_start_event = {
            "event": {
                "promptStart": {
                    "promptName": self.prompt_name,
                    "textOutputConfiguration": {
                        "mediaType": "text/plain"
                    },
                    "audioOutputConfiguration": {
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": 24000,
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                        "voiceId": "tiffany",
                        "encoding": "base64",
                        "audioType": "SPEECH"
                    },
                    "toolUseOutputConfiguration": {
                        "mediaType": "application/json"
                    },
                    "toolConfiguration": {
                        "tools": [
                            {
                                "toolSpec": {
                                    "name": "getDateAndTimeTool",
                                    "description": "get information about the current date and time",
                                    "inputSchema": {
                                        "json": get_default_tool_schema
                                    }
                                }
                            },
                            {
                                "toolSpec": {
                                    "name": "serpAPITool",
                                    "description": "search products using SERPAPI and return formatted results",
                                    "inputSchema": {
                                    "json": get_search_tool_schema
                                    }
                                }
                            },
                            {
                                "toolSpec": {
                                    "name": "generateInvoicePDF",
                                    "description": "generate a PDF of the invoice",
                                    "inputSchema": {
                                        "json": generate_invoice_pdf_schema
                                    }
                                }
                            },
                            {
                                "toolSpec": {
                                    "name": "sendEmail",
                                    "description": "send mail to the user",
                                    "inputSchema": {
                                        "json": send_invoice_email
                                    }
                                }
                            },
                        ]
                    }
                }
            }
        }
        
        return json.dumps(prompt_start_event)
    
    def tool_result_event(self, content_name, content, role):
        """Create a tool result event"""

        if isinstance(content, dict):
            content_json_string = json.dumps(content)
        else:
            content_json_string = content
            
        tool_result_event = {
            "event": {
                "toolResult": {
                    "promptName": self.prompt_name,
                    "contentName": content_name,
                    "content": content_json_string
                }
            }
        }
        return json.dumps(tool_result_event)
   
    def __init__(self, report, user_email, model_id='amazon.nova-sonic-v1:0', region='us-east-1'):
        """Initialize the stream manager."""
        self.model_id = model_id
        self.region = region
        
        # Replace RxPy subjects with asyncio queues
        self.audio_input_queue = asyncio.Queue()
        self.audio_output_queue = asyncio.Queue()
        self.output_queue = asyncio.Queue()
        
        self.response_task = None
        self.stream_response = None
        self.is_active = False
        self.barge_in = False
        self.bedrock_client = None
        
        # Audio playback components
        self.audio_player = None
        
        # Text response components
        self.display_assistant_text = False
        self.role = None

        # Session information
        self.prompt_name = str(uuid.uuid4())
        self.content_name = str(uuid.uuid4())
        self.audio_content_name = str(uuid.uuid4())
        self.toolUseContent = ""
        self.toolUseId = ""
        self.toolName = ""
        self.inspection_report = report
        # self.conversation_history = conversation_history
        self.user_email = user_email
        self.options = {1:{"product":"","price":"","source":""},
                        2:{"product":"","price":"","source":""},
                        3:{"product":"","price":"","source":""}}
        self.option = 0

    def _initialize_client(self):
        """Initialize the Bedrock client."""
        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{self.region}.amazonaws.com",
            region=self.region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
            http_auth_scheme_resolver=HTTPAuthSchemeResolver(),
            http_auth_schemes={"aws.auth#sigv4": SigV4AuthScheme()}
        )
        self.bedrock_client = BedrockRuntimeClient(config=config)
    
    async def initialize_stream(self):
        """Initialize the bidirectional stream with Bedrock."""
        if not self.bedrock_client:
            self._initialize_client()
        try:
            self.stream_response = await time_it_async("invoke_model_with_bidirectional_stream", lambda : self.bedrock_client.invoke_model_with_bidirectional_stream( InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id)))
            self.is_active = True
            default_system_prompt = f'''You are a friendly assistant. The user and you will engage in a spoken dialog exchanging the transcripts of a natural real-time conversation. The following is an inspection report: {self.inspection_report}, guide the user on inspection analysis and answer questions about the inspection. When user asks why a step was not performed, say that the wing inspection was incomplete due to limited access angle. When user asks about damages, always include the following in your response: The inspection revealed that the main landing gear tires are worn out and may need replacement. When user asks about tire inventory, always respond with: Yes, we have a few items available in our inventory. There is one main landing gear tire available in Section B, but given the wear pattern, you might want to consider ordering additional tires for future maintenance. If user asks you to buy something or regarding any purchase, invoke the SerpAPITool with query as the product that user wants to buy and just tell him the product names of the products that are available. When the User selects an option, always ask whether he/she would like to go ahead with that purchase. If the user confirms the purchase of any product, invoke the generateInvoicePDF tool with the option provided by the user. Pass option as 1 if user says option 1, pass 2 if user says option 2 and pass 3 if user says option 3. If the invoice pdf gets generated successfully, ask the user if he wants to get the invoice sent to his email. If user says confirm or yes, invoke the sendemail tool and pass the same option that you passed to invoicepdf tool. Make sure the conversation is as natural as possible.'''.replace('\n', ' ')
            # Send initialization events
            prompt_event = self.start_prompt()
            text_content_start = self.TEXT_CONTENT_START_EVENT % (self.prompt_name, self.content_name, "SYSTEM")
            text_content = self.TEXT_INPUT_EVENT % (self.prompt_name, self.content_name, default_system_prompt)
            text_content_end = self.CONTENT_END_EVENT % (self.prompt_name, self.content_name)
            
            init_events = [self.START_SESSION_EVENT, prompt_event, text_content_start, text_content, text_content_end]
            
            for event in init_events:
                await self.send_raw_event(event)
                # Small delay between init events
                await asyncio.sleep(0.1)
            
            # Start listening for responses
            self.response_task = asyncio.create_task(self._process_responses())
            
            # Start processing audio input
            asyncio.create_task(self._process_audio_input())
            
            # Wait a bit to ensure everything is set up
            await asyncio.sleep(0.1)
            
            debug_print("Stream initialized successfully")
            return self
        except Exception as e:
            self.is_active = False
            print(f"Failed to initialize stream: {str(e)}")
            raise
    
    async def send_raw_event(self, event_json):
        """Send a raw event JSON to the Bedrock stream."""
        if not self.stream_response or not self.is_active:
            debug_print("Stream not initialized or closed")
            return
       
        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=event_json.encode('utf-8'))
        )
        
        try:
            await self.stream_response.input_stream.send(event)
            # For debugging large events, you might want to log just the type
            if DEBUG:
                if len(event_json) > 200:
                    event_type = json.loads(event_json).get("event", {}).keys()
                    debug_print(f"Sent event type: {list(event_type)}")
                else:
                    debug_print(f"Sent event: {event_json}")
        except Exception as e:
            debug_print(f"Error sending event: {str(e)}")
            if DEBUG:
                import traceback
                traceback.print_exc()
    
    async def send_audio_content_start_event(self):
        """Send a content start event to the Bedrock stream."""
        content_start_event = self.CONTENT_START_EVENT % (self.prompt_name, self.audio_content_name)
        await self.send_raw_event(content_start_event)
    
    async def _process_audio_input(self):
        """Process audio input from the queue and send to Bedrock."""
        while self.is_active:
            try:
                # Get audio data from the queue
                data = await self.audio_input_queue.get()
                
                audio_bytes = data.get('audio_bytes')
                if not audio_bytes:
                    debug_print("No audio bytes received")
                    continue
                
                # Base64 encode the audio data
                blob = base64.b64encode(audio_bytes)
                audio_event = self.AUDIO_EVENT_TEMPLATE % (
                    self.prompt_name, 
                    self.audio_content_name, 
                    blob.decode('utf-8')
                )
                
                # Send the event
                await self.send_raw_event(audio_event)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                debug_print(f"Error processing audio: {e}")
                if DEBUG:
                    import traceback
                    traceback.print_exc()
    
    def add_audio_chunk(self, audio_bytes):
        """Add an audio chunk to the queue."""
        self.audio_input_queue.put_nowait({
            'audio_bytes': audio_bytes,
            'prompt_name': self.prompt_name,
            'content_name': self.audio_content_name
        })
    
    async def send_audio_content_end_event(self):
        """Send a content end event to the Bedrock stream."""
        if not self.is_active:
            debug_print("Stream is not active")
            return
        
        content_end_event = self.CONTENT_END_EVENT % (self.prompt_name, self.audio_content_name)
        await self.send_raw_event(content_end_event)
        debug_print("Audio ended")
    
    async def send_tool_start_event(self, content_name):
        """Send a tool content start event to the Bedrock stream."""
        content_start_event = self.TOOL_CONTENT_START_EVENT % (self.prompt_name, content_name, self.toolUseId)
        debug_print(f"Sending tool start event: {content_start_event}")  
        await self.send_raw_event(content_start_event)

    async def send_tool_result_event(self, content_name, tool_result):
        """Send a tool content event to the Bedrock stream."""
        # Use the actual tool result from processToolUse
        tool_result_event = self.tool_result_event(content_name=content_name, content=tool_result, role="TOOL")
        debug_print(f"Sending tool result event: {tool_result_event}")
        await self.send_raw_event(tool_result_event)
    
    async def send_tool_content_end_event(self, content_name):
        """Send a tool content end event to the Bedrock stream."""
        tool_content_end_event = self.CONTENT_END_EVENT % (self.prompt_name, content_name)
        debug_print(f"Sending tool content event: {tool_content_end_event}")
        await self.send_raw_event(tool_content_end_event)
    
    async def send_prompt_end_event(self):
        """Close the stream and clean up resources."""
        if not self.is_active:
            debug_print("Stream is not active")
            return
        
        prompt_end_event = self.PROMPT_END_EVENT % (self.prompt_name)
        await self.send_raw_event(prompt_end_event)
        debug_print("Prompt ended")
        
    async def send_session_end_event(self):
        """Send a session end event to the Bedrock stream."""
        if not self.is_active:
            debug_print("Stream is not active")
            return

        await self.send_raw_event(self.SESSION_END_EVENT)
        self.is_active = False
        debug_print("Session ended")
    
    async def _process_responses(self):
        """Process incoming responses from Bedrock."""
        try:            
            while self.is_active:
                try:
                    output = await self.stream_response.await_output()
                    result = await output[1].receive()
                    if result.value and result.value.bytes_:
                        try:
                            response_data = result.value.bytes_.decode('utf-8')
                            json_data = json.loads(response_data)
                            
                            # Handle different response types
                            if 'event' in json_data:
                                if 'contentStart' in json_data['event']:
                                    debug_print("Content start detected")
                                    content_start = json_data['event']['contentStart']
                                    # set role
                                    self.role = content_start['role']
                                    # Check for speculative content
                                    if 'additionalModelFields' in content_start:
                                        try:
                                            additional_fields = json.loads(content_start['additionalModelFields'])
                                            if additional_fields.get('generationStage') == 'SPECULATIVE':
                                                debug_print("Speculative content detected")
                                                self.display_assistant_text = True
                                            else:
                                                self.display_assistant_text = False
                                        except json.JSONDecodeError:
                                            debug_print("Error parsing additionalModelFields")
                                elif 'textOutput' in json_data['event']:
                                    text_content = json_data['event']['textOutput']['content']
                                    role = json_data['event']['textOutput']['role']
                                    # Check if there is a barge-in
                                    if '{ "interrupted" : true }' in text_content:
                                        debug_print("Barge-in detected. Stopping audio output.")
                                        self.barge_in = True

                                    if (self.role == "ASSISTANT" and self.display_assistant_text):
                                        print(f"Assistant: {text_content}")
                                        with chat_container.chat_message("assistant", avatar="✈️"):
                                            with st.spinner("..."):
                                                st.write(f"{text_content}")
                                        st.session_state.messages.append({"role": "assistant", "content": f"{text_content}"})
                                        # self.conversation_history.append(f"Assistant: {text_content}")
                                        # with open("transcript_nova_sonic_tool.txt", "a") as file:
                                        #     file.write(f"Assistant: {text_content}\n")
                                    elif (self.role == "USER"):
                                        print(f"User: {text_content}")
                                        with chat_container.chat_message("user", avatar="👤"):
                                            st.write(f"{text_content}")
                                        st.session_state.messages.append({"role": "user", "content": f"{text_content}"})
                                        # self.conversation_history.append(f"User: {text_content}")
                                        # with open("transcript_nova_sonic_tool.txt", "a") as file:
                                        #     file.write(f"User: {text_content}\n")

                                elif 'audioOutput' in json_data['event']:
                                    audio_content = json_data['event']['audioOutput']['content']
                                    audio_bytes = base64.b64decode(audio_content)
                                    await self.audio_output_queue.put(audio_bytes)
                                elif 'toolUse' in json_data['event']:
                                    self.toolUseContent = json_data['event']['toolUse']
                                    self.toolName = json_data['event']['toolUse']['toolName']
                                    self.toolUseId = json_data['event']['toolUse']['toolUseId']
                                    with chat_container.chat_message("tool", avatar="🛠️"):
                                        st.markdown(
                                            f'''<div class="fancy-tool-message"><b>Tool use detected:</b> {self.toolName}</div>''',
                                            unsafe_allow_html=True
                                        )
                                    st.session_state.messages.append({"role": "tool", "content": f"Tool use detected: {self.toolName}"})       
                                    debug_print(f"Tool use detected: {self.toolName}, ID: {self.toolUseId}")

                                elif 'contentEnd' in json_data['event'] and json_data['event'].get('contentEnd', {}).get('type') == 'TOOL':
                                    debug_print("Processing tool use and sending result")
                                    toolResult = await self.processToolUse(self.toolName, self.toolUseContent)
                                    toolContent = str(uuid.uuid4())
                                    await self.send_tool_start_event(toolContent)
                                    await self.send_tool_result_event(toolContent, toolResult)
                                    await self.send_tool_content_end_event(toolContent)
                                
                                elif 'completionEnd' in json_data['event']:
                                    # Handle end of conversation, no more response will be generated
                                    print("End of response sequence")
                            
                            # Put the response in the output queue for other components
                            await self.output_queue.put(json_data)
                        except json.JSONDecodeError:
                            await self.output_queue.put({"raw_data": response_data})
                except StopAsyncIteration:
                    # Stream has ended
                    break
                except Exception as e:
                   # Handle ValidationException properly
                    if "ValidationException" in str(e):
                        error_message = str(e)
                        print(f"Validation error: {error_message}")
                    else:
                        print(f"Error receiving response: {e}")
                    break
                    
        except Exception as e:
            print(f"Response processing error: {e}")
        finally:
            self.is_active = False

    async def processToolUse(self, toolName, toolUseContent):
        """Return the tool result"""
        tool = toolName.lower()
        debug_print(f"Tool Use Content: {toolUseContent}")
        
        if tool == "getdateandtimetool":
            print('\nTimeToolInvoked\n')
            # Get current date in PST timezone
            pst_timezone = pytz.timezone("America/Los_Angeles")
            pst_date = datetime.datetime.now(pst_timezone)
            
            return {
                "formattedTime": pst_date.strftime("%I:%M %p"),
                "date": pst_date.strftime("%Y-%m-%d"),
                "year": pst_date.year,
                "month": pst_date.month,
                "day": pst_date.day,
                "dayOfWeek": pst_date.strftime("%A").upper(),
                "timezone": "PST"
            }
        
        elif tool == "serpapitool":
            try:
                # print('\nSerpAPITOOL invoked\n')
                content = toolUseContent.get("content", {})
                content_data = json.loads(content)
                query = content_data.get("query")
                return_result = {"results" : ""}
                # print(f"\nquery={query}\n")
                # # For aircraft tires, return predefined realistic options
                # if "tire" in query.lower() or "tyre" in query.lower():
                #     return_result['results'] = """Here are the available options from authorized aviation suppliers: 1. Goodyear Flight Custom III Main Landing Gear Tire Price: $2,850.00 2. Michelin Air X Main Landing Gear Tire Price: $3,200.00 3. Dunlop Aircraft Tyres Main Landing Gear Tire Price: $2,950.00 All tires come with full certification documentation and warranty."""
                #     return return_result
                # For other parts, use the original SerpAPI search
                from serpapi import GoogleSearch
                params = {
                    "q": query,
                    "tbm": "shop",
                    "hl": "en",
                    "gl": "us",
                    "api_key": os.getenv("SERPAPI_API_KEY")
                }
                results = GoogleSearch(params).get_dict()
                if "shopping_results" in results:
                    products = results["shopping_results"][:3]
                    formatted_results = "Here are some options: "
                    i = 1
                    for product in products:
                        formatted_results += f"- {product.get('title', 'N/A')} "
                        self.options[i]["product"] = f"{product.get('title', 'N/A')}"
                        formatted_results += f"  Price: {product.get('price', 'N/A')} "
                        self.options[i]["price"] = f"{product.get('price', 'N/A')}"
                        formatted_results += f"  Source: {product.get('source', 'N/A')} "
                        self.options[i]["source"] = f"{product.get('source', 'N/A')}"
                        i=i+1
                    # print(self.options)
                    return_result['results'] = formatted_results
                    # print(f"\nresults={formatted_results}\n")
                    return return_result
                else:
                    return_result['results'] = "No product results found. Please try a different search query."
                    # print(f"\nresults={return_result}\n")
                    return return_result
            except ImportError:
                return_result['results'] = "Error: SerpAPI package not installed. Please install it using 'pip install google-search-results'"
                return return_result
            except Exception as e:
                return_result['results'] = f"Error searching for products: {str(e)}"
                return return_result
        
        elif tool == "generateinvoicepdf":
            try:
                return_result = {"results" : ""}
                content = toolUseContent.get("content", {})
                content_data = json.loads(content)
                self.option = content_data.get("option")
                print("\n")
                print(self.option)
                print("\n")
                selected_product = self.options[int(self.option)]

                # Create a professional invoice content
                invoice_content = f"""# Flight Inspection Purchase Invoice

                ## Order Details
                - Product: {selected_product["product"]}
                - Order Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
                - Order Status: Confirmed
                - Price: {selected_product["price"]}
                - Payment Status: Pending

                ## Product Specifications
                - Type: {selected_product["product"]}
                - Source: {selected_product["source"]}
                - Condition: New

                ## Purchase Summary
                This invoice confirms your purchase of {selected_product["product"]} for your aircraft. The selected product is a high-quality {selected_product["product"]} that meets all safety and performance standards.

                ## Next Steps
                1. Review the order details above
                2. Complete the payment process
                3. Provide shipping information
                4. Track your order status

                ## Contact Information
                For any questions regarding this purchase, please contact our support team.

                Thank you for choosing our service for your aircraft maintenance needs."""
                
                # Generate PDF directly from content
                try:
                    pdf_path = convert_2_pdf(invoice_content)
                    if not os.path.exists(pdf_path):
                        raise Exception("Failed to create PDF file")
                    print(f"✓ PDF invoice generated: {pdf_path}")
                    return_result["results"] = "PDF Generated"
                    return return_result
                except Exception as e:
                    return_result["results"] = "Error generating PDF"
                    return return_result
            except Exception as e:
                        print(f"\n❌ Error in documentation process: {str(e)}")
                        print("Error type:", type(e))
                        print("Error details:", e.__dict__)
                        print("Please check the configuration and try again.")
        
        elif tool=="sendemail":
            try:
                return_result = {"results" : ""}
                # content = toolUseContent.get("content", {})
                # content_data = json.loads(content)
                # option = content_data.get("option")
                selected_product = self.options[int(self.option)]
                print("\n3. Sending Email...")
                # transcript_content = "\n".join(self.conversation_history)
                transcript_content = ""
                for msg in st.session_state.messages:
                    role = msg["role"]
                    content = msg["content"]
                    if role!="tool":
                        transcript_content += f"{role}: {content}" + "\n"
                email_body = f"""
                <html>
                <body style="font-family: Arial, sans-serif; line-height: 1.6;">
                    <h2 style="color: #2c3e50;">Flight Inspection Purchase Confirmation</h2>
                    
                    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0;">
                        <h3 style="color: #2c3e50;">Order Details</h3>
                        <ul style="list-style-type: none; padding-left: 0;">
                            <li><strong>Product:</strong> {selected_product["product"]}</li>
                            <li><strong>Order Status:</strong> Confirmed</li>
                            <li><strong>Price:</strong> {selected_product["price"]}</li>
                            <li><strong>Source:</strong> {selected_product["source"]}</li>
                            <li><strong>Date:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</li>
                        </ul>
                    </div>

                    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0;">
                        <h3 style="color: #2c3e50;">Conversation Summary</h3>
                        <div style="white-space: pre-wrap;">{transcript_content}</div>
                    </div>

                    <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee;">
                        <p>Please find attached the detailed invoice in PDF format.</p>
                        <p>Your order has been confirmed and will be processed shortly.</p>
                        <p>If you have any questions, please don't hesitate to contact us.</p>
                    </div>
                </body>
                </html>
                """
            
                # Send email
                send_email = SendEmail(message_content=email_body, to_email=self.user_email)
                email_status = send_email._run()
                
                if "Error" in email_status:
                    raise Exception(f"Failed to send email: {email_status}")
                print(f"✓ Email sent successfully to: {self.user_email}")
                
                print("\n=== Documentation Process Completed Successfully ===")
                print("All files have been generated and email has been sent.")
                print("Please check your inbox and spam folder.")
                return_result["results"] = "Email sent"
                return return_result
            except Exception as e:
                        print(f"\n❌ Error in sending email: {str(e)}")
                        print("Error type:", type(e))
                        print("Error details:", e.__dict__)
                        print("Please check the configuration and try again.")
                        return_result["results"] = "Email not sent"
                        return return_result
    
    async def close(self):
        """Close the stream properly."""
        if not self.is_active:
            return
       
        self.is_active = False
        if self.response_task and not self.response_task.done():
            self.response_task.cancel()

        await self.send_audio_content_end_event()
        await self.send_prompt_end_event()
        await self.send_session_end_event()

        if self.stream_response:
            await self.stream_response.input_stream.close()

class AudioStreamer:
    """Handles continuous microphone input and audio output using separate streams."""
    
    def __init__(self, stream_manager):
        self.stream_manager = stream_manager
        self.is_streaming = False
        self.loop = asyncio.get_event_loop()

        # Initialize PyAudio
        debug_print("AudioStreamer Initializing PyAudio...")
        self.p = time_it("AudioStreamerInitPyAudio", pyaudio.PyAudio)
        debug_print("AudioStreamer PyAudio initialized")

        # Initialize separate streams for input and output
        # Input stream with callback for microphone
        debug_print("Opening input audio stream...")
        self.input_stream = time_it("AudioStreamerOpenAudio", lambda  : self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=INPUT_SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
            stream_callback=self.input_callback
        ))
        debug_print("input audio stream opened")

        # Output stream for direct writing (no callback)
        debug_print("Opening output audio stream...")
        self.output_stream = time_it("AudioStreamerOpenAudio", lambda  : self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=OUTPUT_SAMPLE_RATE,
            output=True,
            frames_per_buffer=CHUNK_SIZE
        ))

        debug_print("output audio stream opened")

    def input_callback(self, in_data, frame_count, time_info, status):
        """Callback function that schedules audio processing in the asyncio event loop"""
        if self.is_streaming and in_data:
            # Schedule the task in the event loop
            asyncio.run_coroutine_threadsafe(
                self.process_input_audio(in_data), 
                self.loop
            )
        return (None, pyaudio.paContinue)

    async def process_input_audio(self, audio_data):
        """Process a single audio chunk directly"""
        try:
            # Send audio to Bedrock immediately
            self.stream_manager.add_audio_chunk(audio_data)
        except Exception as e:
            if self.is_streaming:
                print(f"Error processing input audio: {e}")
    
    async def play_output_audio(self):
        """Play audio responses from Nova Sonic"""
        while self.is_streaming:
            try:
                # Check for barge-in flag
                if self.stream_manager.barge_in:
                    # Clear the audio queue
                    while not self.stream_manager.audio_output_queue.empty():
                        try:
                            self.stream_manager.audio_output_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    self.stream_manager.barge_in = False
                    # Small sleep after clearing
                    await asyncio.sleep(0.05)
                    continue
                
                # Get audio data from the stream manager's queue
                audio_data = await asyncio.wait_for(
                    self.stream_manager.audio_output_queue.get(),
                    timeout=0.1
                )
                
                if audio_data and self.is_streaming:
                    # Write directly to the output stream in smaller chunks
                    chunk_size = CHUNK_SIZE  # Use the same chunk size as the stream
                    
                    # Write the audio data in chunks to avoid blocking too long
                    for i in range(0, len(audio_data), chunk_size):
                        if not self.is_streaming:
                            break
                        
                        end = min(i + chunk_size, len(audio_data))
                        chunk = audio_data[i:end]
                        
                        # Create a new function that captures the chunk by value
                        def write_chunk(data):
                            return self.output_stream.write(data)
                        
                        # Pass the chunk to the function
                        await asyncio.get_event_loop().run_in_executor(None, write_chunk, chunk)
                        
                        # Brief yield to allow other tasks to run
                        await asyncio.sleep(0.001)
                    
            except asyncio.TimeoutError:
                # No data available within timeout, just continue
                continue
            except Exception as e:
                if self.is_streaming:
                    print(f"Error playing output audio: {str(e)}")
                    import traceback
                    traceback.print_exc()
                await asyncio.sleep(0.05)
    
    async def start_streaming(self):
        """Start streaming audio."""
        if self.is_streaming:
            return
        
        print("Starting audio streaming. Speak into your microphone...")
        
        # Send audio content start event
        await time_it_async("send_audio_content_start_event", lambda : self.stream_manager.send_audio_content_start_event())
        
        self.is_streaming = True
        
        # Start the input stream if not already started
        if not self.input_stream.is_active():
            self.input_stream.start_stream()
        
        # Start processing tasks
        self.output_task = asyncio.create_task(self.play_output_audio())
        
        # Keep streaming while session state is active
        while st.session_state.streaming and self.is_streaming:
            await asyncio.sleep(0.1)  # Small delay to prevent CPU overuse
            
        # If we exit the loop, stop streaming
        print("inside start streaming")
        await self.stop_streaming()
    
    async def stop_streaming(self):
        """Stop streaming audio."""
        if not self.is_streaming:
            return
        print("inside stop streaming")
        self.is_streaming = False

        # Cancel the tasks
        tasks = []
        if hasattr(self, 'input_task') and not self.input_task.done():
            tasks.append(self.input_task)
        if hasattr(self, 'output_task') and not self.output_task.done():
            tasks.append(self.output_task)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        # Stop and close the streams
        if self.input_stream:
            if self.input_stream.is_active():
                self.input_stream.stop_stream()
            self.input_stream.close()
        if self.output_stream:
            if self.output_stream.is_active():
                self.output_stream.stop_stream()
            self.output_stream.close()
        if self.p:
            self.p.terminate()
        
        await self.stream_manager.close() 

async def nova_sonic_tool(report, user_email):
    stream_manager = BedrockStreamManager(report, user_email, model_id='amazon.nova-sonic-v1:0', region='us-east-1')

    # Create audio streamer
    audio_streamer = AudioStreamer(stream_manager)

    # Initialize the stream
    # await time_it_async("initialize_stream", stream_manager.initialize_stream)
    await stream_manager.initialize_stream()

    try:
        # This will run until the user presses Enter
        await audio_streamer.start_streaming()
    except Exception as e:
        if isinstance(e, KeyboardInterrupt):
            print("Interrupted by user - keyboard interrupt")
        else:
            print(f"Error: {e}")
    # Remove the finally block and handle cleanup in stop_streaming
    await audio_streamer.stop_streaming()
#---------------------------------------------------------------------------------------------------

st.set_page_config(page_title="Nova Sonic", page_icon="✈️", layout="wide")

#some CSS for frontend
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Rubik:ital,wght@0,300..900;1,300..900&display=swap" rel="stylesheet">
    <style>
    html, body, .stApp {
        color: #fff !important;
        background: linear-gradient(45deg,rgba(0, 0, 0, 1) 0%, rgba(35, 35, 35, 1) 50%, rgba(0, 0, 0, 1) 100%);
    }
    [class*="st-"] {
        font-family: 'Rubik', sans-serif !important;
    }
    .st-emotion-cache-zy6yx3 {
        width: 100%;
        padding: 5rem 5rem 10rem;
        max-width: initial;
        min-width: auto;
    }
    .stAppHeader {
        display: none;
    }
    .stButton > button {
        background: linear-gradient(45deg,rgba(0, 0, 0, 1) 0%, rgba(35, 35, 35, 1) 50%, rgba(0, 0, 0, 1) 100%);
        padding: 1px 10px;
        transition: all 0.3s ease;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stButton > button:hover {
        background: linear-gradient(45deg,rgba(0, 0, 0, 1) 0%, rgba(35, 35, 35, 1) 50%, rgba(0, 0, 0, 1) 100%);
        border-color: rgba(131, 145, 201, 0.8);
        color: white;
        box-shadow: 0 0 20px rgba(0, 255, 0, 0.2);
        transform: translateY(-2px);
    }
    /* Chat message text color white */
    .stChatMessage, div[data-testid^="stChatMessage-"] {
        background: rgb(81,86,100)
        color: #fff !important;
    }
    .custom-footer {
        position: fixed;
        right: 1.5rem;
        bottom: 1.2rem;
        background: none;
        color: white;
        text-align: right;
        font-size: 1.1em;
        font-family: 'Rubik', sans-serif;
        padding: 0;
        margin: 0;
        z-index: 9999;
        letter-spacing: 0.1em;
        box-shadow: none;
        opacity: 0.7;
        pointer-events: none;
    }
    .stAlert{
        color: #e0e6ed !important;
        border-radius: 0.7rem !important;
        box-shadow: 0 2px 16px rgba(0,0,0,0.10);
        padding: 0.7em 0.9em 0.7em 1em !important;
        margin-bottom: 1em !important;
        font-size: 0.97em !important;
        font-family: 'Rubik', sans-serif !important;
        opacity: 0;
        transform: translateY(-20px);
        animation: fadeSlideIn 0.8s cubic-bezier(0.23, 1, 0.32, 1) forwards;
    }
    .stAlert-info{
        background: #23272f !important;
    }
    .stChatMessage.st-emotion-cache-1c7y2kd {
        background: #232b3a;
        box-shadow: 0 2px 8px rgba(0,0,0,0.35);
        color: #e0e6ed;
    }
    .st-emotion-cache-janbn0 {
        display: flex;
        align-items: flex-start;
        gap: 0.5rem;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: rgb(153 193 170 / 50%);
    }
    .stChatMessage.st-emotion-cache-4oy321 {
        background: #17403a;
        box-shadow: 0 2px 8px rgba(0,0,0,0.35), 0 0 8px 2px #00ffd0a0;
        color: #e0e6ed;
        position: relative;
        font-weight: 600;
        overflow: hidden;
    }
    @keyframes tool-glow {
        from { filter: drop-shadow(0 0 4px #00ffd0); }
        to { filter: drop-shadow(0 0 12px #00ffd0); }
    }
    @keyframes fadeSlideIn {
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    .inspection-report-title-custom {
        font-size: 1.5em;
        font-weight: 700;
        margin-bottom: 0.2em;
        letter-spacing: 0.04em;
        color: #2eca6d;
        text-transform: uppercase;
        font-family: 'Rubik', sans-serif;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.4em;
    }
    .inspection-report-emoji {
        font-size: 1.05em;
        display: inline-block;
        animation: bounce 1.2s infinite;
        margin-right: 0.05em;
    }
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-7px); }
    }
    .inspection-report-content {
        font-size: 0.93em;
        line-height: 1.18;
        white-space: pre-line;
        margin-top: 0.05em;
    }
    .fancy-tool-message {
        background: #17403a;
        color: #e0e6ed;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(0,0,0,0.35), 0 0 8px 2px #00ffd0a0;
        padding: 0.7em 1em;
        margin: 0.2em 0;
        border-radius: 0.5em;
        position: relative;
        overflow: hidden;
        font-size: 1em;
        letter-spacing: 0.01em;
        display: flex;
        align-items: center;
        gap: 0.5em;
        animation: tool-glow 1.2s infinite alternate;
        margin-bottom: 1em;
        margin-right: 1em;
    }
    .fancy-tool-message b {
        color: #00ffd0;
        font-weight: 700;
    }
    .gradient-heading {
        display: inline-block;
        background: linear-gradient(135deg,rgba(63, 136, 181, 1) 0%, rgb(89, 203, 203) 50%, rgba(214, 255, 255, 1) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        color: transparent;
        text-shadow: none;
    }
    .gradient-ai {
        background: linear-gradient(135deg,rgba(98, 102, 112, 1) 0%, rgba(120, 141, 161, 1) 50%, rgba(255, 255, 255, 1) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        color: transparent;
        text-shadow: none;
    }
    .tagline {
        text-align: center;
        font-size: 1.15em;
        font-weight: 400;
        color: #b0b8c9;
        margin-top: -0.5em;
        margin-bottom: 1.5em;
        letter-spacing: 0.08em;
        font-family: 'Rubik', sans-serif;
        opacity: 0.92;
    }
    </style>
    <div class="main">
        <h1 style="text-align: center;">
            <span class="gradient-heading">FlySafe</span><span class="gradient-ai">.ai</span>
        </h1>
        <div class="tagline" style="font-style: italic">The intelligent assistant for modern maintenance operations with Vision, Voice and Action.</div>
    </div>
""", unsafe_allow_html=True)

# Create input fields
video_path = st.text_input("Enter Video Path (S3 URL):", placeholder="Enter your S3 URL here- s3://bucket-name/video.mp4", label_visibility="hidden")
user_email = st.text_input("Enter Your Email:", placeholder="your.email@example.com", label_visibility="hidden")

# Create a container for the conversation
alerts = st.empty()
alerts2 = st.empty()
alerts3 = st.empty()
footer = st.container()
chat_container = st.container(height=300, border=True)

if 'clicked' not in st.session_state:
    st.session_state.clicked = False
    st.session_state.stopped = False
    st.session_state.streaming = False

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []

def start_conv():
    st.session_state.clicked = True
    st.session_state.streaming = True

def stop_conv():
    if st.session_state.clicked:
        time.sleep(2)
        st.session_state.clicked = False
        st.session_state.stopped = True
        st.session_state.streaming = False

def clear_history():
    if st.session_state.stopped:
        st.session_state.clicked = False
        st.session_state.stopped = False

# the email where you would like to get the invoice
# user_email="xyz@gmail.com"

if not st.session_state.streaming and not st.session_state.clicked and not st.session_state.stopped:
    st.button('Start Conversation', on_click=start_conv)
if st.session_state.clicked:
    if not video_path:
        alerts.error("Please provide the video path.")
    else:
        # First, analyze the video
        alerts.badge("Analyzing video...")
        report = "On Analyzing, I see that: \n"
        report += _vrun_flight_inspector(video_path)
        # report += '''\n- ✅ Step 1: DONE – The outer body of the aircraft is inspected, showing no visible damage.\n- ✅ Step 2: DONE – The fan blades are inspected and appear to be in good condition.\n- ✅ Step 3: DONE – The exhaust nozzle is inspected and appears to be in good condition.\n- ❌ Step 4: NOT DONE – The oil access panel is not clearly visible or inspected in the video.\n- ❌ Step 5: NOT DONE – The wing lights are not clearly visible and appear to be damaged.'''
        alerts.empty()
        alerts.badge("Video Analysis Complete!", icon=":material/check:", color="green")
        initialize_session_state()
        alerts2.markdown(
            f'''\
            <div class="stAlert stAlert-info" data-testid="stAlert-info">
                <span class="inspection-report-title-custom"><span class="inspection-report-emoji">📝</span>Inspection Report<span class="inspection-report-emoji">📝</span></span>
                <span class="inspection-report-content">{report}</span>
            </div>
            ''',
            unsafe_allow_html=True
        )
        alerts3.badge("You can now speak on your mic to start the conversation.", icon=":material/mic:")
        with footer:
            st.button('Stop', on_click=stop_conv)
        footer.float("bottom: 0.5rem")
        asyncio.run(nova_sonic_tool(report.replace('\n',' '),user_email))

if not st.session_state.streaming and st.session_state.stopped:
    # Create transcript and display history
    conversation_history=[]
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        if role!="tool":
            conversation_history.append(f"{role}: {content}")
    print("\n1. Creating Transcript...")
    transcript_content = "\n".join(conversation_history)
    transcript_path = write_transcript(transcript_content)
    if not os.path.exists(transcript_path):
        raise Exception("Failed to create transcript file")
    print(f"✓ Transcript created successfully: {transcript_path}")
    alerts.badge("Transcript Created!", icon=":material/check:", color="green")
    st.button('Clear History', on_click=clear_history)
    with chat_container:
        for message in st.session_state.messages:
            icon = ""
            if message["role"]=="user":
                icon = "👤"
            elif message["role"]=="tool":
                icon =  "🛠️"
            else:
                icon = "✈️"
            with st.chat_message(message["role"],avatar=icon):
                st.write(message["content"])
    st.session_state.messages = []

st.markdown(
    '<div class="custom-footer">powered by AWS</div>',
    unsafe_allow_html=True
)
