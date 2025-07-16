import os
import json
import logging
from dotenv import load_dotenv
import boto3

# ——— Logging ———
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ——— Load AWS creds, region & bucket-owner from .env ———
# .env should contain:
#   AWS_ACCESS_KEY_ID=…
#   AWS_SECRET_ACCESS_KEY=…
#   AWS_REGION=us-east-1
#   BUCKET_OWNER=111122223333
load_dotenv()
AWS_REGION   = os.getenv("AWS_REGION", "us-east-1")
BUCKET_OWNER = os.getenv("BUCKET_OWNER")

# ——— Bedrock client ———
client = boto3.client("bedrock-runtime", region_name=AWS_REGION)

# ——— Model ID ———
MODEL_ID = "us.amazon.nova-pro-v1:0"

def build_inspection_payload(s3_uri: str) -> dict:
    system_prompt = {
        "text": (
            "You are a Flight Inspection Agent specializing in analyzing aircraft inspection videos.\n\n"
            "Your job is to carefully observe the visual content and determine whether each of the following inspection steps was performed, and whether they appear to be completed properly.\n\n"
            "For **each step**, report clearly whether the inspection was **DONE** or **NOT DONE**, based solely on visual evidence in the video. Maintain the exact sequence and numbering listed below.\n\n"
            "Inspection Steps:\n"
            "1. Outer‐body damage\n"
            "2. Fan blade condition\n"
            "3. Exhaust nozzle condition\n"
            "4. Oil access panel condition\n"
            "5. Wing Light condition\n\n"
            "Output Format:\n"
            "For each step, respond in this format:\n"
            " ✅/❌ Step 1: DONE or NOT DONE – [short comment]\n"
            " ✅/❌ Step 2: DONE or NOT DONE – [short comment]\n"
            " ✅/❌ Step 3: DONE or NOT DONE – [short comment]\n"
            " ✅/❌ Step 4: DONE or NOT DONE – [short comment]\n\n"
            "Use ✅ if the step was done and ❌ if the step was not done."
            "In the short comment, be concise, objective, and only include observations that can be inferred visually. DO NOT assume anything not clearly visible in the video."
        )
    }

    user_content = [
        {
            "video": {
                "format": "mp4",
                "source": {
                    "s3Location": {
                        "uri": s3_uri,
                    }
                }
            }
        },
        {
            "text": "Please perform the flight inspection as described above."
        }
    ]

    return {
        "schemaVersion": "messages-v1",
        "system": [system_prompt],
        "messages": [
            {"role": "user", "content": user_content}
        ],
        "inferenceConfig": {
            "maxTokens": 500,
            "temperature": 0.2,
            "topP": 0.9,
            "topK": 40
        }
    }

def invoke_flight_inspector(s3_uri: str):
    payload = build_inspection_payload(s3_uri)
    logger.info("Invoking Amazon Nova with S3 video source…")

    response = client.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(payload)
    )

    result = json.loads(response["body"].read())
    report = result["output"]["message"]["content"][0]["text"]

    print("\n===== Flight Inspection Report =====\n")
    print(report)
    print("\n====================================\n")

    return report

def _vrun_flight_inspector(uri):
    # our S3 URI here:
    # my_video_s3 = ""
    return invoke_flight_inspector(uri)

def main():
    path = ""
    video_result = _vrun_flight_inspector(path)
    print(video_result)

if __name__=="__main__":
    main()