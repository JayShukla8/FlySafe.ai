# FlySafe.ai ✈️

**FlySafe.ai** is a personal AI-powered inspection agent that uses **Amazon Nova Pro** and **Amazon Nova Sonic** to assist with **video analysis**, **natural language interaction**, and **tool-based automation**.

Here's the video [demo](https://youtu.be/d1ixI9FTxuo).

## Features ✨

### Video Analysis (Nova Pro)
- Uses the **Amazon Nova Pro** model to analyze inspection videos.
- Automatically generates a detailed **inspection report**.

### Conversational Agent (Nova Sonic)
- Uses the **Amazon Nova Sonic** model to:
  - Answer **user questions** about the inspection.
  - Perform **tool-based actions** using integrated APIs.

## Tool-Calling Functionalities 🔧

### 1. Google Shopping Search 
- Triggered when the user asks the agent to look for any **replacement parts or products**.
- Uses the **SERP API** to fetch and display the **top 3 results** from Google Shopping.

### 2. Invoice Generation 
- Based on the selected product from the search results.
- Automatically generates a PDF invoice with the relevant info

### 3. Email Sending 
- Uses the **SendGrid API**.
- Sends the **generated invoice PDF** and the **conversation transcript** to the user’s email on confirmation

## Usage 📦

Before you begin, make sure you have the following:

- **Python 3.13+**
- API keys: (check .env file for more info)

Now follow these steps to use the project-

1. **Clone the repository**
   ```bash
   git clone https://github.com/JayShukla8/FlySafe.ai.git
   cd FlySafe.ai

2. **Create a virtual env and activate it**
   ```bash
    python -m venv .venv

    # Activate the virtual environment

    # On Windows:
    .venv\Scripts\activate

    # On macOS/Linux:
    # source .venv/bin/

3. **With the virtual environment activated, install the required packages:**

    ```bash
    python -m pip install -r requirements.txt --force-reinstall

4. **Configure AWS credentials:**

    The application uses environment variables for AWS authentication. Set these before running the application:

    ```bash
    export AWS_ACCESS_KEY_ID="your-access-key"
    export AWS_SECRET_ACCESS_KEY="your-secret-key"
    export AWS_DEFAULT_REGION="us-east-1"

5. **Specify the user Email**

    Please specify the email which will send the notifications at [line](https://github.com/JayShukla8/FlySafe.ai/blob/4201cb9869412b9d20097ac37bd953d00c348921/src/utils/tools/sendemail.py#L28).



6. **Start**

    Start the interface via the command
    ```bash
    streamlit run src/workflow/interface_streamlit.py

## Contribution 🌱

This project will remain open for contribution, please feel free to help.

## Acknowledgments 🤜🤛

Thanks to Pavan, Shubham and Manan for helping throughtout.

I have made changes to [this](https://github.com/aws-samples/amazon-nova-samples/blob/main/speech-to-speech/sample-codes/console-python/nova_sonic_tool_use.py) code for running the Nova Sonic model. Refer to its Readme file for more details.


