# local_testing_webhooks.py

# using flask for local testing
from flask import Flask, request
import json
import logging
from lambda_function import webhook, create_application
from config_local import LocalTestingConfig

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
async def local_webhook():
    """Handle webhook requests for local testing"""
    try:
        # Print incoming request details
        print("\n=== Incoming Webhook Request ===")
        print(f"Headers: {dict(request.headers)}")
        print(f"Data: {json.dumps(request.json, indent=2)}")
        print("==============================\n")
            
        if not request.json:
            return 'No data received', 400
            
        # Create event dict to match Lambda format
        event = {
            'body': json.dumps(request.json)
        }
        
        # Use existing webhook function
        application = create_application(custom_config=LocalTestingConfig())
        result = await webhook(event, None, application)
        
        # Print response
        print(f"Response: {result['body']} (Status: {result['statusCode']})")
        
        return result['body'], result['statusCode']
        
    except Exception as e:
        print(f"Error processing webhook: {e}")
        return f'Error: {str(e)}', 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)


# explanation:
# This code creates a simple Flask web server that listens for incoming POST requests at the '/webhook' endpoint.
# it then calls lambda_function.py, namely the webhook function, which is the main function that processes the incoming messages.

