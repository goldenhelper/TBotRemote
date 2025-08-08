import boto3
import json
import os

if __name__ == "__main__":
    param_name = os.getenv("ALLOWED_MODEL_LIMITS_PARAM")
    region = os.getenv("AWS_REGION")

    if param_name is None:
        print("ALLOWED_MODEL_LIMITS_PARAM is not set")
        exit(1)
    if region is None:
        print("AWS_REGION is not set")
        exit(1)

    provider = input("Provider (gemini/claude/openai): ").strip()
    model_name = input("Model name: ").strip()
    allowed_model_limit = int(input("Allowed model limit: ").strip())

    ssm = boto3.client("ssm", region_name=region)

    try:
        resp = ssm.get_parameter(Name=param_name)
        data = json.loads(resp["Parameter"]["Value"])
    except ssm.exceptions.ParameterNotFound:
        data = {}

    data.setdefault(provider, {})
    data[provider][model_name] = allowed_model_limit

    ssm.put_parameter(
        Name=param_name,
        Value=json.dumps(data),
        Type="String",
        Overwrite=True,
    )

    print(f"Updated {param_name} with {provider}.{model_name}={allowed_model_limit}")
    
    