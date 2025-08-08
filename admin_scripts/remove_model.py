import boto3
import json
import os

if __name__ == "__main__":
    param_name = os.getenv("ALLOWED_MODEL_LIMITS_PARAM", "/telegram-bot/allowed_model_limits")
    region = os.getenv("AWS_REGION", "us-east-1")

    provider = input("Provider (gemini/claude/openai): ").strip()
    model_name = input("Model name to remove: ").strip()

    ssm = boto3.client("ssm", region_name=region)

    try:
        resp = ssm.get_parameter(Name=param_name)
        data = json.loads(resp["Parameter"]["Value"])
    except ssm.exceptions.ParameterNotFound:
        print("Config parameter not found; nothing to remove.")
        raise SystemExit(0)

    if provider in data and model_name in data[provider]:
        del data[provider][model_name]
        if not data[provider]:
            del data[provider]

        ssm.put_parameter(
            Name=param_name,
            Value=json.dumps(data),
            Type="String",
            Overwrite=True,
        )
        print(f"Removed {provider}.{model_name} from {param_name}")
    else:
        print("Model not found; nothing changed.")

