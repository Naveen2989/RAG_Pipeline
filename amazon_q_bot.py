import boto3
import json
import os

def lambda_handler(event, context):
    client = boto3.client("qbusiness")

    try:
        response = client.create_application(
            displayName="TravelAssistantApp",
            description="Amazon Q application created via Lambda",
            roleArn= "arn:aws:iam::740595473561:role/service-role/amazon_q_func-role-f5bvuecw",
            identityType="AWS_SSO"  # or "AWS_SSO"
        )

        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Amazon Q Application created",
                "applicationId": response["applicationId"],
                "arn": response["arn"]
            })
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps(str(e))
        }
