# S3_Agent.py
# Natural language -> AWS S3 operations using Ollama LLM

import os
import json
import boto3
import streamlit as st
from dotenv import load_dotenv
import ollama
from typing import Any, Dict, List, Optional, Tuple

# Load environment variables
load_dotenv()


# ==================== S3 Client ====================
class S3Manager:
    def __init__(
        self,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: str = "us-east-1",
    ):
        """Initialize S3 client with AWS credentials."""
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=region_name,
        )
        self.s3_resource = boto3.resource(
            "s3",
            aws_access_key_id=aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=region_name,
        )

    def list_buckets(self) -> List[Dict[str, Any]]:
        """List all S3 buckets."""
        try:
            response = self.s3_client.list_buckets()
            buckets = [
                {
                    "name": b["Name"],
                    "creation_date": str(b["CreationDate"]),
                }
                for b in response.get("Buckets", [])
            ]
            return buckets
        except Exception as e:
            raise RuntimeError(f"Error listing buckets: {str(e)}")

    def list_objects(
        self, bucket_name: str, prefix: str = "", max_keys: int = 100
    ) -> List[Dict[str, Any]]:
        """List objects in an S3 bucket."""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=bucket_name, Prefix=prefix, MaxKeys=max_keys
            )
            objects = [
                {
                    "key": obj["Key"],
                    "size": obj["Size"],
                    "last_modified": str(obj["LastModified"]),
                    "storage_class": obj.get("StorageClass", "STANDARD"),
                }
                for obj in response.get("Contents", [])
            ]
            return objects
        except Exception as e:
            raise RuntimeError(f"Error listing objects in {bucket_name}: {str(e)}")

    def get_bucket_size(self, bucket_name: str) -> Dict[str, Any]:
        """Calculate total size of bucket."""
        try:
            bucket = self.s3_resource.Bucket(bucket_name)
            total_size = sum(obj.size for obj in bucket.objects.all())
            object_count = sum(1 for _ in bucket.objects.all())
            return {
                "bucket_name": bucket_name,
                "total_size_bytes": total_size,
                "total_size_gb": round(total_size / (1024**3), 2),
                "object_count": object_count,
            }
        except Exception as e:
            raise RuntimeError(f"Error getting bucket size: {str(e)}")

    def upload_file(self, bucket_name: str, file_path: str, object_name: str) -> Dict[str, Any]:
        """Upload a file to S3."""
        try:
            self.s3_client.upload_file(file_path, bucket_name, object_name)
            return {
                "status": "success",
                "bucket": bucket_name,
                "object_name": object_name,
                "file_path": file_path,
            }
        except Exception as e:
            raise RuntimeError(f"Error uploading file: {str(e)}")

    def download_file(self, bucket_name: str, object_name: str, file_path: str) -> Dict[str, Any]:
        """Download a file from S3."""
        try:
            self.s3_client.download_file(bucket_name, object_name, file_path)
            return {
                "status": "success",
                "bucket": bucket_name,
                "object_name": object_name,
                "downloaded_to": file_path,
            }
        except Exception as e:
            raise RuntimeError(f"Error downloading file: {str(e)}")

    def delete_object(self, bucket_name: str, object_name: str) -> Dict[str, Any]:
        """Delete an object from S3."""
        try:
            self.s3_client.delete_object(Bucket=bucket_name, Key=object_name)
            return {
                "status": "success",
                "bucket": bucket_name,
                "object_name": object_name,
                "action": "deleted",
            }
        except Exception as e:
            raise RuntimeError(f"Error deleting object: {str(e)}")

    def get_object_metadata(self, bucket_name: str, object_name: str) -> Dict[str, Any]:
        """Get metadata of an S3 object."""
        try:
            response = self.s3_client.head_object(Bucket=bucket_name, Key=object_name)
            return {
                "object_name": object_name,
                "size_bytes": response.get("ContentLength", 0),
                "last_modified": str(response.get("LastModified", "")),
                "content_type": response.get("ContentType", "unknown"),
                "storage_class": response.get("StorageClass", "STANDARD"),
                "etag": response.get("ETag", ""),
            }
        except Exception as e:
            raise RuntimeError(f"Error getting object metadata: {str(e)}")

    def search_objects(self, bucket_name: str, search_prefix: str) -> List[Dict[str, Any]]:
        """Search for objects with a specific prefix."""
        try:
            objects = self.list_objects(bucket_name, prefix=search_prefix, max_keys=50)
            return objects
        except Exception as e:
            raise RuntimeError(f"Error searching objects: {str(e)}")


# ==================== LLM Functions ====================
def call_llm_ollama(prompt: str, model: str, temperature: float = 0.0) -> str:
    """Call Ollama LLM with a prompt."""
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful AWS S3 assistant."},
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": temperature,
            },
        )
        return response["message"]["content"].strip()
    except Exception as e:
        raise RuntimeError(f"Ollama API error: {str(e)}")


def analyze_user_intent(
    user_query: str,
    s3_context: Dict[str, Any],
    model: str,
) -> Dict[str, Any]:
    """Analyze user intent and suggest S3 operations."""
    context = json.dumps(s3_context, ensure_ascii=False, indent=2)

    prompt = f"""
You are an AWS S3 assistant. Analyze the user's request and suggest the most appropriate S3 operations.

Current S3 Context:
{context}

User Request:
{user_query}

Provide your response in JSON format with the following structure:
{{
    "understood_intent": "brief description of what the user wants",
    "suggested_operation": "list_buckets | list_objects | get_bucket_size | search_objects | get_metadata",
    "parameters": {{"bucket_name": "...", "prefix": "..."}},
    "explanation": "explain what you're about to do"
}}

Only respond with valid JSON.
"""
    response = call_llm_ollama(prompt=prompt, model=model, temperature=0.1)
    
    try:
        # Extract JSON from response
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            json_str = response[json_start:json_end]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    return {
        "understood_intent": user_query,
        "suggested_operation": "list_buckets",
        "parameters": {},
        "explanation": "Unable to parse intent, defaulting to listing buckets",
    }


def generate_summary(
    operation_result: Dict[str, Any],
    user_query: str,
    model: str,
) -> str:
    """Generate a natural language summary of S3 operation results."""
    result_json = json.dumps(operation_result, ensure_ascii=False, indent=2)

    prompt = f"""
You are an AWS S3 assistant. Summarize the following S3 operation results in a clear, concise way.

User Query:
{user_query}

Operation Results:
{result_json}

Provide a natural language summary that is easy to understand.
"""
    return call_llm_ollama(prompt=prompt, model=model, temperature=0.3)


# ==================== Main Agent ====================
def run_s3_agent(
    user_query: str,
    model: str,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    region_name: str = "us-east-1",
) -> Tuple[str, Dict[str, Any]]:
    """Run S3 agent to process user query."""
    
    # Initialize S3 manager
    s3_manager = S3Manager(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name,
    )

    # Get current S3 context
    try:
        buckets = s3_manager.list_buckets()
        s3_context = {"buckets": buckets}
    except Exception as e:
        s3_context = {"error": str(e)}

    # Analyze user intent
    intent_analysis = analyze_user_intent(
        user_query=user_query,
        s3_context=s3_context,
        model=model,
    )

    # Execute suggested operation
    operation_result = {}
    operation_type = intent_analysis.get("suggested_operation", "")
    parameters = intent_analysis.get("parameters", {})

    try:
        if operation_type == "list_buckets":
            operation_result = {"buckets": s3_manager.list_buckets()}

        elif operation_type == "list_objects":
            bucket = parameters.get("bucket_name")
            prefix = parameters.get("prefix", "")
            if bucket:
                operation_result = {
                    "bucket": bucket,
                    "objects": s3_manager.list_objects(bucket, prefix=prefix),
                }

        elif operation_type == "get_bucket_size":
            bucket = parameters.get("bucket_name")
            if bucket:
                operation_result = s3_manager.get_bucket_size(bucket)

        elif operation_type == "search_objects":
            bucket = parameters.get("bucket_name")
            prefix = parameters.get("prefix", "")
            if bucket:
                operation_result = {
                    "search_prefix": prefix,
                    "results": s3_manager.search_objects(bucket, prefix),
                }

        elif operation_type == "get_metadata":
            bucket = parameters.get("bucket_name")
            object_name = parameters.get("object_name")
            if bucket and object_name:
                operation_result = s3_manager.get_object_metadata(bucket, object_name)

    except Exception as e:
        operation_result = {"error": str(e)}

    # Generate summary
    summary = generate_summary(
        operation_result=operation_result,
        user_query=user_query,
        model=model,
    )

    return summary, {
        "intent": intent_analysis,
        "operation_result": operation_result,
    }


# ==================== Streamlit UI ====================
def main():
    st.set_page_config(
        page_title="S3 Agent",
        page_icon="🪣",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🪣 AWS S3 Natural Language Agent")
    st.markdown("Interact with AWS S3 using natural language and AI.")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        st.subheader("🔐 AWS Credentials")
        aws_access_key = st.text_input(
            "AWS Access Key ID",
            value=os.getenv("AWS_ACCESS_KEY_ID", ""),
            type="password",
            help="Your AWS access key (from environment or enter here)",
        )
        aws_secret_key = st.text_input(
            "AWS Secret Access Key",
            value=os.getenv("AWS_SECRET_ACCESS_KEY", ""),
            type="password",
            help="Your AWS secret key (from environment or enter here)",
        )
        aws_region = st.text_input(
            "AWS Region",
            value=os.getenv("AWS_REGION", "us-east-1"),
            help="AWS region (e.g., us-east-1, eu-west-1)",
        )

        st.divider()
        st.subheader("🤖 LLM Options")
        model = st.selectbox(
            "Ollama Model",
            options=["llama2", "llama3", "llama3.1", "mistral", "neural-chat", "starling"],
            index=0,
        )

        st.divider()
        st.markdown("### 📋 Information")
        st.info(
            "This agent understands natural language queries about S3 and performs the appropriate operations. "
            "It uses Ollama for intelligent intent analysis."
        )
        
        st.markdown("### 💡 Examples")
        st.code("""
- "List all my S3 buckets"
- "What's in my data-bucket?"
- "How much storage am I using?"
- "Find all files starting with 'backup'"
- "Get details about logs/error.txt"
        """)

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        user_query = st.text_area(
            "Ask a question about your S3 storage:",
            height=100,
            placeholder="e.g., 'List all my buckets' or 'What files are in my data bucket?'",
        )

    with col2:
        st.markdown("### Actions")
        submit_button = st.button("🚀 Execute", use_container_width=True, type="primary")

    # Execute query
    if submit_button and user_query:
        if not aws_access_key or not aws_secret_key:
            st.error("❌ Please enter AWS credentials in the configuration panel.")
            return

        try:
            with st.spinner("🔄 Processing your request..."):
                summary, results = run_s3_agent(
                    user_query=user_query,
                    model=model,
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key,
                    region_name=aws_region,
                )

            # Display results in tabs
            tab1, tab2, tab3 = st.tabs(["📝 Summary", "🔍 Intent Analysis", "📊 Raw Results"])

            with tab1:
                st.markdown("### AI-Powered Summary")
                st.markdown(summary)

            with tab2:
                st.markdown("### Intent Analysis")
                intent = results.get("intent", {})
                st.write(f"**Understood Intent:** {intent.get('understood_intent', 'N/A')}")
                st.write(f"**Operation:** {intent.get('suggested_operation', 'N/A')}")
                st.write(f"**Explanation:** {intent.get('explanation', 'N/A')}")

            with tab3:
                st.markdown("### Operation Results")
                operation_result = results.get("operation_result", {})
                st.json(operation_result)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)

    elif submit_button:
        st.warning("⚠️ Please enter a question first.")


if __name__ == "__main__":
    main()
