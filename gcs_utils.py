# gcs_utils.py
import os
import json
import logging
import streamlit as st
from google.cloud import storage
from google.oauth2 import service_account

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
# Try to get bucket name from environment variables (useful for local testing)
# or Streamlit secrets (preferred for deployment)
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")

# --- Initialization ---
@st.cache_resource # Cache the client for efficiency
def get_gcs_client():
    """Initializes and returns a Google Cloud Storage client."""
    try:
        # Try loading credentials from Streamlit secrets first
        if "gcp_service_account" in st.secrets:
            creds_dict = st.secrets["gcp_service_account"]
            credentials = service_account.Credentials.from_service_account_info(creds_dict)
            logger.info("GCS client initialized using Streamlit secrets.")
            print('done gcs auth')
            return storage.Client(credentials=credentials)
        else:
            logger.info("Attempting GCS client initialization using Application Default Credentials.")
            client = storage.Client()
            logger.info("GCS client initialized using Application Default Credentials.")
            return client
    except Exception as e:
        logger.error(f"Failed to initialize GCS client: {e}", exc_info=True)
        st.error(f"Error initializing Google Cloud Storage client: {e}. Ensure credentials are set correctly (Streamlit secrets or ADC).")
        return None

def get_bucket(client):
    """Gets the GCS bucket object, handling potential errors."""
    global GCS_BUCKET_NAME # Allow modification if needed
    if not GCS_BUCKET_NAME and "gcs_bucket_name" in st.secrets:
         GCS_BUCKET_NAME = st.secrets["gcs_bucket_name"]

    if not client:
        logger.error("GCS client is not available.")
        return None
    if not GCS_BUCKET_NAME:
        logger.error("GCS_BUCKET_NAME is not set in environment variables or Streamlit secrets.")
        st.error("Google Cloud Storage bucket name is not configured. Set it in Streamlit secrets.")
        return None
    try:
        bucket = client.bucket(GCS_BUCKET_NAME)
        if not bucket.exists():
             logger.warning(f"Bucket '{GCS_BUCKET_NAME}' does not exist or insufficient permissions.")
             # Optionally try to create it, but requires storage.buckets.create permission
             # try:
             #     client.create_bucket(GCS_BUCKET_NAME)
             #     logger.info(f"Bucket '{GCS_BUCKET_NAME}' created.")
             #     bucket = client.bucket(GCS_BUCKET_NAME)
             # except Exception as create_e:
             #     logger.error(f"Failed to create bucket '{GCS_BUCKET_NAME}': {create_e}")
             #     st.error(f"GCS Bucket '{GCS_BUCKET_NAME}' not found and could not be created.")
             #     return None
             st.error(f"GCS Bucket '{GCS_BUCKET_NAME}' not found or access denied.")
             return None
        return bucket
    except Exception as e:
        logger.error(f"Error accessing bucket '{GCS_BUCKET_NAME}': {e}", exc_info=True)
        st.error(f"Error accessing GCS bucket '{GCS_BUCKET_NAME}': {e}")
        return None

# --- GCS Operations ---

def blob_exists(bucket, blob_name):
    """Checks if a blob exists in the bucket."""
    if not bucket:
        return False
    try:
        blob = bucket.blob(blob_name)
        return blob.exists()
    except Exception as e:
        logger.error(f"Error checking existence of blob '{blob_name}': {e}", exc_info=True)
        return False

def upload_blob(bucket, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    if not bucket:
        logger.error(f"Cannot upload '{source_file_name}', bucket not available.")
        return False
    try:
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        logger.info(f"File {source_file_name} uploaded to {destination_blob_name}.")
        return True
    except Exception as e:
        logger.error(f"Error uploading file '{source_file_name}' to blob '{destination_blob_name}': {e}", exc_info=True)
        return False

def download_blob(bucket, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    if not bucket:
        logger.error(f"Cannot download '{source_blob_name}', bucket not available.")
        return False
    try:
        blob = bucket.blob(source_blob_name)
        if not blob.exists():
            logger.warning(f"Blob {source_blob_name} does not exist in bucket {bucket.name}.")
            return False
        blob.download_to_filename(destination_file_name)
        logger.info(f"Blob {source_blob_name} downloaded to {destination_file_name}.")
        return True
    except Exception as e:
        logger.error(f"Error downloading blob '{source_blob_name}' to '{destination_file_name}': {e}", exc_info=True)
        return False

def download_json_blob(bucket, blob_name):
    """Downloads and parses a JSON blob from the bucket."""
    if not bucket:
        logger.error(f"Cannot download JSON '{blob_name}', bucket not available.")
        return None
    try:
        blob = bucket.blob(blob_name)
        if not blob.exists():
            logger.info(f"JSON blob '{blob_name}' not found in bucket {bucket.name}.")
            return {} # Return empty dict if not found, consistent with original load_results
        json_data = blob.download_as_string()
        logger.info(f"JSON blob '{blob_name}' downloaded.")
        return json.loads(json_data)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from blob '{blob_name}': {e}", exc_info=True)
        return None # Indicate error
    except Exception as e:
        logger.error(f"Error downloading JSON blob '{blob_name}': {e}", exc_info=True)
        return None # Indicate error

def upload_json_blob(bucket, data, blob_name):
    """Uploads a dictionary as a JSON blob to the bucket."""
    if not bucket:
        logger.error(f"Cannot upload JSON '{blob_name}', bucket not available.")
        return False
    try:
        blob = bucket.blob(blob_name)
        json_data = json.dumps(data, indent=2)
        blob.upload_from_string(json_data, content_type='application/json')
        logger.info(f"Dictionary uploaded as JSON blob '{blob_name}'.")
        return True
    except Exception as e:
        logger.error(f"Error uploading JSON blob '{blob_name}': {e}", exc_info=True)
        return False

