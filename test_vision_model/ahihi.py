import os
import io
import boto3
from PyPDF2 import PdfReader
import sagemaker
import streamlit as st

# Set up Streamlit page configuration
st.set_page_config(
    page_title="UniSum",
    page_icon=":🦄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS styles
st.markdown("""
    <style>
    .block-container {padding-top: 1rem;padding-bottom: 0rem;padding-left: 5rem;padding-right: 5rem}
    h1 {text-align: center;}
    MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


# Create necessary clients
target_region = os.environ.get("AWS_REGION")
bedrock_client = boto3.client("bedrock-runtime")
s3 = boto3.client(service_name='s3', region_name=target_region)

# Define Model ID
LITE_MODEL_ID = "us.amazon.nova-lite-v1:0"


# Summarize transcript
def summarize_transcript(data):

    system = [{ "text": """
        You are a responsible AI assistant that summarizes key information from the medical consultation transcript between the clinician and the patient.
        Summarize the given text in a concise, clear, and professional tone.
        Ensure the report is formatted in rich text, with headings, bullet points, and highlighted only the reason for the visit, history of illnesses, assessment, and treatment plan.
        """ }]

    messages = [
        {"role": "user", "content": [{"text": data}]},
    ]

    inf_params = {"maxTokens": 3000, "topP": 0.1, "temperature": 0.3}

    additionalModelRequestFields = {
        "inferenceConfig": {
             "topK": 20
        }
    }

    model_response = bedrock_client.converse(
        modelId=LITE_MODEL_ID,
        messages=messages,
        system=system,
        inferenceConfig=inf_params,
        toolConfig=additionalModelRequestFields,
    )

    response = model_response["output"]["message"]["content"][0]["text"]
    return response


# Read S3 object
def read_s3_pdf(bucket_name, file_name):

    try:
        response = s3.get_object(Bucket=bucket_name, Key=file_name)
        pdf_content = response['Body'].read()
        pdf_file = io.BytesIO(pdf_content)
        pdf_reader = PdfReader(pdf_file)
        return pdf_reader
    except Exception as e:
        print(f"Error decoding PDF: {e}")
        return None


if __name__ == "__main__":

    st.title(":rainbow[UniCare Generative AI]")
    st.header(":stethoscope: :rainbow[UniSum]")
    st.subheader("_Medical Summary Assistant powered by Amazon Nova_", divider='rainbow')
    st.write("Our clinicians are busy! Please help us summarize medical transcripts so that we can focus on improving patient experience.")

    sess = sagemaker.Session()
    s3_bucket = sess.default_bucket()

    uploaded_file = st.file_uploader(label="Upload a transcript PDF here.")

    if uploaded_file is not None:
        s3.upload_fileobj(uploaded_file, s3_bucket, uploaded_file.name)
        file_path = f's3://{s3_bucket}/{uploaded_file.name}'
        st.write(f'Successfully uploaded to {file_path}')

        st.divider()

        pdf_reader = read_s3_pdf(s3_bucket, uploaded_file.name)

        if pdf_reader:
            number_of_pages = len(pdf_reader.pages)
            input_text = []

            for p in range(number_of_pages):
                page = pdf_reader.pages[p]
                text = page.extract_text()
                input_text.append(text)

                full_response = summarize_transcript(str(input_text))

            st.write(full_response)
