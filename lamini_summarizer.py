import tempfile
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch
import base64
import os
from docx import Document
import io
import PyPDF2

class DocumentSummarizationApp:
    def __init__(self):
        # Create a temporary directory for offloading
        self.temp_dir = tempfile.TemporaryDirectory()  # This will be cleaned up automatically
        self.data_folder = "./data"  # Keep the data folder persistent if needed
        self.checkpoint = "MBZUAI/LaMini-Flan-T5-248M"

        # Create necessary directories for data if needed
        self._create_directories()

        # Load tokenizer and model with offloading enabled, using the temp directory
        self.tokenizer = T5Tokenizer.from_pretrained(self.checkpoint, legacy=False)
        self.base_model = T5ForConditionalGeneration.from_pretrained(
            self.checkpoint, device_map='auto', torch_dtype=torch.float32, offload_folder=self.temp_dir.name
        )

    def _create_directories(self):
        """Create necessary persistent directories for the application."""
        try:
            os.makedirs(self.data_folder, exist_ok=True)
        except Exception as e:
            st.error(f"Error creating data directories: {e}")

    def load_docx(self, file):
        """Load content from a DOCX file."""
        try:
            doc = Document(file)
            return "\n".join(paragraph.text for paragraph in doc.paragraphs)
        except Exception as e:
            st.error(f"Error loading DOCX file: {e}")
            return ""

    def load_pdf(self, file):
        """Extract text from a PDF file."""
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = []
            for page in pdf_reader.pages:
                text.append(page.extract_text())
            return "\n".join(text)
        except Exception as e:
            st.error(f"Error processing PDF file: {e}")
            return ""

    def file_preprocessing(self, file):
        """Preprocess the uploaded file to extract text."""
        try:
            if file.name.endswith('.pdf'):
                return self.load_pdf(file)
            elif file.name.endswith('.docx'):
                return self.load_docx(file)
            else:
                raise ValueError("Unsupported file format")
        except Exception as e:
            st.error(f"Error processing file: {e}")
            return None

    def llm_pipeline(self, file):
        """Generate a summary using the loaded model."""
        try:
            pipe_sum = pipeline(
                'summarization',
                model=self.base_model,
                tokenizer=self.tokenizer,
                max_length=500,
                min_length=50
            )
            input_text = self.file_preprocessing(file)
            
            if input_text is None:
                return None
            
            result = pipe_sum(input_text)
            return result[0]['summary_text']
        except Exception as e:
            st.error(f"Error generating summary: {e}")
            return None

    @staticmethod
    @st.cache_data
    def display_pdf(file):
        """Display a PDF file in the Streamlit app."""
        try:
            pdf_bytes = file.read()
            base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error displaying PDF: {e}")

    def display_docx(self, file):
        """Display the content of a DOCX file in the Streamlit app."""
        try:
            doc_content = self.load_docx(file)
            st.text_area("Document Content", value=doc_content, height=400)
        except Exception as e:
            st.error(f"Error displaying DOCX content: {e}")

    def run(self):
        """Run the Streamlit app."""
        st.set_page_config(layout="wide", page_title="Document Summarization App", page_icon="✨")

        st.markdown(
            "<h1 style='text-align: center; color: #FF4B4B; font-size: 3em;'>📄✨ Document Summarization App ✨📄</h1>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<h4 style='text-align: center; color: #4B9CD3;'>🤖 Powered by LaMini-Flan-T5-248M 🚀</h4>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<p style='text-align: center;'>Upload your document and let AI generate a concise summary!</p>",
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader("🔽 **Drag and drop your PDF or DOCX file here!**", type=['pdf', 'docx'])

        if uploaded_file is not None:
            if st.button("✨ Summarize 🚀"):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(
                        "<h4 style='color: #4B9CD3;'>📂 Uploaded File Preview:</h4>",
                        unsafe_allow_html=True
                    )
                    if uploaded_file.name.endswith('.pdf'):
                        self.display_pdf(uploaded_file)
                    elif uploaded_file.name.endswith('.docx'):
                        self.display_docx(uploaded_file)

                with col2:
                    st.markdown(
                        "<h4 style='color: #28A745;'>📝 Summarized Text:</h4>",
                        unsafe_allow_html=True
                    )
                    summary = self.llm_pipeline(uploaded_file)
                    
                    if summary is None:
                        st.warning("⚠️ Failed to generate summary. Please try again.")
                    else:
                        st.text_area("✨ Summarized Content:", value=summary, height=400)

            st.markdown("<hr>", unsafe_allow_html=True)

# Main execution point of the app
if __name__ == "__main__":
    app = DocumentSummarizationApp()
    app.run()

    # Cleanup: Automatically deletes the temporary directory when the app finishes running
    app.temp_dir.cleanup()
