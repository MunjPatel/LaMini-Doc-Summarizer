## LaMini Document Summarizer

LaMini Document Summarizer is a Streamlit app designed to summarize text documents using the [LaMini-Flan-T5-248M model](https://huggingface.co/MBZUAI/LaMini-Flan-T5-248M). It allows users to upload various text-based documents and generate concise summaries, making it easier to digest large amounts of information quickly.

You can test the app directly by going to [LaMini Document Summarizer Production App](https://lamini-doc-summarizer.streamlit.app/), where the model is running live.

## Why LaMini Document Summarizer?

The LaMini Document Summarizer provides a simple and intuitive interface for summarizing long documents. Whether you're conducting research, reviewing lengthy reports, or just want to save time by skimming through the key points, this app helps you generate accurate summaries efficiently.

## Example Use Cases

- **Business Reports**: Summarize complex reports to extract the main points and insights, saving time in reviewing lengthy documents.
- **Research Papers**: Condense academic papers or articles into manageable summaries for quick understanding or referencing.
- **Meeting Notes**: Generate concise meeting summaries from detailed notes, providing a clear overview of the discussion points and action items.
- **Legal Documents**: Summarize contracts, agreements, or case files for easier review and analysis.

## About LaMini-Flan-T5-248M

The [LaMini-Flan-T5-248M](https://huggingface.co/MBZUAI/LaMini-Flan-T5-248M) model is a transformer-based text-to-text model fine-tuned on multiple natural language tasks. It is based on Google's T5 (Text-to-Text Transfer Transformer) architecture and fine-tuned using the FLAN instruction-tuning method.

### Why Use LaMini-Flan-T5-248M?

1. **Efficient Summarization**: LaMini-Flan-T5-248M is optimized for generating concise summaries from large documents. It understands the structure of natural language and can accurately condense long pieces of text.
2. **Small Footprint**: The model is lightweight (248 million parameters) compared to larger models, making it suitable for real-time inference on moderately sized servers without significant resource requirements.
3. **Wide Range of Use Cases**: The model has been trained on a diverse range of text-to-text tasks, making it versatile for various NLP applications, including summarization, translation, and question-answering.

### Advantages

- **Fast Inference**: Due to its smaller size compared to other large-scale models, LaMini-Flan-T5-248M can generate summaries quickly.
- **Instruction-Tuned**: The model has been fine-tuned with instructions, which means it's better suited for tasks requiring human-readable outputs.
- **Cost-Effective**: It requires less computational power, making it more accessible for smaller organizations or individual users who want to run models locally.

### Disadvantages

- **Limited Context**: While the model performs well for most summarization tasks, its ability to handle extremely large contexts may be limited compared to larger models.
- **Generalization Limits**: Although the model is trained on a variety of text, its smaller size can sometimes lead to less accurate generalizations for very domain-specific texts.
- **Reduced Performance on Complex Text**: For highly complex or technical documents, the model may produce summaries that miss some nuances compared to larger models trained specifically on those domains.

## Features

- **Upload Documents**: Upload files in formats such as `.pdf`, and `.docx` for summarization.
- **Customizable Summaries**: Adjust summary length and customize parameters to suit your preferences.
- **Real-time Summarization**: Generate quick summaries as you upload and process documents.
- **Download Summaries**: Export the generated summaries for further use or reference.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/MunjPatel/LaMini-Document-Summarizer.git
    ```

2. Change the working directory:

    ```bash
    cd LaMini-Document-Summarizer
    ```

3. Create and activate a virtual environment:

    ```bash
    conda create --prefix ./env python=3.9 -y
    conda activate ./env
    ```

4. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit app:

    ```bash
    streamlit run lamini_summarizer.py
    ```

2. Open the browser and navigate to the local server link provided (usually `http://localhost:8501`).

3. Upload a document (TXT, PDF, or DOCX) via the interface.

4. Customize the summary length or parameters if desired.

5. Click on **Generate Summary** to process the document.

6. View the summarized text on the screen, and download it if needed.


## Error Handling

- **File Format Errors**: If the uploaded file is not in a supported format (PDF, DOCX), the app will return an error message.
- **Large Files**: If the document is too large, the app may slow down or time out. Consider using smaller files or breaking the document into chunks.
- **Backend Connectivity Issues**: The app connects to the model's backend for processing. If there are server issues, an error message will appear.

## Performance Optimization

The app uses a temporary file structure to handle document processing and model offloading. Once the document is processed, the temporary files are removed to ensure optimal storage usage without affecting performance.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
