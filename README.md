# 🤖 NVIDIA NIM Document QA Application

An intelligent document question-answering application powered by NVIDIA's NIM (NVIDIA Inference Microservices) and LangChain. This application allows you to upload PDF documents and ask questions about their content, leveraging NVIDIA's state-of-the-art language models.

## ✨ Features

- 📄 PDF document processing and embedding
- 🔍 Semantic search capabilities
- 💡 Intelligent question answering
- 🤔 Transparent thought process visualization
- 🎯 Relevant context display

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- NVIDIA NIM API key
- Virtual environment manager (venv/conda)

### Setup Instructions

1. **Get Your NVIDIA NIM API Key**
   - Visit [NVIDIA AI Playground](https://www.nvidia.com/en-us/ai-on-rtx/chat-with-rtx-generative-ai/)
   - Sign up for a NVIDIA account
   - Navigate to API section to generate your API key
   - Copy the API key for later use

2. **Set Up Virtual Environment**
   ```bash
   # Create a new virtual environment
   python -m venv nim-env
   
   # Activate the environment
   # For Windows:
   nim-env\Scripts\activate
   # For Unix/MacOS:
   source nim-env/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**
   - Create a `.env` file in the project root
   - Add your NVIDIA API key:
     ```
     NVIDIA_API_KEY=your_api_key_here
     ```

5. **Run the Application**
   ```bash
   streamlit run app.py
   ```

## 📖 Usage Guide

1. **Start Document Embedding**
   - Place your PDF documents in the `./pdfs` directory
   - Click the "Start Doc Embedding" button
   - Wait for the "Document Embedding Completed" success message

2. **Ask Questions**
   - Enter your question in the text input field
   - The application will:
     - Search through embedded documents
     - Generate a detailed response
     - Show the model's thought process (expandable)
     - Display relevant document contexts

## 🔧 Troubleshooting

- Ensure all PDFs are readable and not password-protected
- Check if the NVIDIA API key is correctly set in the `.env` file
- Verify that all dependencies are installed correctly
- Make sure the `pdfs` directory exists and contains your documents

## 📝 Note

This application uses NVIDIA's DeepSeek-R1 model for generating responses. The quality and accuracy of answers depend on the clarity of questions and the content of provided documents.

## ⚠️ Requirements

See `requirements.txt` for a complete list of dependencies. Key packages include:
- streamlit
- langchain
- langchain-nvidia-ai-endpoints
- python-dotenv
- faiss-cpu

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
