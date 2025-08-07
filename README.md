# AI Policy Drafting System (ADA)

An AI-powered system for drafting policies based on client profiles and existing documents. This system leverages Large Language Models (LLMs) and vector databases to generate customized policy documents by analyzing client requirements and historical policy data.

## 🚀 Features

- **AI-Powered Policy Generation**: Automatically draft policies based on client profiles and requirements
- **Document Processing**: Extract and analyze content from existing policy documents
- **Vector Search**: Semantic search capabilities using Azure AI Search vector database
- **Multi-Model Support**: Integration with OpenAI models.
- **RESTful API**: Complete API for policy management and user operations
- **Azure Integration**: Support for Azure Blob Storage
- **Production Ready**: Docker containerization with development and production configurations

## 🏗️ Architecture

The system follows a microservices architecture with the following components:

- **FastAPI Backend**: RESTful API server
- **Azure Services**: Cloud storage and search capabilities


## 🛠️ Technology Stack

- **Backend**: FastAPI (Python)
- **Database**: Azure AI Search (Vector Database)
- **LLM Integration**: LangChain, OpenAI
- **Cloud Services**: Azure Blob Storage, Azure Cognitive Search
- **Containerization**: Docker & Docker Compose

## 🚦 Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)

### Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/hozRifaigt/ada.git
   cd ada
   ```

2. **Create environment file**:
   ```bash
   mkdir env
   touch env/.env.dev
   ```

3. **Start the development environment**:
   ```bash
   docker-compose up
   ```

4. **Access the API**:
   - API Documentation: http://localhost:8080/docs
   - Health Check: http://localhost:8080/healthcheck


## 🔧 Configuration

### Docker Configuration

The system uses two Docker Compose configurations:

- **Development** (`docker-compose.yml`): Hot-reload enabled, local development
- **Production** (`docker-compose.prod.yml`): Optimized for production use

## 📊 API Endpoints

### Policy Management
- `POST /api/v1/policy/draft` - Generate a new policy draft
- `GET /api/v1/policy/{policy_id}` - Retrieve a specific policy
- `PUT /api/v1/policy/{policy_id}` - Update a policy
- `DELETE /api/v1/policy/{policy_id}` - Delete a policy

### Document Processing
- `POST /api/v1/processing/index` - Index documents for search
- `POST /api/v1/processing/process` - Process documents through pipeline
- `GET /api/v1/processing/status/{job_id}` - Check processing status

### User Management
- Not Yet Implemented

### Administration
- `POST /api/v1/admin/reset-index` - run pre-processing pipeline

## 🔍 Document Processing Pipeline

The system processes documents through the following stages:

1. **File Loading**: Support for DOCX files only for now
2. **Text Extraction**: Extract and clean text content
3. **Chunking**: Split documents into manageable chunks
4. **Embedding**: Generate vector embeddings using LLMs
5. **Indexing**: Store embeddings in Azure AI Search


## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
