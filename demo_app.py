import streamlit as st
from rag_pipeline import RAGPipeline
import os
import tempfile
from pathlib import Path

def main():
    st.title("RAG Pipeline Demo: Reducing Hallucinations")
    st.write("This demo shows how RAG grounds responses in real documents to reduce hallucinations.")
    
    # Initialize RAG pipeline
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = RAGPipeline()
    
    rag = st.session_state.rag_pipeline
    
    # Sidebar for knowledge base management
    st.sidebar.header("Knowledge Base Management")
    
    # Load existing knowledge base
    if st.sidebar.button("Load Existing Knowledge Base"):
        if rag.load_knowledge_base():
            st.sidebar.success("Knowledge base loaded successfully!")
        else:
            st.sidebar.error("No existing knowledge base found")
    
    # File upload section
    st.sidebar.subheader("📁 Upload Your Documents")
    uploaded_files = st.sidebar.file_uploader(
        "Upload your documents here",
        type=['txt', 'pdf', 'docx', 'md'],
        accept_multiple_files=True,
        help="Upload text files (.txt), PDFs (.pdf), Word documents (.docx), or Markdown files (.md)"
    )
    
    if uploaded_files:
        st.sidebar.write(f"📄 {len(uploaded_files)} file(s) uploaded")
        
        if st.sidebar.button("🚀 Build Knowledge Base from Uploaded Files"):
            with st.spinner("Processing uploaded files..."):
                # Create temporary directory for uploaded files
                temp_dir = tempfile.mkdtemp()
                
                try:
                    # Save uploaded files to temporary directory
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(temp_dir, uploaded_file.name)
                        
                        # Handle different file types
                        if uploaded_file.type == "text/plain" or uploaded_file.name.endswith('.txt'):
                            # Text files
                            with open(file_path, 'w', encoding='utf-8') as f:
                                f.write(uploaded_file.getvalue().decode('utf-8'))
                        elif uploaded_file.name.endswith('.md'):
                            # Markdown files
                            with open(file_path, 'w', encoding='utf-8') as f:
                                f.write(uploaded_file.getvalue().decode('utf-8'))
                        else:
                            # For PDF and DOCX, save as binary and convert later
                            with open(file_path, 'wb') as f:
                                f.write(uploaded_file.getvalue())
                    
                    # Build knowledge base from uploaded files
                    num_chunks = rag.build_knowledge_base(temp_dir)
                    st.sidebar.success(f"✅ Knowledge base built with {num_chunks} chunks from your uploaded files!")
                    
                except Exception as e:
                    st.sidebar.error(f"❌ Error processing files: {str(e)}")
                
                finally:
                    # Clean up temporary directory
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Build new knowledge base from directory
    st.sidebar.subheader("📂 Or Use Document Directory")
    docs_directory = st.sidebar.text_input(
        "Documents Directory Path", 
        value="./sample_documents",
        help="Path to directory containing your documents"
    )
    
    if st.sidebar.button("Build Knowledge Base from Directory"):
        if os.path.exists(docs_directory):
            with st.spinner("Building knowledge base..."):
                num_chunks = rag.build_knowledge_base(docs_directory)
            st.sidebar.success(f"✅ Knowledge base built with {num_chunks} chunks!")
        else:
            st.sidebar.error("❌ Directory not found")
    
    # Main interface
    st.header("💬 Ask Questions")
    
    # Show knowledge base status
    if rag.is_indexed:
        st.success("✅ Knowledge base is ready! You can now ask questions.")
    else:
        st.warning("⚠️ Please upload documents or build a knowledge base first.")
    
    question = st.text_input("Enter your question:", placeholder="e.g., What is machine learning?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎯 Get Grounded Response"):
            if question and rag.is_indexed:
                with st.spinner("Generating response..."):
                    result = rag.query(question)
                
                if 'error' in result:
                    st.error(result['error'])
                else:
                    st.subheader("🎯 Grounded Response")
                    st.write(result['response'])
                    
                    st.subheader("📚 Sources Used")
                    for i, source in enumerate(result['sources'], 1):
                        with st.expander(f"Source {i}: {source['title']} (Score: {source['similarity_score']:.3f})"):
                            st.write(source['content'])
            elif not rag.is_indexed:
                st.error("Please build or load a knowledge base first")
            else:
                st.error("Please enter a question")
    
    with col2:
        if st.button("⚖️ Compare Responses"):
            if question and rag.is_indexed:
                with st.spinner("Comparing responses..."):
                    comparison = rag.compare_responses(question)
                
                st.subheader("⚖️ Response Comparison")
                
                st.write("**🎯 Grounded Response (with context):**")
                st.info(comparison['grounded_response'])
                
                st.write("**🚫 Ungrounded Response (without context):**")
                st.warning(comparison['ungrounded_response'])
                
                st.write("**📚 Sources for Grounded Response:**")
                for source in comparison['sources']:
                    st.caption(f"• {source['title']} (Score: {source['similarity_score']:.3f})")
            elif not rag.is_indexed:
                st.error("Please build or load a knowledge base first")
            else:
                st.error("Please enter a question")

if __name__ == "__main__":
    main()