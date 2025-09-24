from rag_pipeline import RAGPipeline
from create_sample_data import create_sample_data
import os

def main():
    print("=== RAG Pipeline Demo ===")
    
    # Create sample documents
    print("\n1. Creating sample documents...")
    docs_dir = create_sample_data()  # Also update this function call
    
    # Initialize RAG pipeline
    print("\n2. Initializing RAG pipeline...")
    rag = RAGPipeline()
    
    # Build knowledge base
    print("\n3. Building knowledge base...")
    num_chunks = rag.build_knowledge_base(docs_dir)
    print(f"Knowledge base built with {num_chunks} chunks")
    
    # Demo queries
    print("\n4. Running demo queries...")
    
    demo_questions = [
        "What are the main types of machine learning?",
        "What are the effects of climate change?",
        "What are the main sources of renewable energy?",
        "What are some key milestones in space exploration?"
    ]
    
    for question in demo_questions:
        print(f"\n--- Question: {question} ---")
        
        # Get grounded response
        result = rag.query(question)
        print("\nGrounded Response:")
        print(result['response'])
        
        print("\nSources used:")
        for i, source in enumerate(result['sources'], 1):
            print(f"{i}. {source['title']} (Score: {source['similarity_score']:.3f})")
        
        print("-" * 80)
    
    # Comparison demo
    print("\n5. Comparing grounded vs ungrounded responses...")
    test_question = "What is the most important greenhouse gas?"
    
    comparison = rag.compare_responses(test_question)
    print(f"\nQuestion: {test_question}")
    print(f"\nGrounded Response: {comparison['grounded_response']}")
    print(f"\nUngrounded Response: {comparison['ungrounded_response']}")
    
    print("\n=== Demo Complete ===")
    print("\nTo run the Streamlit web interface, use:")
    print("streamlit run demo_app.py")

if __name__ == "__main__":
    main()