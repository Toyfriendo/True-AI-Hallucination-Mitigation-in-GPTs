from openai import OpenAI
from typing import List, Dict
from config import Config

class LLMInterface:
    """Interface for interacting with Language Models."""
    
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or Config.OPENAI_API_KEY
        self.model = model or Config.LLM_MODEL
        
        # Initialize OpenAI client with minimal configuration
        try:
            self.client = OpenAI(
                api_key=self.api_key,
                timeout=30.0
            )
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
            # Fallback initialization
            self.client = OpenAI(api_key=self.api_key)
    
    def construct_prompt(self, query: str, retrieved_chunks: List[Dict]) -> str:
        """Construct a prompt with retrieved context."""
        context_parts = []
        
        for i, chunk in enumerate(retrieved_chunks, 1):
            context_parts.append(
                f"Source {i} ({chunk['title']}):\n{chunk['content']}\n"
            )
        
        context = "\n".join(context_parts)
        
        prompt = f"""Context:
{context}

Question: {query}

Please answer the question using ONLY the information provided in the context above. If the context doesn't contain enough information to answer the question, please say "I don't have enough information in the provided context to answer this question."

Answer:"""
        
        return prompt
    
    def generate_response(self, prompt: str) -> str:
        """Generate response using the LLM."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based only on the provided context. Do not use external knowledge."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=Config.MAX_TOKENS,
                temperature=Config.TEMPERATURE
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def generate_grounded_response(self, query: str, retrieved_chunks: List[Dict]) -> Dict:
        """Generate a grounded response with sources."""
        prompt = self.construct_prompt(query, retrieved_chunks)
        response = self.generate_response(prompt)
        
        return {
            'query': query,
            'response': response,
            'sources': retrieved_chunks,
            'prompt_used': prompt
        }