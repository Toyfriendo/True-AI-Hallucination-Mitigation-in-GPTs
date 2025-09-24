import os

def create_sample_data():  # Changed from create_sample_documents()
    """Create sample documents for testing the RAG pipeline."""
    
    sample_docs_dir = "./sample_documents"
    os.makedirs(sample_docs_dir, exist_ok=True)
    
    # Sample document 1: AI and Machine Learning
    doc1 = """
Artificial Intelligence and Machine Learning

Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines that can perform tasks that typically require human intelligence. Machine Learning (ML) is a subset of AI that focuses on the development of algorithms that can learn and improve from experience without being explicitly programmed.

There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning. Supervised learning uses labeled data to train models, unsupervised learning finds patterns in unlabeled data, and reinforcement learning learns through interaction with an environment.

Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data. It has been particularly successful in areas such as image recognition, natural language processing, and speech recognition.

Common applications of AI and ML include recommendation systems, autonomous vehicles, medical diagnosis, fraud detection, and virtual assistants. The field continues to evolve rapidly with new techniques and applications being developed regularly.
"""
    
    # Sample document 2: Climate Change
    doc2 = """
Climate Change and Global Warming

Climate change refers to long-term shifts in global or regional climate patterns. Global warming is the long-term heating of Earth's climate system observed since the mid-20th century due to increased levels of greenhouse gases produced by human activities.

The primary greenhouse gases include carbon dioxide (CO2), methane (CH4), nitrous oxide (N2O), and fluorinated gases. Carbon dioxide is the most significant greenhouse gas, primarily released through burning fossil fuels, deforestation, and industrial processes.

Effects of climate change include rising sea levels, changing precipitation patterns, more frequent extreme weather events, and shifts in ecosystems. These changes can impact agriculture, water resources, human health, and biodiversity.

Mitigation strategies include transitioning to renewable energy sources, improving energy efficiency, protecting and restoring forests, and developing carbon capture technologies. Adaptation measures involve building resilient infrastructure and developing climate-resistant agricultural practices.
"""
    
    # Sample document 3: Renewable Energy
    doc3 = """
Renewable Energy Sources

Renewable energy comes from natural sources that are constantly replenished. The main types of renewable energy include solar, wind, hydroelectric, geothermal, and biomass energy.

Solar energy harnesses sunlight using photovoltaic cells or solar thermal systems. Wind energy uses turbines to convert wind movement into electricity. Hydroelectric power generates electricity from flowing water, typically through dams.

Geothermal energy utilizes heat from the Earth's core, while biomass energy comes from organic materials like wood, agricultural crops, and waste. Each renewable source has its advantages and challenges in terms of cost, efficiency, and environmental impact.

The adoption of renewable energy is crucial for reducing greenhouse gas emissions and combating climate change. Many countries are investing heavily in renewable energy infrastructure and setting ambitious targets for clean energy transition.

Challenges include energy storage, grid integration, and the intermittent nature of some renewable sources. However, technological advances are continuously improving the efficiency and cost-effectiveness of renewable energy systems.
"""
    
    # Sample document 4: Space Exploration
    doc4 = """
Space Exploration and Technology

Space exploration involves the investigation of outer space through the use of astronomy and space technology. It has led to numerous scientific discoveries and technological innovations that benefit life on Earth.

Key milestones in space exploration include the first artificial satellite (Sputnik 1), the first human in space (Yuri Gagarin), the Moon landing (Apollo 11), and the development of space stations like the International Space Station (ISS).

Modern space exploration involves both government agencies like NASA, ESA, and private companies like SpaceX, Blue Origin, and Virgin Galactic. These organizations are working on missions to Mars, asteroid mining, space tourism, and establishing permanent human settlements beyond Earth.

Space technology has practical applications including satellite communications, GPS navigation, weather forecasting, and Earth observation for climate monitoring and disaster management. Many everyday technologies originated from space research, including memory foam, water purification systems, and advanced materials.

Future goals include returning humans to the Moon, establishing a permanent lunar base, sending crewed missions to Mars, and potentially exploring the outer planets and their moons.
"""
    
    # Write documents to files
    documents = [
        ("ai_machine_learning.txt", doc1),
        ("climate_change.txt", doc2),
        ("renewable_energy.txt", doc3),
        ("space_exploration.txt", doc4)
    ]
    
    for filename, content in documents:
        filepath = os.path.join(sample_docs_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    
    print(f"Created {len(documents)} sample documents in {sample_docs_dir}")
    return sample_docs_dir

if __name__ == "__main__":
    create_sample_data()  # Changed from create_sample_data()