import os
import time
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from ragas.run_config import RunConfig
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, ToolMessage
from RAG import create_multi_document_agent

load_dotenv()

def extract_contexts_from_state(messages) -> list[str]:
    """Helper to pull the retrieved text chunks out of the LangGraph tool messages."""
    contexts = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            contexts.append(msg.content)
    return contexts

def run_evaluation():
    print("Setting up Evaluation Pipeline for Photoelectric Effect Paper...")
    
    # 1. Setup the Grader Models (Using Gemini)
    eval_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    eval_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # 2. Define the Test Dataset based exactly on Einstein_Photoelectric_Effet.pdf
    eval_questions = [
        "What is the definition of the photoelectric effect?",
        "What is the stopping potential and what happens to the current when it is reached?",
        "State Einstein's photoelectric equation and define the variables.",
        "How does increasing the intensity of radiation affect the photocurrent according to quantum mechanics?",
        "What properties can be derived from the linear graph plot of stopping potential versus frequency?"
    ]
    
    ground_truths = [
        "The photoelectric effect is the process of emitting electrons from a metal surface when the surface is exposed to electromagnetic radiation of a sufficiently high frequency.",
        "Stopping potential is the value of the retarding voltage V when even the most energetic electron is not allowed to reach the cathode, causing the current to drop to zero.",
        "The equation is E = h*v = h*v_0 + T_max, where E is the total energy of the incident photon, v is the frequency of incident radiation, v_0 is the threshold frequency of the metal, and T_max is the maximum kinetic energy of the ejected electron.",
        "In quantum mechanics, increasing intensity increases the number of photons falling per second per unit area. This leads to an increased number of collisions with electrons and their subsequent ejection, which directly increases the photocurrent.",
        "When plotting stopping potential versus frequency, the intercept on the X-axis gives the threshold frequency, and the slope of the curve provides the value of h/e."
    ]
    
    # 3. Initialize your agent with the specific target PDF
    pdf_data = [
        {"path": "Einstein_Photoelectric_Effet.pdf", "name": "Einstein_Photoelectric_Effet.pdf"} 
    ]
    agent = create_multi_document_agent(pdf_data)

    # ---> ADD THIS BRAND NEW BLOCK <---
    print("⏳ PDF ingested! Waiting 60 seconds for the Gemini API quota to reset...")
    time.sleep(60)

    answers = []
    contexts = []

    # 4. Generate Predictions using your LangGraph Agent (WITH FREE-TIER THROTTLING)
    print("Running Agent against Photoelectric Effect Test Dataset...")
    for i, question in enumerate(eval_questions):
        print(f"Processing Question {i+1} of {len(eval_questions)}...")
        config = {"configurable": {"thread_id": f"photoelectric_test_{i}"}}
        
        result = agent.invoke({"messages": [HumanMessage(content=question)]}, config=config)
        messages = result['messages']
        
       # Extract the final answer cleanly
        raw_content = messages[-1].content
        if isinstance(raw_content, list):
            # Dig into the list and pull out only the actual text
            final_answer = " ".join([block.get("text", "") for block in raw_content if isinstance(block, dict) and block.get("type") == "text"])
        else:
            final_answer = str(raw_content)
        answers.append(final_answer)
        
        # Extract the exact chunks the hybrid retriever fed to the LLM
        retrieved_contexts = extract_contexts_from_state(messages)
        contexts.append(retrieved_contexts)
        
        # FREE TIER FIX 1: 35-second cooldown to avoid 429 RESOURCE_EXHAUSTED
        if i < len(eval_questions) - 1:
            print("⏳ Waiting 35 seconds to respect Gemini Free Tier API limits...")
            time.sleep(35)

   # 5. Format Data for RAGAS Evaluation (Updated for Newest RAGAS Version)
    data = {
        "user_input": eval_questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": ground_truths 
    }
    dataset = Dataset.from_dict(data)

    # FREE TIER FIX 2: Restrict RAGAS parallel processing
    safe_config = RunConfig(max_workers=1, max_retries=10)

    # 6. Compute RAGAS Metrics
    print("\nExecuting RAGAS Assessment Framework (This will take a few minutes...)")
    result = evaluate(
        dataset=dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
        llm=eval_llm,
        embeddings=eval_embeddings,
        run_config=safe_config  # Apply the throttle here
    )

    # 7. Output Final Scorecard
    print("\n=== SYSTEM PERFORMANCE SCORECARD ===")
    print(result)
    
    # Save detailed row-by-row matrix to CSV
    df = result.to_pandas()
    df.to_csv("photoelectric_rag_evaluation.csv", index=False)
    print("\nDetailed breakdown saved to 'photoelectric_rag_evaluation.csv'")

if __name__ == "__main__":
    run_evaluation()