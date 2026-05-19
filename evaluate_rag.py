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
    "What type of light is required to eject electrons from an alkali metal?",
    "In the experimental setup, which plate acts as the anode?",
    "Why is there a current even when the retarding potential is applied, before it reaches the stopping potential?",
    "What happens to the current when the retarding potential exceeds the value V_0?",
    "Is there a time delay between irradiating the surface and the ejection of electrons?",
    "At a fixed frequency above the threshold, how does increasing the intensity of incident light affect the photocurrent?",
    "According to the laws of photoelectric emission, what happens if the incident frequency is below the threshold frequency?",
    "What determines the kinetic energy of the ejected electrons above the threshold frequency?",
    "Which classical theory fails to explain the photoelectric effect?",
    "How did Planck describe the emission and propagation of radiation?",
    "How did Einstein modify Planck's idea to explain the photoelectric effect?",
    "State Einstein's photoelectric equation as presented in the text.",
    "How does wave mechanics define the intensity of radiation?",
    "How does quantum mechanics define the intensity of radiation?",
    "Why does an increase in photon frequency increase the maximum kinetic energy of the ejected electrons?",
    "Define saturation current.",
    "Why does the stopping potential remain the same if the intensity is increased at a fixed frequency?",
    "In the linear equation V = (h/e)v - (h/e)v_0, what does the slope of the curve represent?",
    "In a plot of stopping potential versus frequency, what does the X-axis intercept represent?"
    ]

    ground_truths = [
        "Photoelectric effect is the process of emitting the electrons from the a metal surface when the metal surface is exposed to an electromagnetic radiation of sufficiently high frequency. [cite: 2]",
        "Ultraviolet light is required in the case of ejection of electrons from an alkali metal. [cite: 3]",
        "The metal plate whose surface is to be irradiated acts as the anode. [cite: 21]",
        "Some of the photoelectrons that emerge from the radiated surface have sufficient energy to reach the cathode despite its negative polarity and they constitute the current. [cite: 22]",
        "When V exceeds a certain value V_0 no further electrons are able to strike the cathode and the current drops to zero. [cite: 24]",
        "There is no time lag between the irradiation of the surface and the ejection of the electrons. [cite: 26]",
        "At a particular fixed frequency of incident radiation the rate of the emission of photo electrons i.e. the photocurrent increases with increase in the intensity of the incident light. [cite: 27]",
        "Photo electric effect does not occur at frequency less than threshold frequency. [cite: 28]",
        "At the frequency above the threshold frequency, the kinetic energy of the ejected electrons depends only on the frequency of the exposed radiation and not on its intensity. [cite: 29]",
        "The photoelectric effect cannot be explained on the basis of electromagnetic theory. [cite: 31, 32]",
        "Plank assumed that while the radiation is emitted continuously as little bursts of energy called quanta but propagated continuously in space as electromagnetic waves. [cite: 40]",
        "Einstein proposed that light not only was emitted as quanta at a time but also propagated as individual quanta, sufficiently small to be absorbed by the electron. [cite: 41, 42, 43, 44]",
        "The equation is E(=hv) = hv_0 + T_max. Here E is the total energy of the photon incident on the metallic surface, v is the frequency of the incident radiation, v_0 is the threshold frequency of the metal and T_max is the maximum kinetic energy. [cite: 49, 50]",
        "In wave mechanics the intensity of radiation is defined as the total continuous energy falling normal to a surface per second per unit area. [cite: 51]",
        "In quantum mechanics intensity should be considered to be related to the number of photons falling per second per unit area. [cite: 52]",
        "When frequency is increased the energy of individual photons increases. The work function is fixed. Hence, the any increase in the energy of individual photons results in increase in maximum kinetic energy of the ejected electrons. [cite: 55, 56]",
        "For a given intensity when all the ejected electrons are pulled by the cathode there are no more electrons left to reach the cathode. After this even if V is increased the current does not increase. This is the saturation current. [cite: 59, 60]",
        "If the frequency of the incident radiation is fixed T_max will not change. Hence, the stopping potential will remain the same even if the intensity is increased or decreased. [cite: 68, 69, 70, 71]",
        "The slope of the curve will give h/e. [cite: 88]",
        "The intercept on the X-axis will give the threshold frequency. [cite: 88]"
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