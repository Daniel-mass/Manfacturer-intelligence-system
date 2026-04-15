# src/octo_researcher.py
import sys
import os
import json
from datetime import datetime
from dotenv import load_dotenv
import ollama
import time

load_dotenv()

# ── OctoTools path setup ───────────────────────────────────────────────────────
OCTO_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "octotools")
sys.path.insert(0, OCTO_PATH)

# Use ONLY the tools you confirmed are in your 'tools' folder
from octotools.tools.arxiv_paper_searcher.tool import ArXiv_Paper_Searcher_Tool
from octotools.tools.wikipedia_knowledge_searcher.tool import Wikipedia_Knowledge_Searcher_Tool

# ── paths ──────────────────────────────────────────────────────────────────────
MODEL_DIR          = "models"
RESEARCH_OUT       = f"{MODEL_DIR}/research_context.json"
SYSTEM_INTEL_PATH  = f"{MODEL_DIR}/system_intelligence.json"

# ── LLM for summarization ──────────────────────────────────────────────────────
SUMMARY_MODEL = "llama3.2:latest"

# ── Manufacturing research queries ────────────────────────────────────────────
RESEARCH_QUERIES = {
    "nasa_rul_benchmark": {
        "arxiv":   "remaining useful life prediction turbofan CMAPSS deep learning",
        "wiki":    "Remaining useful life",
        "context": "NASA RUL model performance benchmarking"
    },
    "predictive_maintenance": {
        "arxiv":   "predictive maintenance machine learning manufacturing industry",
        "wiki":    "Predictive maintenance",
        "context": "Predictive maintenance industry standards"
    },
    "tool_wear_standard": {
        "arxiv":   "tool wear detection machine learning classification",
        "wiki":    "Tool wear",
        "context": "Tool wear detection benchmarks"
    },
    "causal_inference_manufacturing": {
        "arxiv":   "causal inference industrial predictive maintenance DoWhy",
        "wiki":    "Causal inference",
        "context": "Causal analysis in manufacturing"
    },
    "fleet_maintenance_optimization": {
        "arxiv":   "agent based simulation maintenance optimization manufacturing",
        "wiki":    "Fleet management",
        "context": "Fleet maintenance simulation and optimization"
    }
}

def _summarize_to_lines(raw_text: str, context: str, max_lines: int = 3) -> str:
    if not raw_text or len(raw_text.strip()) < 50:
        return "Technical data being synthesized from academic repositories."
    
    prompt = f"""
    You are a Senior Manufacturing Research Lead. 
    Context: {context}
    
    Analyze these ACADEMIC results. Extract SPECIFIC benchmarks, RMSE, F1-scores, or methodologies.
    
    Source Text:
    {raw_text[:2500]}
    
    Output exactly {max_lines} concise technical sentences. Focus on facts, not definitions.
    """
    try:
        response = ollama.chat(
            model=SUMMARY_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "num_predict": 250}
        )
        return response["message"]["content"].strip()
    except Exception as e:
        return f"Summary unavailable: {e}"

def _format_arxiv_results(results) -> str:
    if not results: return ""
    if isinstance(results, list):
        return "\n".join([f"Title: {r.get('title')}\nAbstract: {r.get('abstract', r.get('summary', ''))[:400]}" for r in results[:4]])
    return str(results)[:2000]

def _format_wiki_results(results) -> str:
    if not results: return ""
    if isinstance(results, dict):
        return results.get("summary", results.get("content", str(results)))[:1500]
    return str(results)[:1500]

class ManufacturingResearcher:
    def __init__(self):
        print("🔭 Initializing Academic-First Manufacturing Researcher...")
        try:
            self.arxiv = ArXiv_Paper_Searcher_Tool()
            print("  ✅ ArxivPaperSearcherTool loaded")
            self.wiki = Wikipedia_Knowledge_Searcher_Tool()
            print("  ✅ WikipediaKnowledgeSearcherTool loaded")
        except Exception as e:
            print(f"  ❌ Tool Load Failed: {e}")

    def research_topic(self, topic_key: str, query_config: dict) -> dict:
        print(f"\n  📚 Researching: {query_config['context']}")
        findings = {"topic": topic_key, "context": query_config["context"], "sources": {}}

        # ── Arxiv ─────────────────────────────────────────────────────────────
        try:
            print(f"    📄 Arxiv: {query_config['arxiv'][:50]}...")
            raw = self.arxiv.execute(query=query_config["arxiv"])
            raw_text = _format_arxiv_results(raw)
            summary = _summarize_to_lines(raw_text, query_config["context"], max_lines=2)
            findings["sources"]["arxiv"] = {"summary": summary}
            print("    ✅ Arxiv: Done")
        except Exception as e:
            print(f"    ⚠️ Arxiv error: {e}")

        # ── Wikipedia ─────────────────────────────────────────────────────────
        try:
            print(f"    📖 Wikipedia: {query_config['wiki']}...")
            raw = self.wiki.execute(query=query_config["wiki"])
            raw_text = _format_wiki_results(raw)
            summary = _summarize_to_lines(raw_text, query_config["context"], max_lines=1)
            findings["sources"]["wikipedia"] = {"summary": summary}
            print("    ✅ Wikipedia: Done")
        except Exception as e:
            print(f"    ⚠️ Wiki error: {e}")

        # ── Compile ───────────────────────────────────────────────────────────
        all_text = " ".join([v["summary"] for v in findings["sources"].values() if "summary" in v])
        findings["compiled_finding"] = _summarize_to_lines(all_text, query_config["context"], max_lines=3)
        return findings

    def run(self):
        print("\n🚀 Starting Manufacturing Research Pipeline")
        print("=" * 60)
        
        results = {"generated_at": datetime.now().isoformat(), "topics": {}}
        for key, config in RESEARCH_QUERIES.items():
            results["topics"][key] = self.research_topic(key, config)

        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(RESEARCH_OUT, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        
        print(f"\n✅ Pipeline Complete. Data saved to: {RESEARCH_OUT}")

if __name__ == "__main__":
    ManufacturingResearcher().run()