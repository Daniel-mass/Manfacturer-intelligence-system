#src/ai_reasoning_agent.py
import json
import os
import sys
import ollama

# Ensure pathing for local results
KB_PATH = "models/octo_knowledge_base.json"
REPORT_PATH = "models/ai_final_report.txt"

class ManufacturingAnalyst:
    def __init__(self, model="llama3.2:latest"):
        self.model = model
        # Using a structured prompt similar to your YAML-based project
        self.system_prompt = """
        ROLE: Senior Industrial AI Lead & Maintenance Foreman.
        CONTEXT: You are reviewing a Digital Twin (NASA) and a Causal Optimizer (AI4I).
        STYLE: Brutally honest, technical, data-driven. No fluff.
        
        GOAL: Analyze the provided JSON data and explain the ROI and Safety of the AI intervention.
        """

    def generate_briefing(self):
        if not os.path.exists(KB_PATH):
            return "❌ Error: Run octo_aggregator.py first to generate data."

        with open(KB_PATH, "r") as f:
            data = json.load(f)

        # Extracting the hard truths from your modules
        nasa = data["nasa_analytics"]
        ai4i = data["ai4i_analytics"]
        
        # Constructing the user message
        user_message = f"""
        ENGINEERING DATA:
        - NASA Simulation: Legacy failed at {nasa['simulation']['legacy_fail']}, AI-Managed at {nasa['simulation']['ai_managed_fail']}.
        - Performance Gain: {nasa['simulation']['gain']}.
        - Prognostic Accuracy: R2={nasa['r2_score']}, Causal Impact={nasa['causal_impact']}.
        - Optimization: {ai4i['total_anomalies']} anomalies detected. Safe Setpoint: {ai4i['optimized_setpoint']}.

        TASKS:
        1. Explain how the causal impact coefficient (-23.78) physically manifests as the 93.1% life extension.
        2. Defend the R2 score of 0.848. Is it high enough for a production line?
        3. Explain why we throttled to {ai4i['optimized_setpoint']} (RPM/Torque) given the 185min tool wear.
        """

        print(f"🤖 Reasoning via {self.model}...")
        response = ollama.chat(model=self.model, messages=[
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': user_message}
        ])

        report = response['message']['content']
        
        # Save for the Dashboard
        os.makedirs("models", exist_ok=True)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(report)
            
        return report

if __name__ == "__main__":
    analyst = ManufacturingAnalyst()
    print("\n" + "="*70)
    print("📋 SENIOR MAINTENANCE BRIEFING")
    print("="*70)
    print(analyst.generate_briefing())
    print("="*70)