"""
Use the trained model for predictions - Complete Fixed Version
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

class MutualFundAdvisor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Check available models
        model_paths = [
            ("models/enhanced_model", True),  # (path, is_peft)
            ("models/fine_tuned", True),
            ("microsoft/phi-2", False)  # Fallback to base model
        ]
        
        for path, is_peft in model_paths:
            if Path(path).exists() or not is_peft:
                self.model_path = path
                self.is_peft = is_peft
                break
        
        print(f"Loading model from {self.model_path} (PEFT: {self.is_peft})")
        
        if self.is_peft and Path(self.model_path).exists():
            # Load PEFT model
            try:
                # Load base model first
                base_model_name = "microsoft/phi-2"
                self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
                
                # Add padding token if needed
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Load base model
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True,
                )
                
                # Load PEFT weights
                self.model = PeftModel.from_pretrained(base_model, self.model_path)
                self.model.eval()
                print("✅ PEFT model loaded successfully")
                
            except Exception as e:
                print(f"⚠️ Error loading PEFT model: {e}")
                print("Falling back to base model...")
                self.load_base_model()
        else:
            self.load_base_model()
    
    def load_base_model(self):
        """Load base model as fallback"""
        model_name = "microsoft/phi-2"
        print(f"Loading base model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )
        self.model.eval()
    
    def get_recommendation(self, amount, years, risk, age, goal):
        """Get mutual fund recommendations"""
        
        # Create a well-structured prompt
        prompt = f"""You are an expert mutual fund advisor. Based on the investment profile below, provide specific mutual fund recommendations with allocation percentages.

Investment Profile:
Amount: ₹{amount:,}
Investment Horizon: {years} years
Risk Tolerance: {risk}
Age: {age} years
Financial Goal: {goal}

Recommended Portfolio:
Based on your {risk.lower()} risk profile and {years}-year investment horizon, here's the recommended allocation:

"""
        
        # Tokenize
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            max_length=256,
            truncation=True
        )
        
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part
        recommendation = full_response[len(prompt):]
        
        # Clean up the response
        recommendation = self.clean_response(recommendation)
        
        # If response is poor, use fallback
        if len(recommendation) < 50 or not self.is_valid_recommendation(recommendation):
            recommendation = self.get_fallback_recommendation(amount, years, risk, age, goal)
        
        return recommendation
    
    def clean_response(self, text):
        """Clean up generated text"""
        # Remove repetitive text
        lines = text.split('\n')
        cleaned_lines = []
        seen = set()
        
        for line in lines:
            line = line.strip()
            if line and line not in seen:
                cleaned_lines.append(line)
                seen.add(line)
        
        return '\n'.join(cleaned_lines[:20])  # Limit to 20 lines
    
    def is_valid_recommendation(self, text):
        """Check if recommendation is valid"""
        keywords = ['fund', 'equity', 'debt', 'allocation', '%', 'return']
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in keywords)
    
    def get_fallback_recommendation(self, amount, years, risk, age, goal):
        """Rule-based fallback recommendations"""
        
        recommendations = {
            "Low": f"""1. **Debt Funds - 70%** (₹{amount*0.7:,.0f})
   • Liquid Funds: 30%
   • Short Duration Funds: 20%
   • Corporate Bond Funds: 20%

2. **Large Cap Equity - 20%** (₹{amount*0.2:,.0f})
   • Index Funds (Nifty 50): 10%
   • Blue-chip Funds: 10%

3. **Hybrid Funds - 10%** (₹{amount*0.1:,.0f})
   • Conservative Hybrid: 10%

**Expected Returns:** 7-9% annually
**Risk Level:** Low
**Suitable for:** {goal}""",

            "Moderate": f"""1. **Large Cap Equity - 40%** (₹{amount*0.4:,.0f})
   • Index Funds (Nifty 50): 20%
   • Blue-chip Growth Funds: 20%

2. **Mid Cap Equity - 20%** (₹{amount*0.2:,.0f})
   • Mid Cap Growth Funds: 20%

3. **Debt Funds - 30%** (₹{amount*0.3:,.0f})
   • Corporate Bond Funds: 15%
   • Gilt Funds: 15%

4. **Hybrid Funds - 10%** (₹{amount*0.1:,.0f})
   • Balanced Advantage Funds: 10%

**Expected Returns:** 10-12% annually
**Risk Level:** Moderate
**Suitable for:** {goal}""",

            "High": f"""1. **Large Cap Equity - 30%** (₹{amount*0.3:,.0f})
   • Growth Funds: 30%

2. **Mid Cap Equity - 30%** (₹{amount*0.3:,.0f})
   • High Growth Mid Cap: 30%

3. **Small Cap Equity - 25%** (₹{amount*0.25:,.0f})
   • Small Cap Funds: 25%

4. **International Funds - 10%** (₹{amount*0.1:,.0f})
   • US Equity/Nasdaq Funds: 10%

5. **Debt Funds - 5%** (₹{amount*0.05:,.0f})
   • Liquid Funds (Emergency): 5%

**Expected Returns:** 12-15% annually
**Risk Level:** High
**Suitable for:** {goal}"""
        }
        
        base_rec = recommendations.get(risk, recommendations["Moderate"])
        
        # Add investment strategy with age consideration
        strategy = f"""

**Investment Strategy:**
- Start with SIP of ₹{min(amount/12, 50000):,.0f} monthly
- Review portfolio quarterly
- Rebalance annually
- Investment horizon: {years} years
- Age-appropriate allocation for {age} years old investor

**Tax Consideration:**
- ELSS funds for 80C benefits (if applicable)
- Long-term capital gains tax applies after 1 year for equity
- Indexation benefit for debt funds held > 3 years"""
        
        return base_rec + strategy

def main():
    print("=" * 60)
    print("AI-POWERED MUTUAL FUND ADVISOR")
    print("=" * 60)
    
    advisor = MutualFundAdvisor()
    
    # Test cases
    test_cases = [
        (100000, 5, "Moderate", 35, "Wealth Creation"),
        (500000, 10, "High", 30, "Retirement Planning"),
        (50000, 3, "Low", 45, "Emergency Fund")
    ]
    
    for amount, years, risk, age, goal in test_cases:
        print(f"\n{'='*60}")
        print(f"Investment Profile:")
        print(f"  Amount: ₹{amount:,}")
        print(f"  Horizon: {years} years")
        print(f"  Risk: {risk}")
        print(f"  Age: {age}")
        print(f"  Goal: {goal}")
        print(f"{'='*60}")
        
        recommendation = advisor.get_recommendation(amount, years, risk, age, goal)
        print("\n" + recommendation)
        print("\n" + "="*60)
    
    # Interactive mode
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    
    try:
        while True:
            print("\nEnter investment details (or 'quit' to exit):")
            
            user_input = input("Investment Amount (₹): ")
            if user_input.lower() == 'quit':
                break
            
            try:
                amount = float(user_input)
                years = int(input("Investment Horizon (years): "))
                
                print("Risk Tolerance Options: Low, Moderate, High")
                risk = input("Risk Tolerance: ").capitalize()
                if risk not in ["Low", "Moderate", "High"]:
                    risk = "Moderate"
                
                age = int(input("Your Age: "))
                
                print("Goal Options: Wealth Creation, Retirement Planning, Child Education, Emergency Fund, House Purchase")
                goal = input("Financial Goal: ")
                
                print("\nGenerating recommendations...")
                recommendation = advisor.get_recommendation(amount, years, risk, age, goal)
                
                print("\n" + "="*60)
                print("YOUR PERSONALIZED RECOMMENDATION")
                print("="*60)
                print(recommendation)
                print("\n" + "="*60)
                
            except ValueError:
                print("Invalid input. Please enter valid numbers.")
            except Exception as e:
                print(f"Error: {e}")
    
    except KeyboardInterrupt:
        print("\n\nThank you for using MF Advisor!")

if __name__ == "__main__":
    main()