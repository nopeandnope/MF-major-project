"""
FastAPI server for MF Advisor Web Application
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from typing import Optional, List, Dict
import uvicorn
from pathlib import Path
from src.inference_stable import MutualFundAdvisor

# Initialize FastAPI app FIRST
app = FastAPI(
    title="MF Advisor API",
    version="1.0.0",
    description="AI-Powered Mutual Fund Investment Advisor"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize advisor
advisor = MutualFundAdvisor()

# Request/Response models
class InvestmentRequest(BaseModel):
    amount: float
    horizon: int
    risk_tolerance: str
    financial_goal: str
    age: Optional[int] = 30

class FundRecommendation(BaseModel):
    fund_name: str
    category: str
    allocation: float
    amount: float
    risk_level: str

class InvestmentOption(BaseModel):
    option_name: str
    option_description: str
    recommendations: List[Dict]
    total_investment: float
    expected_return_min: float
    expected_return_max: float
    risk_level: str
    strategy: str

class InvestmentResponse(BaseModel):
    options: List[InvestmentOption]
    user_profile: Dict

# Generate 3 different investment options within the SAME risk category
def get_three_investment_options(amount: float, risk: str, horizon: int, age: int, goal: str) -> List[Dict]:
    """Generate 3 different investment strategies within the same risk category"""

    # Base return expectations based on user's selected risk
    returns_map = {
        "Low": (7, 9),
        "Moderate": (10, 12),
        "High": (12, 15)
    }
    base_min, base_max = returns_map.get(risk, (10, 12))

    # Generate risk-appropriate strategies
    if risk == "Low":
        strategies = get_low_risk_strategies(amount, horizon, age, goal)
    elif risk == "High":
        strategies = get_high_risk_strategies(amount, horizon, age, goal)
    else:  # Moderate
        strategies = get_moderate_risk_strategies(amount, horizon, age, goal)

    # Create 3 options with slight return variations
    options = []
    for i, strategy in enumerate(strategies):
        return_adjustment = [-0.5, 0, 0.5][i]  # Slight variations in expected returns
        option = {
            "option_name": strategy["name"],
            "option_description": strategy["description"],
            "recommendations": strategy["funds"],
            "total_investment": amount,
            "expected_return_min": max(base_min + return_adjustment, 5),
            "expected_return_max": min(base_max + return_adjustment, 18),
            "risk_level": risk,
            "strategy": strategy["strategy_text"]
        }
        options.append(option)

    return options

def get_low_risk_strategies(amount: float, horizon: int, age: int, goal: str) -> List[Dict]:
    """Generate 3 different low-risk investment strategies"""
    return [
        {
            "name": "Strategy A: Maximum Safety",
            "description": "Ultra-conservative approach with highest safety and liquidity",
            "strategy_text": "Focus on liquid funds and government securities for maximum capital protection",
            "funds": [
                {"name": "HDFC Liquid Fund", "category": "Liquid", "allocation": 40, "amount": amount * 0.40, "type": "Debt"},
                {"name": "SBI Magnum Gilt Fund", "category": "Gilt", "allocation": 30, "amount": amount * 0.30, "type": "Debt"},
                {"name": "ICICI Prudential Short Term Fund", "category": "Short Duration", "allocation": 20, "amount": amount * 0.20, "type": "Debt"},
                {"name": "UTI Nifty 50 Index Fund", "category": "Large Cap", "allocation": 10, "amount": amount * 0.10, "type": "Equity"},
            ]
        },
        {
            "name": "Strategy B: Steady Income",
            "description": "Debt-focused strategy for consistent income generation",
            "strategy_text": "Emphasizes corporate bonds and banking PSU funds for regular income",
            "funds": [
                {"name": "ICICI Prudential Corporate Bond Fund", "category": "Corporate Bond", "allocation": 35, "amount": amount * 0.35, "type": "Debt"},
                {"name": "Axis Banking & PSU Debt Fund", "category": "Banking & PSU", "allocation": 25, "amount": amount * 0.25, "type": "Debt"},
                {"name": "Kotak Liquid Fund", "category": "Liquid", "allocation": 20, "amount": amount * 0.20, "type": "Debt"},
                {"name": "HDFC Index Fund - Sensex Plan", "category": "Large Cap", "allocation": 15, "amount": amount * 0.15, "type": "Equity"},
                {"name": "HDFC Hybrid Debt Fund", "category": "Hybrid", "allocation": 5, "amount": amount * 0.05, "type": "Hybrid"},
            ]
        },
        {
            "name": "Strategy C: Conservative Growth",
            "description": "Low-risk strategy with slight growth orientation",
            "strategy_text": "Balanced between debt funds and blue-chip equity for modest growth",
            "funds": [
                {"name": "SBI Corporate Bond Fund", "category": "Corporate Bond", "allocation": 30, "amount": amount * 0.30, "type": "Debt"},
                {"name": "Franklin India Short Term Income Fund", "category": "Short Duration", "allocation": 25, "amount": amount * 0.25, "type": "Debt"},
                {"name": "Mirae Asset Large Cap Fund", "category": "Large Cap", "allocation": 20, "amount": amount * 0.20, "type": "Equity"},
                {"name": "Axis Liquid Fund", "category": "Liquid", "allocation": 15, "amount": amount * 0.15, "type": "Debt"},
                {"name": "ICICI Prudential Equity & Debt Fund", "category": "Hybrid", "allocation": 10, "amount": amount * 0.10, "type": "Hybrid"},
            ]
        }
    ]

def get_moderate_risk_strategies(amount: float, horizon: int, age: int, goal: str) -> List[Dict]:
    """Generate 3 different moderate-risk investment strategies"""
    return [
        {
            "name": "Strategy A: Balanced Core",
            "description": "Classic balanced approach with equal emphasis on growth and stability",
            "strategy_text": "Time-tested 60:40 equity-debt allocation for steady wealth creation",
            "funds": [
                {"name": "HDFC Top 100 Fund", "category": "Large Cap", "allocation": 35, "amount": amount * 0.35, "type": "Equity"},
                {"name": "Axis Midcap Fund", "category": "Mid Cap", "allocation": 25, "amount": amount * 0.25, "type": "Equity"},
                {"name": "ICICI Prudential Corporate Bond", "category": "Corporate Bond", "allocation": 25, "amount": amount * 0.25, "type": "Debt"},
                {"name": "SBI Liquid Fund", "category": "Liquid", "allocation": 15, "amount": amount * 0.15, "type": "Debt"},
            ]
        },
        {
            "name": "Strategy B: Diversified Growth",
            "description": "Multi-cap approach with diversified fund selection",
            "strategy_text": "Spreads risk across market caps and fund houses for optimal diversification",
            "funds": [
                {"name": "Mirae Asset Large Cap Fund", "category": "Large Cap", "allocation": 30, "amount": amount * 0.30, "type": "Equity"},
                {"name": "DSP Midcap Fund", "category": "Mid Cap", "allocation": 20, "amount": amount * 0.20, "type": "Equity"},
                {"name": "Kotak Emerging Equity Fund", "category": "Multi Cap", "allocation": 15, "amount": amount * 0.15, "type": "Equity"},
                {"name": "HDFC Corporate Bond Fund", "category": "Corporate Bond", "allocation": 20, "amount": amount * 0.20, "type": "Debt"},
                {"name": "Franklin India Short Term Income", "category": "Short Duration", "allocation": 10, "amount": amount * 0.10, "type": "Debt"},
                {"name": "SBI Equity Hybrid Fund", "category": "Hybrid", "allocation": 5, "amount": amount * 0.05, "type": "Hybrid"},
            ]
        },
        {
            "name": "Strategy C: Index Plus Active",
            "description": "Combination of low-cost index funds with active management",
            "strategy_text": "Blends passive index investing with selective active funds for cost-effective growth",
            "funds": [
                {"name": "UTI Nifty 50 Index Fund", "category": "Index", "allocation": 40, "amount": amount * 0.40, "type": "Equity"},
                {"name": "Nippon India Multi Cap Fund", "category": "Multi Cap", "allocation": 20, "amount": amount * 0.20, "type": "Equity"},
                {"name": "Axis Corporate Debt Fund", "category": "Corporate Bond", "allocation": 25, "amount": amount * 0.25, "type": "Debt"},
                {"name": "HDFC Liquid Fund", "category": "Liquid", "allocation": 15, "amount": amount * 0.15, "type": "Debt"},
            ]
        }
    ]

def get_high_risk_strategies(amount: float, horizon: int, age: int, goal: str) -> List[Dict]:
    """Generate 3 different high-risk investment strategies"""
    return [
        {
            "name": "Strategy A: Aggressive Growth",
            "description": "High-growth strategy focused on mid and small-cap opportunities",
            "strategy_text": "Targets maximum capital appreciation through mid and small-cap funds",
            "funds": [
                {"name": "Axis Bluechip Fund", "category": "Large Cap", "allocation": 25, "amount": amount * 0.25, "type": "Equity"},
                {"name": "Kotak Emerging Equity Fund", "category": "Mid Cap", "allocation": 35, "amount": amount * 0.35, "type": "Equity"},
                {"name": "SBI Small Cap Fund", "category": "Small Cap", "allocation": 30, "amount": amount * 0.30, "type": "Equity"},
                {"name": "Axis Liquid Fund", "category": "Liquid", "allocation": 10, "amount": amount * 0.10, "type": "Debt"},
            ]
        },
        {
            "name": "Strategy B: Global Diversification",
            "description": "International exposure with domestic growth opportunities",
            "strategy_text": "Combines Indian growth stories with global market opportunities",
            "funds": [
                {"name": "Canara Robeco Bluechip Equity", "category": "Large Cap", "allocation": 30, "amount": amount * 0.30, "type": "Equity"},
                {"name": "Invesco India Mid Cap Fund", "category": "Mid Cap", "allocation": 25, "amount": amount * 0.25, "type": "Equity"},
                {"name": "Nippon India Small Cap Fund", "category": "Small Cap", "allocation": 20, "amount": amount * 0.20, "type": "Equity"},
                {"name": "Motilal Oswal Nasdaq 100 FOF", "category": "International", "allocation": 15, "amount": amount * 0.15, "type": "Equity"},
                {"name": "PGIM India Global Equity Opportunities", "category": "International", "allocation": 5, "amount": amount * 0.05, "type": "Equity"},
                {"name": "Kotak Liquid Fund", "category": "Liquid", "allocation": 5, "amount": amount * 0.05, "type": "Debt"},
            ]
        },
        {
            "name": "Strategy C: Sectoral Focus",
            "description": "Growth strategy with selective sectoral exposure",
            "strategy_text": "Focuses on high-growth sectors and emerging opportunities",
            "funds": [
                {"name": "ICICI Prudential Technology Fund", "category": "Sectoral", "allocation": 25, "amount": amount * 0.25, "type": "Equity"},
                {"name": "Aditya Birla Sun Life Frontline Equity", "category": "Large Cap", "allocation": 25, "amount": amount * 0.25, "type": "Equity"},
                {"name": "Franklin India Prima Fund", "category": "Multi Cap", "allocation": 20, "amount": amount * 0.20, "type": "Equity"},
                {"name": "DSP Small Cap Fund", "category": "Small Cap", "allocation": 20, "amount": amount * 0.20, "type": "Equity"},
                {"name": "SBI Liquid Fund", "category": "Liquid", "allocation": 10, "amount": amount * 0.10, "type": "Debt"},
            ]
        }
    ]

# Routes should be defined after app is created

@app.get("/favicon.ico")
async def favicon():
    return Response(content="", media_type="image/x-icon")

@app.get("/")
async def root():
    return FileResponse("web/index.html")

@app.post("/api/recommend", response_model=InvestmentResponse)
async def get_recommendations(request: InvestmentRequest):
    """Get 3 different AI-powered mutual fund recommendation options"""
    try:
        # Get AI strategy text
        recommendation_text = advisor.get_recommendation(
            request.amount,
            request.horizon,
            request.risk_tolerance,
            request.age,
            request.financial_goal
        )

        # Generate 3 different investment options
        options = get_three_investment_options(
            request.amount,
            request.risk_tolerance,
            request.horizon,
            request.age,
            request.financial_goal
        )

        # Add AI strategy to each option
        for option in options:
            option["strategy"] = recommendation_text

        # User profile for context
        user_profile = {
            "amount": request.amount,
            "horizon": request.horizon,
            "risk_tolerance": request.risk_tolerance,
            "age": request.age,
            "financial_goal": request.financial_goal
        }

        return InvestmentResponse(
            options=options,
            user_profile=user_profile
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "model": "loaded"}

# Serve static files
app.mount("/static", StaticFiles(directory="web/static"), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
