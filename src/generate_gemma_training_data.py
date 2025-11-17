"""
Generate training data from 5-10 years of historical mutual fund data
"""
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
from loguru import logger

from api.models import SessionLocal, MutualFund, NAVHistory, PerformanceMetrics

class TrainingDataGenerator:
    def __init__(self):
        self.db = SessionLocal()
        self.output_dir = Path("data/training")
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_investment_scenarios(self) -> List[Dict]:
        """Generate diverse investment scenarios"""
        scenarios = []
        
        # Investment amounts (in INR)
        amounts = [10000, 25000, 50000, 100000, 250000, 500000, 1000000, 2500000, 5000000]
        
        # Investment horizons (in years)
        horizons = [1, 2, 3, 5, 7, 10, 15, 20]
        
        # Risk profiles
        risk_profiles = ['Conservative', 'Moderate', 'Aggressive']
        
        # Goals
        goals = [
            'Wealth Creation', 'Retirement Planning', 'Child Education',
            'House Purchase', 'Tax Saving', 'Regular Income',
            'Capital Preservation', 'Emergency Fund'
        ]
        
        # Age groups
        ages = [25, 30, 35, 40, 45, 50, 55, 60]
        
        # Generate combinations
        for amount in amounts:
            for horizon in horizons:
                for risk in risk_profiles:
                    for goal in goals:
                        for age in ages:
                            # Logic checks
                            if age + horizon > 75:  # Skip unrealistic combinations
                                continue
                            
                            if risk == 'Conservative' and horizon > 10:
                                continue  # Conservative investors rarely have 10+ year horizons
                            
                            if risk == 'Aggressive' and age > 55:
                                continue  # Older investors are rarely aggressive
                            
                            scenarios.append({
                                'amount': amount,
                                'horizon': horizon,
                                'risk': risk,
                                'goal': goal,
                                'age': age
                            })
        
        return scenarios
    
    def get_fund_recommendations(self, scenario: Dict) -> List[Dict]:
        """Get fund recommendations based on scenario"""
        recommendations = []
        
        # Query funds based on risk profile
        query = self.db.query(MutualFund).join(PerformanceMetrics)
        
        # Risk-based filtering
        if scenario['risk'] == 'Conservative':
            funds = query.filter(
                MutualFund.risk_level.in_(['Low']),
                MutualFund.category.in_(['Liquid', 'Ultra Short Duration', 'Short Duration', 'Arbitrage'])
            ).all()
        elif scenario['risk'] == 'Moderate':
            funds = query.filter(
                MutualFund.risk_level.in_(['Low', 'Moderate']),
                MutualFund.category.in_(['Large Cap', 'Index', 'Hybrid', 'Corporate Bond'])
            ).all()
        else:  # Aggressive
            funds = query.filter(
                MutualFund.risk_level.in_(['Moderate', 'High']),
                MutualFund.category.in_(['Mid Cap', 'Small Cap', 'Multi Cap', 'ELSS'])
            ).all()
        
        if not funds:
            return []
        
        # Score and select funds
        scored_funds = []
        for fund in funds[:30]:  # Limit processing
            # Get latest performance metrics
            metrics = self.db.query(PerformanceMetrics).filter_by(
                fund_id=fund.id
            ).order_by(PerformanceMetrics.calculation_date.desc()).first()
            
            if metrics:
                score = self.calculate_fund_score(fund, metrics, scenario)
                scored_funds.append({
                    'fund': fund,
                    'metrics': metrics,
                    'score': score
                })
        
        # Sort by score
        scored_funds.sort(key=lambda x: x['score'], reverse=True)
        
        # Select top funds with allocation
        allocations = self.determine_allocations(scenario, scored_funds[:7])
        
        for item, allocation in zip(scored_funds[:len(allocations)], allocations):
            recommendations.append({
                'scheme_name': item['fund'].scheme_name,
                'category': item['fund'].category,
                'risk_level': item['fund'].risk_level,
                'allocation': allocation,
                'expected_return': item['metrics'].return_3y if item['metrics'].return_3y else 10,
                'sharpe_ratio': item['metrics'].sharpe_ratio if item['metrics'].sharpe_ratio else 1.0
            })
        
        return recommendations
    
    def calculate_fund_score(self, fund: MutualFund, metrics: PerformanceMetrics, scenario: Dict) -> float:
        """Calculate fund suitability score"""
        score = 50.0
        
        # Return-based scoring (based on horizon)
        if scenario['horizon'] <= 1 and metrics.return_1y:
            score += min(metrics.return_1y * 2, 30)
        elif scenario['horizon'] <= 3 and metrics.return_3y:
            score += min(metrics.return_3y * 1.5, 30)
        elif scenario['horizon'] >= 5 and metrics.return_5y:
            score += min(metrics.return_5y * 1.2, 30)
        
        # Risk-adjusted returns
        if metrics.sharpe_ratio and metrics.sharpe_ratio > 0:
            score += min(metrics.sharpe_ratio * 10, 20)
        
        # Volatility penalty for conservative investors
        if scenario['risk'] == 'Conservative' and metrics.volatility:
            if metrics.volatility > 15:
                score -= (metrics.volatility - 15)
        
        # Category bonus based on goal
        if scenario['goal'] == 'Tax Saving' and fund.category == 'ELSS':
            score += 20
        elif scenario['goal'] == 'Regular Income' and 'Debt' in fund.category:
            score += 15
        elif scenario['goal'] == 'Wealth Creation' and 'Equity' in fund.category:
            score += 15
        
        return min(score, 100)
    
    def determine_allocations(self, scenario: Dict, top_funds: List) -> List[int]:
        """Determine allocation percentages"""
        num_funds = len(top_funds)
        
        if num_funds == 0:
            return []
        elif num_funds == 1:
            return [100]
        elif num_funds == 2:
            return [60, 40]
        elif num_funds == 3:
            return [50, 30, 20]
        elif num_funds == 4:
            return [40, 30, 20, 10]
        elif num_funds == 5:
            return [30, 25, 20, 15, 10]
        elif num_funds == 6:
            return [25, 20, 20, 15, 10, 10]
        else:
            return [25, 20, 15, 15, 10, 10, 5]
    
    def create_training_examples(self) -> List[Dict]:
        """Create training examples for Gemma"""
        training_data = []
        
        # Get all scenarios
        scenarios = self.generate_investment_scenarios()
        logger.info(f"Generated {len(scenarios)} investment scenarios")
        
        # Sample scenarios (to avoid too much data)
        sampled_scenarios = np.random.choice(scenarios, min(2000, len(scenarios)), replace=False)
        
        for i, scenario in enumerate(sampled_scenarios):
            if i % 100 == 0:
                logger.info(f"Processing scenario {i}/{len(sampled_scenarios)}")
            
            # Get recommendations
            recommendations = self.get_fund_recommendations(scenario)
            
            if not recommendations:
                continue
            
            # Create conversational training example
            prompt = f"""<start_of_turn>user
I want to invest ₹{scenario['amount']:,} for {scenario['horizon']} years. 
My risk tolerance is {scenario['risk']} and I'm {scenario['age']} years old.
My investment goal is {scenario['goal']}.
Please recommend a mutual fund portfolio.<end_of_turn>
<start_of_turn>model"""
            
            # Create detailed response
            response = f"""Based on your investment profile, here's my recommended mutual fund portfolio:

**Investment Summary:**
- Amount: ₹{scenario['amount']:,}
- Time Horizon: {scenario['horizon']} years
- Risk Profile: {scenario['risk']}
- Primary Goal: {scenario['goal']}

**Recommended Portfolio Allocation:**

"""
            for j, rec in enumerate(recommendations, 1):
                amount = scenario['amount'] * rec['allocation'] / 100
                response += f"""{j}. **{rec['scheme_name']}**
   - Category: {rec['category']}
   - Risk Level: {rec['risk_level']}
   - Allocation: {rec['allocation']}% (₹{amount:,.0f})
   - Expected Return: {rec['expected_return']:.1f}% p.a.
   - Sharpe Ratio: {rec['sharpe_ratio']:.2f}

"""
            
            # Add portfolio summary
            avg_return = np.mean([r['expected_return'] for r in recommendations])
            response += f"""**Portfolio Characteristics:**
- Expected Portfolio Return: {avg_return:.1f}% p.a.
- Diversification: Across {len(recommendations)} funds
- Risk Management: Aligned with your {scenario['risk']} risk profile
- Rebalancing: Recommended annually

**Investment Strategy:**
"""
            
            # Add strategy based on goal
            if scenario['goal'] == 'Retirement Planning':
                response += "Focus on long-term wealth creation with gradual shift to debt as you near retirement."
            elif scenario['goal'] == 'Child Education':
                response += "Balanced approach with equity for growth and debt for stability as the goal approaches."
            elif scenario['goal'] == 'Tax Saving':
                response += "Maximize tax benefits under Section 80C with ELSS funds while building long-term wealth."
            elif scenario['goal'] == 'Wealth Creation':
                response += "Growth-oriented strategy with focus on equity funds for maximum long-term returns."
            else:
                response += "Diversified approach balancing growth and stability based on your investment horizon."
            
            response += f"""

**Next Steps:**
1. Start with SIP (Systematic Investment Plan) if investing gradually
2. Review portfolio performance quarterly
3. Rebalance annually to maintain target allocations
4. Consider increasing investment amount with salary increments<end_of_turn>"""
            
            # Combine prompt and response
            full_text = prompt + response
            
            training_data.append({
                'text': full_text,
                'scenario': scenario,
                'recommendations': recommendations
            })
        
        return training_data
    
    def save_training_data(self, training_data: List[Dict]):
        """Save training data in JSONL format for Gemma"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.output_dir / f"gemma_training_{timestamp}.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in training_data:
                # Save only the text for training
                f.write(json.dumps({'text': item['text']}, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(training_data)} training examples to {output_file}")
        
        # Also save metadata
        metadata_file = self.output_dir / f"training_metadata_{timestamp}.json"
        metadata = {
            'total_examples': len(training_data),
            'generation_date': datetime.now().isoformat(),
            'unique_scenarios': len(set(str(item['scenario']) for item in training_data)),
            'funds_used': len(set(fund['scheme_name'] for item in training_data for fund in item['recommendations']))
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {metadata_file}")
    
    def generate_quality_training_data(self):
        """Main function to generate high-quality training data"""
        logger.info("Starting training data generation from historical data...")
        
        try:
            # Check database has sufficient data
            fund_count = self.db.query(MutualFund).count()
            logger.info(f"Found {fund_count} funds in database")
            
            if fund_count < 100:
                logger.error("Insufficient funds in database. Run historical data collection first.")
                return
            
            # Generate training examples
            training_data = self.create_training_examples()
            
            if not training_data:
                logger.error("No training data generated")
                return
            
            # Save training data
            self.save_training_data(training_data)
            
            logger.info(f"Successfully generated {len(training_data)} training examples")
            
        except Exception as e:
            logger.error(f"Error generating training data: {str(e)}")
        finally:
            self.db.close()

if __name__ == "__main__":
    generator = TrainingDataGenerator()
    generator.generate_quality_training_data()