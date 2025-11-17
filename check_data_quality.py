"""Quick script to check data quality"""
from src.data_quality_validator import DataQualityValidator

v = DataQualityValidator()
report = v.validate_database('data/historical_20y.db')
print(f'Quality Score: {report["avg_quality_score"]}/100')
print(f'Total Schemes: {report["total_schemes"]}')
print(f'Passed Schemes: {report["passed_schemes"]}')
print(f'Failed Schemes: {report["failed_schemes"]}')
print(f'Pass Rate: {report["pass_rate"]}%')
