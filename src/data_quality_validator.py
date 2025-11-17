"""
Data Quality Validator for Mutual Fund NAV Data

This module provides comprehensive data quality validation for mutual fund
historical NAV data, including checks for missing values, outliers, date gaps,
and cross-validation with backup sources.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import sqlite3
import json


class DataQualityValidator:
    """Validate and ensure data quality for mutual fund NAV data"""

    def __init__(self, data_source: str = 'mfapi.in'):
        """
        Initialize the data quality validator.

        Args:
            data_source: Name of the data source being validated
        """
        self.data_source = data_source
        self.quality_report = {}

    def validate_data_quality(self, df: pd.DataFrame, scheme_code: str) -> Dict[str, Any]:
        """
        Perform comprehensive data quality checks on fund data.

        Args:
            df: DataFrame with columns ['date', 'nav']
            scheme_code: Unique identifier for the mutual fund scheme

        Returns:
            Dictionary containing quality check results and overall score
        """
        results = {
            'scheme_code': scheme_code,
            'total_records': len(df),
            'date_range': f"{df['date'].min()} to {df['date'].max()}",
            'checks': {}
        }

        # 1. Check for missing values
        missing_pct = (df.isnull().sum() / len(df) * 100).to_dict()
        results['checks']['missing_values'] = missing_pct

        # 2. Check for outliers (>50% daily change likely error)
        df_copy = df.copy()
        df_copy['daily_change'] = df_copy['nav'].pct_change()
        outliers = df_copy[abs(df_copy['daily_change']) > 0.5]
        results['checks']['outliers'] = len(outliers)
        results['checks']['outlier_details'] = outliers[['date', 'nav', 'daily_change']].to_dict('records') if len(outliers) > 0 else []

        # 3. Verify date continuity (check for gaps > 5 days)
        df_copy['date'] = pd.to_datetime(df_copy['date'], format='mixed', dayfirst=True)
        df_copy = df_copy.sort_values('date')
        df_copy['date_diff'] = df_copy['date'].diff().dt.days
        gaps = df_copy[df_copy['date_diff'] > 5]
        results['checks']['date_gaps'] = len(gaps)
        results['checks']['gap_details'] = gaps[['date', 'date_diff']].to_dict('records') if len(gaps) > 0 else []

        # 4. Check for duplicate dates
        duplicates = df_copy[df_copy.duplicated(subset=['date'], keep=False)]
        results['checks']['duplicate_dates'] = len(duplicates)

        # 5. Validate NAV values (should be positive)
        invalid_nav = df_copy[df_copy['nav'] <= 0]
        results['checks']['invalid_nav_values'] = len(invalid_nav)

        # 6. Check data freshness (last update within 7 days?)
        last_date = pd.to_datetime(df_copy['date'].max())
        days_since_update = (datetime.now() - last_date).days
        results['checks']['days_since_last_update'] = days_since_update

        # Calculate overall quality score (0-100)
        score = 100
        if missing_pct.get('nav', 0) > 1:
            score -= 20
        if len(outliers) > 0:
            score -= 15
        if len(gaps) > 10:
            score -= 15
        if len(duplicates) > 0:
            score -= 10
        if len(invalid_nav) > 0:
            score -= 20
        if days_since_update > 7:
            score -= 10

        results['quality_score'] = max(0, score)
        results['status'] = 'PASS' if score >= 70 else 'FAIL'

        return results

    def cross_validate_with_backup(
        self,
        primary_data: pd.DataFrame,
        backup_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Cross-validate data from primary source with backup source.

        Args:
            primary_data: DataFrame from primary source with ['date', 'nav']
            backup_data: DataFrame from backup source with ['date', 'nav']

        Returns:
            Dictionary with validation metrics
        """
        # Ensure dates are datetime objects
        primary_data = primary_data.copy()
        backup_data = backup_data.copy()
        primary_data['date'] = pd.to_datetime(primary_data['date'])
        backup_data['date'] = pd.to_datetime(backup_data['date'])

        # Merge on date
        merged = pd.merge(
            primary_data,
            backup_data,
            on='date',
            suffixes=('_primary', '_backup')
        )

        if len(merged) == 0:
            return {
                'total_compared': 0,
                'discrepancies': 0,
                'avg_difference_pct': 0,
                'max_difference_pct': 0,
                'data_agreement': 0
            }

        # Calculate difference
        merged['nav_diff'] = abs(merged['nav_primary'] - merged['nav_backup'])
        merged['nav_diff_pct'] = (merged['nav_diff'] / merged['nav_primary']) * 100

        # Flag discrepancies (>1% difference)
        discrepancies = merged[merged['nav_diff_pct'] > 1]

        return {
            'total_compared': len(merged),
            'discrepancies': len(discrepancies),
            'avg_difference_pct': round(merged['nav_diff_pct'].mean(), 4),
            'max_difference_pct': round(merged['nav_diff_pct'].max(), 4),
            'data_agreement': round((1 - len(discrepancies)/len(merged)) * 100, 2)
        }

    def generate_data_metadata(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate metadata for dataset versioning and tracking.

        Args:
            df: DataFrame with mutual fund data (must have 'date', 'scheme_code' columns)

        Returns:
            Dictionary containing dataset metadata
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])

        date_min = df['date'].min()
        date_max = df['date'].max()
        years = (date_max - date_min).days / 365.25

        return {
            'collection_date': datetime.now().isoformat(),
            'data_source': self.data_source,
            'num_funds': df['scheme_code'].nunique() if 'scheme_code' in df.columns else 1,
            'total_records': len(df),
            'date_range': {
                'start': date_min.isoformat(),
                'end': date_max.isoformat(),
                'years': round(years, 2)
            },
            'data_version': 'v1.0'
        }

    def validate_database(self, db_path: str) -> Dict[str, Any]:
        """
        Validate entire historical database and generate comprehensive report.

        Args:
            db_path: Path to SQLite database

        Returns:
            Dictionary with database-level validation results
        """
        conn = sqlite3.connect(db_path)

        # Get list of all schemes
        schemes_df = pd.read_sql_query(
            "SELECT DISTINCT scheme_code FROM nav_history_20y",
            conn
        )

        total_schemes = len(schemes_df)
        passed_schemes = 0
        failed_schemes = 0
        scheme_reports = []

        for scheme_code in schemes_df['scheme_code']:
            # Load scheme data
            df = pd.read_sql_query(
                f"SELECT date, nav FROM nav_history_20y WHERE scheme_code = '{scheme_code}'",
                conn
            )

            # Validate
            report = self.validate_data_quality(df, scheme_code)
            scheme_reports.append(report)

            if report['status'] == 'PASS':
                passed_schemes += 1
            else:
                failed_schemes += 1

        conn.close()

        # Calculate aggregate statistics
        avg_quality_score = np.mean([r['quality_score'] for r in scheme_reports])

        return {
            'database_path': db_path,
            'total_schemes': total_schemes,
            'passed_schemes': passed_schemes,
            'failed_schemes': failed_schemes,
            'pass_rate': round((passed_schemes / total_schemes) * 100, 2),
            'avg_quality_score': round(avg_quality_score, 2),
            'failed_scheme_details': [
                {'scheme_code': r['scheme_code'], 'score': r['quality_score']}
                for r in scheme_reports if r['status'] == 'FAIL'
            ]
        }

    def export_quality_report(self, report: Dict[str, Any], output_path: str):
        """
        Export quality validation report to JSON file.

        Args:
            report: Quality report dictionary
            output_path: Path to save JSON report
        """
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"Quality report saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    # Initialize validator
    validator = DataQualityValidator(data_source='mfapi.in')

    # Example: Validate single fund data
    sample_data = pd.DataFrame({
        'date': pd.date_range('2020-01-01', '2023-12-31', freq='D'),
        'nav': np.random.uniform(100, 200, 1461)
    })

    quality_report = validator.validate_data_quality(sample_data, scheme_code='119551')

    print("=" * 70)
    print("DATA QUALITY VALIDATION REPORT")
    print("=" * 70)
    print(f"Scheme Code: {quality_report['scheme_code']}")
    print(f"Total Records: {quality_report['total_records']}")
    print(f"Date Range: {quality_report['date_range']}")
    print(f"Quality Score: {quality_report['quality_score']}/100")
    print(f"Status: {quality_report['status']}")
    print("\nDetailed Checks:")
    for check_name, check_value in quality_report['checks'].items():
        if not check_name.endswith('_details'):
            print(f"  {check_name}: {check_value}")
    print("=" * 70)
