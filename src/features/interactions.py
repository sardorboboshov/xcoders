"""
Interaction feature creation
"""

import pandas as pd
import numpy as np


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create advanced interaction features"""
    print("\n" + "="*80)
    print("CREATING INTERACTION FEATURES")
    print("="*80)
    
    # Credit score * DTI
    if 'credit_score' in df.columns and 'debt_to_income_ratio' in df.columns:
        df['credit_dti_interaction'] = df['credit_score'] * (1 - df['debt_to_income_ratio'])
    
    # Income * Loan amount
    if 'annual_income' in df.columns and 'loan_amount' in df.columns:
        df['income_loan_ratio'] = df['loan_amount'] / (df['annual_income'] + 1)
        df['affordability_score'] = df['annual_income'] / (df['loan_amount'] + 1)
    
    # Credit score * Utilization
    if 'credit_score' in df.columns and 'credit_utilization' in df.columns:
        df['credit_utilization_risk'] = df['credit_score'] * (1 - df['credit_utilization'])
    
    # Age * Employment length
    if 'age' in df.columns and 'employment_length' in df.columns:
        df['age_employment_ratio'] = df['employment_length'] / (df['age'] - 18 + 1)
    
    # Monthly payment * Free cash flow
    if 'monthly_payment' in df.columns and 'monthly_free_cash_flow' in df.columns:
        df['payment_coverage_ratio'] = df['monthly_free_cash_flow'] / (df['monthly_payment'] + 1)
        df['payment_stress'] = (df['monthly_payment'] > df['monthly_free_cash_flow']).astype(int)
    
    # Regional factors * Income
    if 'regional_median_income' in df.columns and 'annual_income' in df.columns:
        df['income_vs_regional'] = df['annual_income'] / (df['regional_median_income'] + 1)
        df['below_regional_income'] = (df['income_vs_regional'] < 0.8).astype(int)
    
    # Loan amount * Interest rate
    if 'loan_amount' in df.columns and 'interest_rate' in df.columns:
        df['total_interest_burden'] = df['loan_amount'] * df['interest_rate'] / 100
    
    # Credit accounts * Utilization
    if 'num_credit_accounts' in df.columns and 'credit_utilization' in df.columns:
        df['account_utilization_score'] = df['num_credit_accounts'] * df['credit_utilization']
    
    # Risk score combinations (critical for default prediction)
    risk_score_components = []
    
    # Credit risk component
    if 'credit_score' in df.columns:
        df['credit_risk_score'] = (850 - df['credit_score']) / 850
        risk_score_components.append('credit_risk_score')
    
    # DTI risk component
    if 'debt_to_income_ratio' in df.columns:
        df['dti_risk_score'] = np.clip(df['debt_to_income_ratio'] / 0.5, 0, 1)
        risk_score_components.append('dti_risk_score')
    
    # Utilization risk component
    if 'credit_utilization' in df.columns:
        df['utilization_risk_score'] = df['credit_utilization']
        risk_score_components.append('utilization_risk_score')
    
    # Delinquency risk component
    if 'num_delinquencies_2yrs' in df.columns:
        df['delinquency_risk_score'] = np.clip(df['num_delinquencies_2yrs'] / 5, 0, 1)
        risk_score_components.append('delinquency_risk_score')
    
    # Cash flow risk component
    if 'monthly_free_cash_flow' in df.columns:
        df['cashflow_risk_score'] = np.clip(1 - (df['monthly_free_cash_flow'] + 1000) / 2000, 0, 1)
        risk_score_components.append('cashflow_risk_score')
    
    # Combined risk score (weighted average)
    if len(risk_score_components) > 0:
        df['combined_risk_score'] = df[risk_score_components].mean(axis=1)
        df['high_combined_risk'] = (df['combined_risk_score'] > 0.6).astype(int)
        df['very_high_combined_risk'] = (df['combined_risk_score'] > 0.8).astype(int)
    
    # Payment stress indicators
    if 'monthly_payment' in df.columns and 'monthly_income' in df.columns:
        df['payment_to_income_ratio'] = df['monthly_payment'] / (df['monthly_income'] + 1)
        df['severe_payment_stress'] = (df['payment_to_income_ratio'] > 0.4).astype(int)
        df['extreme_payment_stress'] = (df['payment_to_income_ratio'] > 0.6).astype(int)
    
    # Credit history risk
    if 'num_delinquencies_2yrs' in df.columns and 'oldest_credit_line_age' in df.columns:
        df['recent_delinquency_rate'] = df['num_delinquencies_2yrs'] / (df['oldest_credit_line_age'] + 1)
        df['high_recent_delinquency'] = (df['recent_delinquency_rate'] > 0.1).astype(int)
    
    # Income stability indicators
    if 'employment_length' in df.columns and 'age' in df.columns:
        df['employment_stability'] = df['employment_length'] / (df['age'] - 18 + 1)
        df['unstable_employment'] = (df['employment_stability'] < 0.2).astype(int)
    
    # Loan burden relative to income
    if 'loan_amount' in df.columns and 'annual_income' in df.columns:
        df['loan_burden_ratio'] = df['loan_amount'] / (df['annual_income'] + 1)
        df['excessive_loan_burden'] = (df['loan_burden_ratio'] > 3).astype(int)
        df['very_excessive_loan_burden'] = (df['loan_burden_ratio'] > 5).astype(int)
    
    # Multiple risk flags count
    risk_flags = []
    if 'has_delinquencies' in df.columns:
        risk_flags.append('has_delinquencies')
    if 'has_public_records' in df.columns:
        risk_flags.append('has_public_records')
    if 'has_collections' in df.columns:
        risk_flags.append('has_collections')
    if 'high_dti' in df.columns:
        risk_flags.append('high_dti')
    if 'high_utilization' in df.columns:
        risk_flags.append('high_utilization')
    if 'negative_cash_flow' in df.columns:
        risk_flags.append('negative_cash_flow')
    if 'unstable_employment' in df.columns:
        risk_flags.append('unstable_employment')
    
    if len(risk_flags) > 0:
        df['total_risk_flags'] = df[risk_flags].sum(axis=1)
        df['multiple_risk_flags'] = (df['total_risk_flags'] >= 3).astype(int)
        df['many_risk_flags'] = (df['total_risk_flags'] >= 5).astype(int)
    
    print(f"Created interaction features. New shape: {df.shape}")
    
    return df

