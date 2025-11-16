"""
Data cleaning functions for each table
"""

import pandas as pd
import numpy as np
from datetime import datetime


def clean_application_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and transform application_metadata table"""
    df = df.copy()
    
    # Standardize employment_type (many variations of same value)
    if 'preferred_contact' in df.columns:
        df['preferred_contact'] = df['preferred_contact'].str.strip()
    
    # Handle referral_code - most are REF0000, create binary feature
    if 'referral_code' in df.columns:
        df['has_referral'] = (df['referral_code'] != 'REF0000').astype(int)
        df['referral_code'] = df['referral_code'].fillna('REF0000')
    
    # Standardize account_status_code
    if 'account_status_code' in df.columns:
        df['account_status_code'] = df['account_status_code'].str.strip()
    
    # Create time-based features
    if 'application_hour' in df.columns:
        df['is_business_hours'] = ((df['application_hour'] >= 9) & (df['application_hour'] <= 17)).astype(int)
        df['is_weekend'] = (df['application_day_of_week'] >= 5).astype(int)
        df['is_evening'] = (df['application_hour'] >= 18).astype(int)
        df['is_morning'] = (df['application_hour'] < 9).astype(int)
    
    # Account age features
    if 'account_open_year' in df.columns:
        current_year = datetime.now().year
        df['account_age_years'] = current_year - df['account_open_year']
        df['is_new_account'] = (df['account_age_years'] <= 1).astype(int)
        df['is_old_account'] = (df['account_age_years'] >= 5).astype(int)
    
    # Interaction features
    if 'num_login_sessions' in df.columns and 'num_customer_service_calls' in df.columns:
        df['login_to_service_ratio'] = df['num_login_sessions'] / (df['num_customer_service_calls'] + 1)
        df['total_engagement'] = df['num_login_sessions'] + df['num_customer_service_calls']
    
    # Drop random_noise_1 (it's noise!)
    if 'random_noise_1' in df.columns:
        df = df.drop('random_noise_1', axis=1)
    
    return df


def clean_credit_history(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and transform credit_history table"""
    df = df.copy()
    
    cols_to_int = ["oldest_credit_line_age", "oldest_account_age_months", "num_delinquencies_2yrs"]
    for col in cols_to_int:
        if col in df.columns:
            df[col] = df[col].round(0).astype("Int64")

    # Handle missing values in num_delinquencies_2yrs
    if 'num_delinquencies_2yrs' in df.columns:
        df['num_delinquencies_2yrs'] = df['num_delinquencies_2yrs'].fillna(0)
    
    # Credit score bins
    if 'credit_score' in df.columns:
        df['credit_score_category'] = pd.cut(
            df['credit_score'],
            bins=[0, 580, 670, 740, 850],
            labels=['Poor', 'Fair', 'Good', 'Excellent']
        )
        df['is_excellent_credit'] = (df['credit_score'] >= 740).astype(int)
        df['is_poor_credit'] = (df['credit_score'] < 580).astype(int)
    
    # Credit utilization features
    if 'total_credit_limit' in df.columns and 'num_credit_accounts' in df.columns:
        df['avg_credit_limit_per_account'] = df['total_credit_limit'] / (df['num_credit_accounts'] + 1)
    
    # Credit history age features
    if 'oldest_credit_line_age' in df.columns and 'oldest_account_age_months' in df.columns:
        df['credit_history_consistency'] = df['oldest_account_age_months'] / (df['oldest_credit_line_age'] * 12 + 1)
    
    # Risk indicators
    if 'num_delinquencies_2yrs' in df.columns:
        df['has_delinquencies'] = (df['num_delinquencies_2yrs'] > 0).astype(int)
    
    if 'num_public_records' in df.columns:
        df['has_public_records'] = (df['num_public_records'] > 0).astype(int)
    
    if 'num_collections' in df.columns:
        df['has_collections'] = (df['num_collections'] > 0).astype(int)
    
    # Inquiry features
    if 'num_inquiries_6mo' in df.columns:
        df['high_inquiry_rate'] = (df['num_inquiries_6mo'] >= 3).astype(int)
    
    # Combined risk score
    risk_features = ['has_delinquencies', 'has_public_records', 'has_collections', 'high_inquiry_rate']
    available_risk = [f for f in risk_features if f in df.columns]
    if available_risk:
        df['total_risk_indicators'] = df[available_risk].sum(axis=1)
    
    return df


def clean_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and transform demographics table"""
    df = df.copy()
    
    # Standardize employment_type (many variations)
    if 'employment_type' in df.columns:
        df['employment_type'] = df['employment_type'].str.strip()
        # Normalize variations
        df['employment_type'] = df['employment_type'].str.replace(' ', '_', regex=False)
        df['employment_type'] = df['employment_type'].str.upper()
        
        # Map variations to standard values
        full_time_variants = ['FULL-TIME', 'FULL_TIME', 'FULLTIME', 'FT']
        self_employed_variants = ['SELF_EMPLOYED', 'SELF_EMP', 'SELF-EMPLOYED']
        part_time_variants = ['PART_TIME', 'PART-TIME', 'PT']
        contractor_variants = ['CONTRACTOR', 'CONTRACT']
        
        df.loc[df['employment_type'].isin(full_time_variants), 'employment_type'] = 'FULL_TIME'
        df.loc[df['employment_type'].isin(self_employed_variants), 'employment_type'] = 'SELF_EMPLOYED'
        df.loc[df['employment_type'].isin(part_time_variants), 'employment_type'] = 'PART_TIME'
        df.loc[df['employment_type'].isin(contractor_variants), 'employment_type'] = 'CONTRACTOR'
    
    # Handle missing employment_length
    if 'employment_length' in df.columns:
        df['employment_length'] = df['employment_length'].fillna(df['employment_length'].median())
        df['is_new_employee'] = (df['employment_length'] < 1).astype(int)
        df['is_experienced'] = (df['employment_length'] >= 5).astype(int)
    
    # Age bins
    if 'age' in df.columns:
        df['age_group'] = pd.cut(
            df['age'],
            bins=[0, 25, 35, 45, 55, 100],
            labels=['18-25', '26-35', '36-45', '46-55', '55+']
        )
        df['is_young'] = (df['age'] < 30).astype(int)
        df['is_senior'] = (df['age'] >= 55).astype(int)
    
    # Income features
    if 'annual_income' in df.columns:
        df["annual_income"] = (
            df["annual_income"]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .astype(float)
        )
        df['income_category'] = pd.cut(
            df['annual_income'],
            bins=[0, 30000, 50000, 75000, 100000, float('inf')],
            labels=['Low', 'Medium', 'Medium-High', 'High', 'Very High']
        )
        df['log_annual_income'] = np.log1p(df['annual_income'])
    
    # Dependents
    if 'num_dependents' in df.columns:
        df['has_dependents'] = (df['num_dependents'] > 0).astype(int)
        df['many_dependents'] = (df['num_dependents'] >= 3).astype(int)
    
    # Education encoding (ordinal)
    if 'education' in df.columns:
        education_order = {
            'High School': 1,
            'Some College': 2,
            'Bachelor': 3,
            'Graduate': 4,
            'Advanced': 5
        }
        df['education_encoded'] = df['education'].map(education_order).fillna(2)
    
    return df


def clean_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and transform financial_ratios table"""
    df = df.copy()
    
    money_cols = ['existing_monthly_debt', 'monthly_income',
              'monthly_payment', 'revolving_balance', 'credit_usage_amount',
              'available_credit', 'total_monthly_debt_payment',
              'total_debt_amount', 'monthly_free_cash_flow']
    for col in money_cols:
        try:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
                .replace("None", None)        # optional, handles literal "None"
                .replace("", None)
                .astype(float)
            )
        except Exception as e:
            print(str(e))
            print(col)

    # Handle missing values
    if 'revolving_balance' in df.columns:
        df['revolving_balance'] = df['revolving_balance'].fillna(0)
    
    # Financial health indicators
    if 'debt_to_income_ratio' in df.columns:
        df['high_dti'] = (df['debt_to_income_ratio'] > 0.43).astype(int)
        df['very_high_dti'] = (df['debt_to_income_ratio'] > 0.6).astype(int)
        df['dti_category'] = pd.cut(
            df['debt_to_income_ratio'],
            bins=[0, 0.36, 0.43, 0.5, float('inf')],
            labels=['Low', 'Moderate', 'High', 'Very High']
        )
    
    if 'credit_utilization' in df.columns:
        df['high_utilization'] = (df['credit_utilization'] > 0.7).astype(int)
        df['maxed_out'] = (df['credit_utilization'] >= 0.95).astype(int)
    
    if 'payment_to_income_ratio' in df.columns:
        df['high_payment_ratio'] = (df['payment_to_income_ratio'] > 0.3).astype(int)
    
    # Cash flow features
    if 'monthly_free_cash_flow' in df.columns:
        df['negative_cash_flow'] = (df['monthly_free_cash_flow'] < 0).astype(int)
        df['low_cash_flow'] = (df['monthly_free_cash_flow'] < 500).astype(int)
        df['log_free_cash_flow'] = np.log1p(df['monthly_free_cash_flow'] - df['monthly_free_cash_flow'].min())
    
    # Loan burden
    if 'loan_to_annual_income' in df.columns:
        df['high_loan_burden'] = (df['loan_to_annual_income'] > 2).astype(int)
        df['very_high_loan_burden'] = (df['loan_to_annual_income'] > 5).astype(int)
    
    # Debt service capacity
    if 'monthly_income' in df.columns and 'total_monthly_debt_payment' in df.columns:
        df['debt_service_capacity'] = df['monthly_income'] - df['total_monthly_debt_payment']
        df['debt_service_ratio_custom'] = df['total_monthly_debt_payment'] / (df['monthly_income'] + 1)
        df['low_debt_capacity'] = (df['debt_service_capacity'] < 500).astype(int)
    
    # Credit usage features
    if 'credit_usage_amount' in df.columns and 'available_credit' in df.columns:
        df['credit_usage_ratio'] = df['credit_usage_amount'] / (df['available_credit'] + df['credit_usage_amount'] + 1)
    
    return df


def clean_geographic_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and transform geographic_data table"""
    df = df.copy()
    
    # Regional economic indicators
    if 'regional_unemployment_rate' in df.columns:
        df['high_unemployment'] = (df['regional_unemployment_rate'] > 5.0).astype(int)
        df['low_unemployment'] = (df['regional_unemployment_rate'] < 4.0).astype(int)
    
    if 'regional_median_income' in df.columns:
        df['above_median_income'] = (df['regional_median_income'] > df['regional_median_income'].median()).astype(int)
    
    if 'cost_of_living_index' in df.columns:
        df['high_cost_area'] = (df['cost_of_living_index'] > 100).astype(int)
        df['low_cost_area'] = (df['cost_of_living_index'] < 85).astype(int)
    
    if 'housing_price_index' in df.columns:
        df['high_housing_cost'] = (df['housing_price_index'] > 120).astype(int)
    
    # Affordability ratio
    if 'regional_median_rent' in df.columns and 'regional_median_income' in df.columns:
        df['rent_to_income_ratio'] = (df['regional_median_rent'] * 12) / (df['regional_median_income'] + 1)
        df['high_rent_burden'] = (df['rent_to_income_ratio'] > 0.3).astype(int)
    
    return df


def clean_loan_details(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and transform loan_details table"""
    df = df.copy()
    
    df["loan_amount"] = (
        df["loan_amount"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(float)
    )
    
    # Standardize loan_type (many variations)
    if 'loan_type' in df.columns:
        df['loan_type'] = df['loan_type'].str.strip()
        df['loan_type'] = df['loan_type'].str.replace(' ', '_', regex=False)
        df['loan_type'] = df['loan_type'].str.upper()
        
        # Map variations
        personal_variants = ['PERSONAL', 'PERSONAL_LOAN']
        mortgage_variants = ['MORTGAGE', 'HOME_LOAN']
        credit_card_variants = ['CREDIT_CARD', 'CREDITCARD']
        
        df.loc[df['loan_type'].isin(personal_variants), 'loan_type'] = 'PERSONAL'
        df.loc[df['loan_type'].isin(mortgage_variants), 'loan_type'] = 'MORTGAGE'
        df.loc[df['loan_type'].isin(credit_card_variants), 'loan_type'] = 'CREDIT_CARD'
    
    # Loan amount features
    if 'loan_amount' in df.columns:
        df['log_loan_amount'] = np.log1p(df['loan_amount'])
        df['loan_amount_category'] = pd.cut(
            df['loan_amount'],
            bins=[0, 10000, 50000, 100000, 200000, float('inf')],
            labels=['Small', 'Medium', 'Large', 'Very Large', 'Jumbo']
        )
        df['large_loan'] = (df['loan_amount'] > 100000).astype(int)
        df['small_loan'] = (df['loan_amount'] < 10000).astype(int)
    
    # Interest rate features
    if 'interest_rate' in df.columns:
        df['high_interest'] = (df['interest_rate'] > 15).astype(int)
        df['low_interest'] = (df['interest_rate'] < 6).astype(int)
        df['interest_rate_category'] = pd.cut(
            df['interest_rate'],
            bins=[0, 6, 10, 15, 20, float('inf')],
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
    
    # Loan term features
    if 'loan_term' in df.columns:
        df['is_short_term'] = (df['loan_term'] <= 12).astype(int)
        df['is_long_term'] = (df['loan_term'] >= 60).astype(int)
        df['is_mortgage_term'] = (df['loan_term'] >= 300).astype(int)
        # Handle 0 term (might be revolving credit)
        df['has_term'] = (df['loan_term'] > 0).astype(int)
    
    # LTV ratio
    if 'loan_to_value_ratio' in df.columns:
        df['has_ltv'] = (df['loan_to_value_ratio'] > 0).astype(int)
        df['high_ltv'] = (df['loan_to_value_ratio'] > 0.8).astype(int)
    
    # Loan purpose encoding
    if 'loan_purpose' in df.columns:
        # Risk-based encoding
        purpose_risk = {
            'Debt Consolidation': 3,  # Higher risk
            'Home Improvement': 2,
            'Major Purchase': 2,
            'Medical': 3,  # Higher risk
            'Other': 3,  # Higher risk
            'Revolving Credit': 2,
            'Home Purchase': 1,  # Lower risk
            'Refinance': 1  # Lower risk
        }
        df['loan_purpose_risk'] = df['loan_purpose'].map(purpose_risk).fillna(2)
    
    return df

