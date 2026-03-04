"""
config.py — Single source of truth for paths, constants, and column lists.

Why this exists:
  Instead of exporting feature lists and metadata between notebooks as JSON/parquet,
  every notebook imports from here. This eliminates cross-notebook dependencies and
  ensures experiment code and production code use identical definitions.

Usage:
  from utils.config import PATHS, SPLIT_DATES, LEAKAGE_COLUMNS, CONTINUOUS_FEATURES
"""

import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────
# Resolve project root relative to this file's location:

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

PATHS = {
    'raw_data':       PROJECT_ROOT / 'data' / '01_raw' / 'LoanData.csv',
    'data_dictionary': PROJECT_ROOT / 'references' / 'data_dictionary.csv',
    'cleaned':        PROJECT_ROOT / 'data' / '02_processed' / '01_estonia_cleaned.parquet',
    'target_built':   PROJECT_ROOT / 'data' / '02_processed' / '02_estonia_target_creation.parquet',
    'final_dir':      PROJECT_ROOT / 'data' / '03_final',
    'models_dir':     PROJECT_ROOT / 'models',
    'reports_dir':    PROJECT_ROOT / 'reports',
}


# ─────────────────────────────────────────────────────────────────────
# TARGET CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────
SNAPSHOT_DATE = pd.Timestamp('2023-10-15')
HORIZON_DAYS = 365
TARGET_COL = 'into_12m_default_ind'


# ─────────────────────────────────────────────────────────────────────
# TEMPORAL SPLIT DATES
# ─────────────────────────────────────────────────────────────────────
SPLIT_DATES = {
    'train_end': pd.Timestamp('2021-06-30'),
    'val_end':   pd.Timestamp('2021-12-31'),
    # OOT = everything after val_end
}


# ─────────────────────────────────────────────────────────────────────
# LEAKAGE & META COLUMNS — must never be used as features
# ─────────────────────────────────────────────────────────────────────
# Grouped by reason so the logic is auditable

LEAKAGE_COLUMNS = [
    # --- Target construction helpers ---
    'DefaultDate', 'days_to_default',

    # --- Identifiers (no predictive value) ---
    'ReportAsOfEOD', 'LoanId', 'LoanNumber', 'PartyId',
    

    # --- Post-origination payment/schedule ---
    'PrincipalPaymentsMade', 'InterestAndPenaltyPaymentsMade',
    'PrincipalBalance', 'PrincipalOverdueBySchedule',
    'NextPaymentNr', 'NrOfScheduledPayments',
    'PlannedPrincipalTillDate', 'PlannedInterestTillDate',
    'ActiveScheduleFirstPaymentReached',

    # --- Post-origination delinquency ---
    'CurrentDebtDaysPrimary', 'CurrentDebtDaysSecondary',
    'ActiveLateCategory', 'ActiveLateLastPaymentCategory',

    # --- Post-default recovery ---
    'PrincipalRecovery', 'InterestRecovery',
    'PrincipalWriteOffs', 'InterestAndPenaltyWriteOffs',
    'PrincipalDebtServicingCost', 'InterestAndPenaltyDebtServicingCost',
    'PlannedPrincipalPostDefault', 'PlannedInterestPostDefault',
    'EAD1', 'EAD2',
    

    # ---  model outputs (predicting the prediction) ---
    'ExpectedLoss', 'LossGivenDefault', 'ExpectedReturn',
    'ProbabilityOfDefault', 'Rating', 'ModelVersion',
    'EL_V0', 'Rating_V0', 'EL_V1', 'Rating_V1', 'Rating_V2',

    # --- Contract lifecycle (changes post-origination) ---
    'Status', 'Restructured',

    # --- Investor-side / auction fields ---
    'BidsPortfolioManager', 'BidsApi', 'BidsManual',

    # --- Post-default  ---
    'RecoveryStage', 'WorseLateCategory', 'InterestAndPenaltyBalance', # Dropping this I suspect it is leakage that customer is paying penalties on current loan, 
    # To me it looks like post-origination that interest and penalty can only be paid later , could be past debt , but I dropped it for now

    # -- Different country rating agency
    'CreditScoreFiAsiakasTietoRiskGrade',
]


# ─────────────────────────────────────────────────────────────────────
# FEATURE DEFINITIONS
# ─────────────────────────────────────────────────────────────────────
# Features the WoE engine treats as continuous (quantile-binned)
CONTINUOUS_FEATURES = [
    'Age', 'AppliedAmount', 'Amount', 'Interest', 'LoanDuration',
    'IncomeTotal', 'LiabilitiesTotal', 'DebtToIncome', 'FreeCash',
    'MonthlyPayment', 'ExistingLiabilities', 'RefinanceLiabilities',
    'NoOfPreviousLoansBeforeLoan', 'AmountOfPreviousLoansBeforeLoan',
    'PreviousRepaymentsBeforeLoan', 'PreviousEarlyRepaymentsCountBeforeLoan',
    'CreditScoreEeMini',
    # Engineered
    'loan_to_income', 'approval_ratio',
]

# Features the WoE engine treats as categorical (level-mapped)
CATEGORICAL_FEATURES = [
    'NewCreditCustomer', 'VerificationType', 'LanguageCode', 'Gender',
    'Education', 'MaritalStatus', 'EmploymentStatus',
    'EmploymentDurationCurrentEmployer', 'OccupationArea',
    'HomeOwnershipType', 'UseOfLoan',
]

# Highly correlated features to drop before WoE (from correlation analysis)
CORRELATED_DROP = [
    'AppliedAmount',                          # r=0.96 with Amount
    'MonthlyPayment',                         # r=0.88 with AppliedAmount
    'NoOfPreviousLoansBeforeLoan',            # r=0.75 with AmountOfPreviousLoans
    'PreviousEarlyRepaymentsCountBeforeLoan', # r=0.72 with PreviousEarlyRepayments
]

# Categoricals for XGBoost native encoding
XGB_CATEGORICAL_COLS = [
    'NewCreditCustomer', 'VerificationType', 'LanguageCode', 'Gender',
    'Education', 'MaritalStatus', 'NrOfDependants', 'EmploymentStatus',
    'EmploymentDurationCurrentEmployer', 'WorkExperience', 'OccupationArea',
    'HomeOwnershipType', 'UseOfLoan',
]


# ─────────────────────────────────────────────────────────────────────
# THRESHOLDS
# ─────────────────────────────────────────────────────────────────────
IV_THRESHOLD = 0.02        # Minimum IV to keep a feature
VIF_THRESHOLD = 5.0        # Maximum VIF before dropping
MISSINGNESS_DROP_PCT = 90  # Drop features above this % missing
CORRELATION_THRESHOLD = 0.7
RANDOM_STATE = 42