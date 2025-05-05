import os
import tempfile

# Database configuration
SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL", f"sqlite:///{tempfile.gettempdir()}/amr_analysis.db")
SQLALCHEMY_TRACK_MODIFICATIONS = False
SQLALCHEMY_ENGINE_OPTIONS = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Upload configuration
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'amr_uploads')
ALLOWED_EXTENSIONS = {'fa', 'fasta', 'fna', 'csv', 'tsv', 'json'}
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB limit

# External API keys and resources
CARD_API_URL = os.environ.get("CARD_API_URL", "https://card.mcmaster.ca/download/0/broad-nucleotide.fa")
NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")
PDB_API_URL = "https://data.rcsb.org/rest/v1/core/entry/"

# Machine learning parameters
XGBOOST_PARAMS = {
    'learning_rate': 0.1,
    'max_depth': 5,
    'n_estimators': 100,
    'objective': 'binary:logistic'
}

SVM_PARAMS = {
    'C': 1.0,
    'kernel': 'rbf',
    'probability': True
}

# Cross-validation settings
CV_FOLDS = 10

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
