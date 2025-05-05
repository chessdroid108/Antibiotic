from datetime import datetime
from app import db
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
import json

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    projects = db.relationship('Project', backref='user', lazy=True, cascade="all, delete-orphan")
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'

class Project(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    genome_data = db.relationship('GenomeData', backref='project', lazy=True, cascade="all, delete-orphan")
    results = db.relationship('Result', backref='project', lazy=True, cascade="all, delete-orphan")
    
    def __repr__(self):
        return f'<Project {self.name}>'

class GenomeData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(512), nullable=False)
    file_type = db.Column(db.String(20), nullable=False)  # 'fasta', 'csv', 'tsv', 'json'
    is_resistant = db.Column(db.Boolean, nullable=True)  # Resistance phenotype
    organism = db.Column(db.String(120))
    antibiotic = db.Column(db.String(120))
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    processed = db.Column(db.Boolean, default=False)
    metadata_json = db.Column(db.Text)  # JSON string with additional metadata
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=False)
    
    def get_metadata(self):
        if self.metadata_json:
            return json.loads(self.metadata_json)
        return {}
    
    def set_metadata(self, metadata_dict):
        self.metadata_json = json.dumps(metadata_dict)
    
    def __repr__(self):
        return f'<GenomeData {self.filename}>'

class Result(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=False)
    
    # Analysis parameters
    parameters = db.Column(db.Text)  # JSON string with analysis parameters
    
    # Results data
    mutations = db.Column(db.Text)  # JSON string with identified mutations
    statistics = db.Column(db.Text)  # JSON string with statistical results
    
    # Machine Learning results
    model_performance = db.Column(db.Text)  # JSON with model metrics
    feature_importance = db.Column(db.Text)  # JSON with feature importance scores
    
    # Visualization data
    visualizations = db.Column(db.Text)  # JSON with visualization data
    
    def get_parameters(self):
        if self.parameters:
            return json.loads(self.parameters)
        return {}
    
    def set_parameters(self, params_dict):
        self.parameters = json.dumps(params_dict)
    
    def get_mutations(self):
        if self.mutations:
            return json.loads(self.mutations)
        return []
    
    def set_mutations(self, mutations_list):
        self.mutations = json.dumps(mutations_list)
    
    def get_statistics(self):
        if self.statistics:
            return json.loads(self.statistics)
        return {}
    
    def set_statistics(self, stats_dict):
        self.statistics = json.dumps(stats_dict)
    
    def get_model_performance(self):
        if self.model_performance:
            return json.loads(self.model_performance)
        return {}
    
    def set_model_performance(self, performance_dict):
        self.model_performance = json.dumps(performance_dict)
    
    def get_feature_importance(self):
        if self.feature_importance:
            return json.loads(self.feature_importance)
        return []
    
    def set_feature_importance(self, importance_list):
        self.feature_importance = json.dumps(importance_list)
    
    def get_visualizations(self):
        if self.visualizations:
            return json.loads(self.visualizations)
        return {}
    
    def set_visualizations(self, viz_dict):
        self.visualizations = json.dumps(viz_dict)
    
    def __repr__(self):
        return f'<Result {self.name}>'
