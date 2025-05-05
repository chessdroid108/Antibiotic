import logging
from app import db
from models import User, Project, GenomeData, Result
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Database access layer for managing data persistence.
    """
    
    @classmethod
    def create_user(cls, username, email, password):
        """
        Create a new user.
        
        Args:
            username (str): Username
            email (str): Email address
            password (str): Password
            
        Returns:
            User: Created user object or None if failed
        """
        try:
            user = User(username=username, email=email)
            user.set_password(password)
            
            db.session.add(user)
            db.session.commit()
            
            return user
        
        except SQLAlchemyError as e:
            logger.error(f"Error creating user: {str(e)}")
            db.session.rollback()
            return None
    
    @staticmethod
    def get_user_by_username(username):
        """
        Get user by username.
        
        Args:
            username (str): Username
            
        Returns:
            User: User object or None if not found
        """
        try:
            return User.query.filter_by(username=username).first()
        
        except SQLAlchemyError as e:
            logger.error(f"Error getting user by username: {str(e)}")
            return None
    
    @staticmethod
    def get_user_by_email(email):
        """
        Get user by email.
        
        Args:
            email (str): Email address
            
        Returns:
            User: User object or None if not found
        """
        try:
            return User.query.filter_by(email=email).first()
        
        except SQLAlchemyError as e:
            logger.error(f"Error getting user by email: {str(e)}")
            return None
    
    @staticmethod
    def create_project(name, description, user_id):
        """
        Create a new project.
        
        Args:
            name (str): Project name
            description (str): Project description
            user_id (int): User ID
            
        Returns:
            Project: Created project object or None if failed
        """
        try:
            project = Project(name=name, description=description, user_id=user_id)
            
            db.session.add(project)
            db.session.commit()
            
            return project
        
        except SQLAlchemyError as e:
            logger.error(f"Error creating project: {str(e)}")
            db.session.rollback()
            return None
    
    @staticmethod
    def get_projects_by_user(user_id):
        """
        Get all projects for a user.
        
        Args:
            user_id (int): User ID
            
        Returns:
            list: List of Project objects
        """
        try:
            return Project.query.filter_by(user_id=user_id).all()
        
        except SQLAlchemyError as e:
            logger.error(f"Error getting projects for user: {str(e)}")
            return []
    
    @staticmethod
    def get_project(project_id):
        """
        Get project by ID.
        
        Args:
            project_id (int): Project ID
            
        Returns:
            Project: Project object or None if not found
        """
        try:
            return Project.query.get(project_id)
        
        except SQLAlchemyError as e:
            logger.error(f"Error getting project: {str(e)}")
            return None
    
    @staticmethod
    def add_genome_data(project_id, filename, file_path, file_type, is_resistant=None, 
                        organism=None, antibiotic=None, metadata=None):
        """
        Add genome data to a project.
        
        Args:
            project_id (int): Project ID
            filename (str): Original filename
            file_path (str): Path to saved file
            file_type (str): File type ('fasta', 'csv', 'tsv', 'json')
            is_resistant (bool): Whether this sample is resistant
            organism (str): Organism name
            antibiotic (str): Antibiotic name
            metadata (dict): Additional metadata
            
        Returns:
            GenomeData: Created genome data object or None if failed
        """
        try:
            genome_data = GenomeData(
                project_id=project_id,
                filename=filename,
                file_path=file_path,
                file_type=file_type,
                is_resistant=is_resistant,
                organism=organism,
                antibiotic=antibiotic
            )
            
            if metadata:
                genome_data.set_metadata(metadata)
            
            db.session.add(genome_data)
            db.session.commit()
            
            return genome_data
        
        except SQLAlchemyError as e:
            logger.error(f"Error adding genome data: {str(e)}")
            db.session.rollback()
            return None
    
    @staticmethod
    def get_genome_data(project_id):
        """
        Get all genome data for a project.
        
        Args:
            project_id (int): Project ID
            
        Returns:
            list: List of GenomeData objects
        """
        try:
            return GenomeData.query.filter_by(project_id=project_id).all()
        
        except SQLAlchemyError as e:
            logger.error(f"Error getting genome data: {str(e)}")
            return []
    
    @staticmethod
    def create_result(project_id, name, parameters=None):
        """
        Create a new analysis result.
        
        Args:
            project_id (int): Project ID
            name (str): Result name
            parameters (dict): Analysis parameters
            
        Returns:
            Result: Created result object or None if failed
        """
        try:
            result = Result(project_id=project_id, name=name)
            
            if parameters:
                result.set_parameters(parameters)
            
            db.session.add(result)
            db.session.commit()
            
            return result
        
        except SQLAlchemyError as e:
            logger.error(f"Error creating result: {str(e)}")
            db.session.rollback()
            return None
    
    @staticmethod
    def get_results(project_id):
        """
        Get all results for a project.
        
        Args:
            project_id (int): Project ID
            
        Returns:
            list: List of Result objects
        """
        try:
            return Result.query.filter_by(project_id=project_id).all()
        
        except SQLAlchemyError as e:
            logger.error(f"Error getting results: {str(e)}")
            return []
    
    @staticmethod
    def get_result(result_id):
        """
        Get result by ID.
        
        Args:
            result_id (int): Result ID
            
        Returns:
            Result: Result object or None if not found
        """
        try:
            return Result.query.get(result_id)
        
        except SQLAlchemyError as e:
            logger.error(f"Error getting result: {str(e)}")
            return None
    
    @staticmethod
    def update_result(result_id, mutations=None, statistics=None, 
                      model_performance=None, feature_importance=None, 
                      visualizations=None):
        """
        Update an analysis result.
        
        Args:
            result_id (int): Result ID
            mutations (list): Identified mutations
            statistics (dict): Statistical results
            model_performance (dict): Model performance metrics
            feature_importance (list): Feature importance scores
            visualizations (dict): Visualization data
            
        Returns:
            Result: Updated result object or None if failed
        """
        try:
            result = Result.query.get(result_id)
            
            if not result:
                logger.error(f"Result with ID {result_id} not found")
                return None
            
            if mutations:
                result.set_mutations(mutations)
            
            if statistics:
                result.set_statistics(statistics)
            
            if model_performance:
                result.set_model_performance(model_performance)
            
            if feature_importance:
                result.set_feature_importance(feature_importance)
            
            if visualizations:
                result.set_visualizations(visualizations)
            
            db.session.commit()
            
            return result
        
        except SQLAlchemyError as e:
            logger.error(f"Error updating result: {str(e)}")
            db.session.rollback()
            return None
