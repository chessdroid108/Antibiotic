import os
import tempfile
import logging
from flask import render_template, request, redirect, url_for, flash, jsonify, send_file
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
from werkzeug.exceptions import BadRequest
from datetime import datetime
import uuid

from app import db
from models import User, Project, GenomeData, Result
from amr_analyzer import AMRAnalyzer
from data_preprocessing import preprocess_genomes
from utils import allowed_file, create_archive

logger = logging.getLogger(__name__)

login_manager = LoginManager()

def register_routes(app):
    login_manager.init_app(app)
    
    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(User, int(user_id))
    
    @login_manager.unauthorized_handler
    def unauthorized():
        flash('You must be logged in to access this page', 'warning')
        return redirect(url_for('login'))
    
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/about')
    def about():
        return render_template('about.html')
    
    @app.route('/register', methods=['GET', 'POST'])
    def register():
        if request.method == 'POST':
            username = request.form.get('username')
            email = request.form.get('email')
            password = request.form.get('password')
            
            if not username or not email or not password:
                flash('All fields are required', 'danger')
                return redirect(url_for('register'))
            
            existing_user = User.query.filter_by(username=username).first()
            if existing_user:
                flash('Username already exists', 'danger')
                return redirect(url_for('register'))
            
            existing_email = User.query.filter_by(email=email).first()
            if existing_email:
                flash('Email already registered', 'danger')
                return redirect(url_for('register'))
            
            new_user = User(username=username, email=email)
            new_user.set_password(password)
            
            db.session.add(new_user)
            db.session.commit()
            
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        
        return render_template('register.html')
    
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')
            
            user = User.query.filter_by(username=username).first()
            
            if user and user.check_password(password):
                login_user(user)
                flash('Login successful!', 'success')
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid username or password', 'danger')
        
        return render_template('login.html')
    
    @app.route('/logout')
    @login_required
    def logout():
        logout_user()
        flash('You have been logged out', 'success')
        return redirect(url_for('index'))
    
    @app.route('/dashboard')
    @login_required
    def dashboard():
        projects = Project.query.filter_by(user_id=current_user.id).order_by(Project.updated_at.desc()).all()
        return render_template('dashboard.html', projects=projects)
    
    @app.route('/project/new', methods=['GET', 'POST'])
    @login_required
    def new_project():
        if request.method == 'POST':
            name = request.form.get('name')
            description = request.form.get('description', '')
            
            if not name:
                flash('Project name is required', 'danger')
                return redirect(url_for('new_project'))
            
            project = Project(name=name, description=description, user_id=current_user.id)
            db.session.add(project)
            db.session.commit()
            
            flash('Project created successfully', 'success')
            return redirect(url_for('project_detail', project_id=project.id))
        
        return render_template('new_project.html')
    
    @app.route('/project/<int:project_id>')
    @login_required
    def project_detail(project_id):
        project = Project.query.get_or_404(project_id)
        
        # Ensure the current user owns this project
        if project.user_id != current_user.id:
            flash('You do not have permission to view this project', 'danger')
            return redirect(url_for('dashboard'))
        
        genome_data = GenomeData.query.filter_by(project_id=project_id).all()
        results = Result.query.filter_by(project_id=project_id).all()
        
        return render_template('project_detail.html', 
                               project=project, 
                               genome_data=genome_data, 
                               results=results)
    
    @app.route('/project/<int:project_id>/upload', methods=['GET', 'POST'])
    @login_required
    def upload_genome(project_id):
        project = Project.query.get_or_404(project_id)
        
        # Ensure the current user owns this project
        if project.user_id != current_user.id:
            flash('You do not have permission to modify this project', 'danger')
            return redirect(url_for('dashboard'))
        
        if request.method == 'POST':
            if 'file' not in request.files:
                flash('No file part', 'danger')
                return redirect(request.url)
            
            file = request.files['file']
            
            if file.filename == '':
                flash('No selected file', 'danger')
                return redirect(request.url)
            
            if not allowed_file(file.filename, app.config['ALLOWED_EXTENSIONS']):
                flash(f'Invalid file type. Allowed types: {", ".join(app.config["ALLOWED_EXTENSIONS"])}', 'danger')
                return redirect(request.url)
            
            # Get form data
            is_resistant = request.form.get('is_resistant', 'unknown')
            organism = request.form.get('organism', '')
            antibiotic = request.form.get('antibiotic', '')
            
            # Process resistance status
            if is_resistant == 'yes':
                resistant = True
            elif is_resistant == 'no':
                resistant = False
            else:
                resistant = None
            
            # Create a unique filename
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            # Save the file
            file.save(file_path)
            
            # Determine file type based on extension
            file_ext = filename.rsplit('.', 1)[1].lower()
            if file_ext in ['fa', 'fasta', 'fna']:
                file_type = 'fasta'
            else:
                file_type = file_ext
            
            # Create genome data record
            genome_data = GenomeData(
                filename=filename,
                file_path=file_path,
                file_type=file_type,
                is_resistant=resistant,
                organism=organism,
                antibiotic=antibiotic,
                project_id=project_id
            )
            
            db.session.add(genome_data)
            db.session.commit()
            
            flash('File uploaded successfully', 'success')
            return redirect(url_for('project_detail', project_id=project_id))
        
        return render_template('upload.html', project=project)
    
    @app.route('/project/<int:project_id>/analyze', methods=['GET', 'POST'])
    @login_required
    def analyze_genomes(project_id):
        project = Project.query.get_or_404(project_id)
        
        # Ensure the current user owns this project
        if project.user_id != current_user.id:
            flash('You do not have permission to analyze this project', 'danger')
            return redirect(url_for('dashboard'))
        
        genome_data = GenomeData.query.filter_by(project_id=project_id).all()
        
        if not genome_data:
            flash('You need to upload genome data before analysis', 'warning')
            return redirect(url_for('upload_genome', project_id=project_id))
        
        # Check if we have both resistant and non-resistant strains
        resistant_count = sum(1 for g in genome_data if g.is_resistant is True)
        non_resistant_count = sum(1 for g in genome_data if g.is_resistant is False)
        
        if resistant_count == 0 or non_resistant_count == 0:
            flash('You need both resistant and non-resistant samples for comparison', 'warning')
            return redirect(url_for('project_detail', project_id=project_id))
        
        if request.method == 'POST':
            # Create a new analysis result
            result = Result(
                name=f"Analysis {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
                project_id=project_id
            )
            
            # Save parameters
            parameters = {
                'p_value_threshold': float(request.form.get('p_value_threshold', 0.05)),
                'min_frequency_diff': float(request.form.get('min_frequency_diff', 0.2)),
                'cv_folds': int(request.form.get('cv_folds', 10)),
                'feature_selection': request.form.get('feature_selection', 'xgboost'),
                'classifier': request.form.get('classifier', 'svm')
            }
            result.set_parameters(parameters)
            
            db.session.add(result)
            db.session.commit()
            
            # Start analysis as a background task
            try:
                # Preprocess genomes
                resistant_genomes = [g for g in genome_data if g.is_resistant is True]
                non_resistant_genomes = [g for g in genome_data if g.is_resistant is False]
                
                # Initialize analyzer and run analysis
                analyzer = AMRAnalyzer(result.id, parameters)
                analyzer.run_analysis(resistant_genomes, non_resistant_genomes)
                
                flash('Analysis completed successfully', 'success')
            except Exception as e:
                logger.error(f"Analysis error: {str(e)}")
                flash(f'Error during analysis: {str(e)}', 'danger')
            
            return redirect(url_for('view_result', result_id=result.id))
        
        return render_template('analyze.html', project=project, genome_data=genome_data)
    
    @app.route('/result/<int:result_id>')
    @login_required
    def view_result(result_id):
        result = Result.query.get_or_404(result_id)
        project = Project.query.get_or_404(result.project_id)
        
        # Ensure the current user owns the project for this result
        if project.user_id != current_user.id:
            flash('You do not have permission to view this result', 'danger')
            return redirect(url_for('dashboard'))
        
        # Get result data
        mutations = result.get_mutations()
        stats = result.get_statistics()
        model_perf = result.get_model_performance()
        feature_imp = result.get_feature_importance()
        viz_data = result.get_visualizations()
        
        return render_template('results.html', 
                               result=result, 
                               project=project,
                               mutations=mutations,
                               stats=stats,
                               model_perf=model_perf,
                               feature_imp=feature_imp,
                               viz_data=viz_data)
    
    @app.route('/result/<int:result_id>/export')
    @login_required
    def export_result(result_id):
        result = Result.query.get_or_404(result_id)
        project = Project.query.get_or_404(result.project_id)
        
        # Ensure the current user owns the project for this result
        if project.user_id != current_user.id:
            flash('You do not have permission to export this result', 'danger')
            return redirect(url_for('dashboard'))
        
        # Create a temporary directory for export files
        export_dir = os.path.join(tempfile.gettempdir(), f'export_{result_id}_{uuid.uuid4()}')
        os.makedirs(export_dir, exist_ok=True)
        
        # Create export files
        try:
            # Mutations CSV
            mutations = result.get_mutations()
            with open(os.path.join(export_dir, 'mutations.csv'), 'w') as f:
                f.write('Mutation,Gene,Position,Resistant_Freq,NonResistant_Freq,P_Value,Significance\n')
                for mut in mutations:
                    f.write(f"{mut['mutation']},{mut['gene']},{mut['position']},"
                            f"{mut['resistant_freq']},{mut['non_resistant_freq']},"
                            f"{mut['p_value']},{mut['significant']}\n")
            
            # Model performance
            model_perf = result.get_model_performance()
            with open(os.path.join(export_dir, 'model_performance.txt'), 'w') as f:
                f.write(f"Accuracy: {model_perf.get('accuracy', 'N/A')}\n")
                f.write(f"Precision: {model_perf.get('precision', 'N/A')}\n")
                f.write(f"Recall: {model_perf.get('recall', 'N/A')}\n")
                f.write(f"F1 Score: {model_perf.get('f1_score', 'N/A')}\n")
                f.write(f"ROC AUC: {model_perf.get('roc_auc', 'N/A')}\n")
            
            # Feature importance
            feature_imp = result.get_feature_importance()
            with open(os.path.join(export_dir, 'feature_importance.csv'), 'w') as f:
                f.write('Feature,Importance\n')
                for feat in feature_imp:
                    f.write(f"{feat['feature']},{feat['importance']}\n")
            
            # Create ZIP archive
            zip_path = os.path.join(tempfile.gettempdir(), f'amr_result_{result_id}.zip')
            create_archive(export_dir, zip_path)
            
            return send_file(zip_path, as_attachment=True, download_name=f'amr_result_{result_id}.zip')
            
        except Exception as e:
            logger.error(f"Export error: {str(e)}")
            flash(f'Error during export: {str(e)}', 'danger')
            return redirect(url_for('view_result', result_id=result_id))
        finally:
            # Clean up temporary files
            import shutil
            if os.path.exists(export_dir):
                shutil.rmtree(export_dir)
    
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('404.html'), 404
    
    @app.errorhandler(500)
    def server_error(e):
        return render_template('500.html'), 500
