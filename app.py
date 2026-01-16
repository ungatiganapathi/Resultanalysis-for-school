from flask import Flask, render_template, request, flash, redirect, url_for
import pandas as pd
import numpy as np
import os
from tabulate import tabulate
import io

app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        if file and file.filename.endswith('.csv'):
            df = pd.read_csv(file)
            subject_max = float(request.form.get('subject_max', 100))
            ai_max = float(request.form.get('ai_max', 100))
            pass_marks = round(subject_max/3)
            subject_cols_input = request.form.get('subject_cols', '3,4,5,6,7,8').split(',')
            subject_cols = [int(x.strip()) for x in subject_cols_input]
            ai_col = int(request.form.get('ai_col', 9))
            target_cols_names = [df.columns[i] for i in subject_cols if i < len(df.columns)]
            
            # === SUBJECT-WISE ANALYSIS (includes AI) ===
            df_filled = df.fillna(0)
            for col_idx in df_filled.columns:
                if pd.api.types.is_numeric_dtype(df_filled[col_idx]):
                    df_filled[col_idx] = df_filled[col_idx].astype(int)
            
            arr = df_filled.to_numpy(dtype=object)
            n_subjects = len(subject_cols)
            headers_subject = ['Statistics'] + [df.columns[i] for i in subject_cols if i < len(df.columns)]
            if ai_col < len(df.columns):
                headers_subject.append(df.columns[ai_col])
            
            array_subject = np.full((9, len(headers_subject)), '', dtype=object)
            stats_labels = ['Students on roll', 'Appeared', 'Passed', 
                          'A: 0-32.9', 'B: 33-44.9', 'C: 45-59.9', 
                          'D: 60-74.9', 'E: 75-89.9', 'F: 90-100']
            array_subject[:len(stats_labels), 0] = stats_labels
            
            # Subject-wise processing (unchanged - keeps AI)
            for idx, s_col in enumerate(subject_cols):
                if s_col >= arr.shape[1]: continue
                col = arr[:, s_col]
                ab_count = np.sum(np.array(col) == 'AB')
                numeric_marks = np.array([int(x) for x in col if str(x).isdigit() and int(x) > 0])
                zero_count = np.sum(np.array(col, dtype=str) == '0')
                appeared = df.shape[0] - zero_count - ab_count
                percentages = numeric_marks / subject_max * 100
                passed = len(numeric_marks) - np.sum(percentages < pass_marks)
                
                array_subject[0, idx+1] = df.shape[0] - zero_count
                array_subject[1, idx+1] = appeared
                array_subject[2, idx+1] = passed
                
                for j, (low, high) in enumerate([(0,32.9),(33,44.9),(45,59.9),(60,74.9),(75,89.9),(90,100)]):
                    count = np.sum((percentages >= low) & (percentages <= high))
                    array_subject[3+j, idx+1] = int(count)
            
            # AI processing (only for subject-wise table)
            if ai_col < arr.shape[1]:
                col_ai = arr[:, ai_col]
                ab_count_ai = np.sum(np.array(col_ai) == 'AB')
                numeric_ai = np.array([int(x) for x in col_ai if str(x).isdigit() and int(x) > 0])
                zero_ai = np.sum(np.array(col_ai, dtype=str) == '0')
                appeared_ai = df.shape[0] - zero_ai - ab_count_ai
                perc_ai = numeric_ai / ai_max * 100
                passed_ai = len(numeric_ai) - np.sum(perc_ai < pass_marks)
                
                array_subject[0, -1] = df.shape[0]
                array_subject[1, -1] = appeared_ai
                array_subject[2, -1] = passed_ai
                for j, (low, high) in enumerate([(0,32.9),(33,44.9),(45,59.9),(60,74.9),(75,89.9),(90,100)]):
                    count = np.sum((perc_ai >= low) & (perc_ai <= high))
                    array_subject[3+j, -1] = int(count)
            
            subject_table = tabulate(array_subject, headers=headers_subject, tablefmt='html', showindex=False)
            
            # === OVERALL RESULT ANALYSIS (6 SUBJECTS ONLY - NO AI) ===
            mask_ab = df_filled.map(lambda x: 'AB' in str(x)).any(axis=1)
            df_present = df_filled[~mask_ab]
            num_absent = df_filled.shape[0] - df_present.shape[0]
            
            # Convert to numeric and filter failures (1-26 marks in ANY of 6 subjects)
            df_numeric = df_present.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
            fail_mask = ((df_numeric[target_cols_names] > 0) & (df_numeric[target_cols_names] < pass_marks)).any(axis=1)
            df_passed = df_numeric[~fail_mask]
            num_passed = df_passed.shape[0]
            num_failed = df_present.shape[0] - num_passed
            
            # Overall % = Average of 6 MAIN SUBJECTS ONLY (ENG,HIN,SKT,MATH,SCI,SST)
            if len(target_cols_names) >= 6:  # Ensure we have all 6 subjects
                total_marks = df_passed[target_cols_names].sum(axis=1)
                total_percent = (total_marks / (5 * subject_max) * 100).round(2)
            else:
                total_percent = np.array([])
                num_passed = 0
            
            # Overall grade distribution
            array_overall = np.empty((10, 2), dtype=object)
            stats_overall = np.array(['Students Present', 'Students on Roll', 'Students Passed', 
                                    'Students Failed', 'A: 0-32.9', 'B: 33-44.9', 'C: 45-59.9',
                                    'D: 60-74.9', 'E: 75-89.9', 'F: 90-100'])
            array_overall[:,0] = stats_overall
            array_overall[0,1] = df_present.shape[0]
            array_overall[1,1] = df_filled.shape[0]
            array_overall[2,1] = num_passed
            array_overall[3,1] = num_failed
            
            if len(total_percent) > 0:
                for j, (low, high) in enumerate([(0,32.9),(33,44.9),(45,59.9),(60,74.9),(75,89.9),(90,100)]):
                    count = np.sum((total_percent >= low) & (total_percent <= high))
                    if j==0:
                        array_overall[4+j, 1] = num_failed
                    else:
                         array_overall[4+j, 1] = int(count)
            else:
                for j in range(6):
                    array_overall[4+j, 1] = 0
            
            overall_table = tabulate(array_overall, headers=['Statistics', 'Overall (6 Subjects)'], tablefmt='html', showindex=False)
            
            return render_template('result.html', 
                                 subject_table=subject_table, 
                                 overall_table=overall_table,
                                 df_shape=df.shape,
                                 stats={'onroll': df.shape[0], 'absent': num_absent, 'passed': num_passed, 'failed': num_failed})
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
