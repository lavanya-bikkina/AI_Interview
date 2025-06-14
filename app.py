import pandas as pd
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request, jsonify, redirect, session, url_for
import random
import re
import google.generativeai as genai


app = Flask(__name__)
app.secret_key = 'your_secret_key'
# Set up Gemini API key
genai.configure(api_key="AIzaSyBJGwt8Vwr4nU0rcAnopelqibbPmSO5IlU")

def get_gemini_model():
    return genai.GenerativeModel("gemini-1.5-pro")

gemini_model = get_gemini_model()

model = SentenceTransformer('all-MiniLM-L6-v2')
# Timer settings for each question (in seconds)
QUESTION_TIMER = 60

# Global variables for tracking score and progress
score = 0
total_questions = 10
questions = []
current_question_index = 0
total_similarity_score = 0
user_answers = []  # Store user answers

# Function to calculate semantic similarity using Gemini API

# Function to calculate semantic similarity using Gemini API
def calculate_gemini_similarity(user_answer, correct_answer, question_text):
    try:
        # If the answer is exactly the same as the question, return 0
        if user_answer.strip().lower() == question_text.strip().lower():
            return 0.0  # Direct copy-paste of the question

        prompt = (
            f"Evaluate how well the user answer matches the correct answer based on meaning. "
            f"If the user has simply copied the question or if user answer is similar to question or written an unrelated response, return a score close to 0. "
            f"If the user answer is having the same meaning as correct answer and contains all the points of correct answer, give a score close to 1.\n\n"
            f"User Answer: {user_answer}\n"
            f"Correct Answer: {correct_answer}\n"
            f"Provide a similarity score between 0 and 1 (strict evaluation):"
        )

        response = gemini_model.generate_content(prompt)
        match = re.search(r"(\d*\.?\d+)", response.text)
        
        if match:
            score = float(match.group(1))
            return max(0, min(score, 1))  # Ensure score is within [0,1]
        else:
            print(f"Unexpected response from Gemini: {response.text}")
            return calculate_semantic_similarity(user_answer, correct_answer)  # Fallback to SBERT
    except Exception as e:
        print(f"Error using Gemini API: {e}")
        return calculate_semantic_similarity(user_answer, correct_answer)



# Fallback function for semantic similarity calculation
def calculate_semantic_similarity(user_answer, correct_answer):
    embeddings = model.encode([user_answer, correct_answer])
    cosine_sim = cosine_similarity([embeddings[0]], [embeddings[1]])
    return cosine_sim[0][0]


# Function to load dataset based on domain
def get_domain_dataset(domain, uploaded_files):
    domain_normalized = domain.replace(" ", "_").lower()
    for filename in uploaded_files:
        filename_normalized = filename.replace(" ", "_").lower()
        if domain_normalized in filename_normalized:
            try:
                return pd.read_csv(os.path.join(os.getcwd(), filename))
            except Exception as e:
                print(f"Error loading file {filename}: {e}")
                return None
    return None

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/details', methods=['GET', 'POST'])
def details():
    if request.method == 'POST':
        session['name'] = request.form['name']
        session['college'] = request.form['college']
        session['reg_no'] = request.form['reg_no']
        session['branch'] = request.form['branch']
        return redirect(url_for('index'))
    return render_template('details.html')

@app.route('/index', methods=['GET', 'POST'])    
def index():
    domain_options = [
        "Software Development", "Data_Science", "Machine Learning", "Artificial_Intelligence",
        "Data_Engineer", "Business Intelligence", "Cloud Computing", "Cybersecurity",
        "DevOps", "Networking", "Database Administration", "System Administration",
        "Full Stack Development", "Front-End Development", "Back-End Development",
        "Quality Assurance (QA)", "Game Development", "Mobile App Development",
        "UX UI_Design", "Product Management", "Project Management", "Finance or Quantitative Analysis",
        "Digital Marketing", "Human Resources", "Sales", "Customer Support",
        "Operations Management", "Healthcare", "Research and Development", "Consulting",
        "Legal and Compliance", "Education and Training", "Blockchain", "Internet of Things (IoT)",
        "Robotics", "Ethical Hacking and Penetration Testing", "Business Analysis", "Accounting",
        "Product Design", "Supply Chain Management", "Public Relations", "Technical Writing",
        "Social Media Management", "Sustainability and Environmental Management",
        "Hospitality and Event Management", "Vlsi", "Embedded", "Cloud Architect",
        "Chartered Accountant", "Excel Expert", "Chemical Engineering", "Quantum Computing","Chemistry"
    ]
    domain_options.sort()
    return render_template('index.html', domain_options=domain_options)

@app.route('/start_interview', methods=['POST'])
def start_interview():
    global questions, score, current_question_index, total_similarity_score, user_answers
    score = 0
    current_question_index = 0
    total_similarity_score = 0
    user_answers = []

    selected_domain = request.form['domain']
    uploaded_files = os.listdir(os.getcwd())
    dataset = get_domain_dataset(selected_domain, uploaded_files)
    
    if dataset is not None:
        if {'Domain', 'Question', 'Answer'}.issubset(dataset.columns):
            dataset_cleaned = dataset.dropna(subset=['Question', 'Answer'])
            selected_dataset = dataset_cleaned[['Domain', 'Question', 'Answer']]
            questions = selected_dataset.sample(n=5, replace=False).to_dict(orient='records')
            return render_template('interview.html', question=questions[0], question_index=current_question_index, timer=QUESTION_TIMER)
        else:
            return "Dataset missing required columns."
    else:
        return "Dataset not found for the selected domain."

@app.route('/submit_answer', methods=['POST'])
def submit_answer():
    global score, current_question_index, total_similarity_score, user_answers
    try:
        user_answer = request.form.get('user_answer', "Skipped")
        question_index = int(request.form['question_index'])
        correct_answer = questions[question_index]['Answer']
        question_text = questions[question_index]['Question']

        user_answers.append({'question': question_text, 'user_answer': user_answer, 'correct_answer': correct_answer})

        # Handle skipping
        if user_answer == "":
            result = "Skipped"
            similarity = 0.0  # No similarity score if skipped
        else:
            similarity = float(calculate_gemini_similarity(user_answer, correct_answer, question_text))

            # Adjust scoring criteria: stricter threshold
            if similarity >= 0.75:  # Increase threshold to 75%
                score += 1
                result = "Correct!"
            else:
                result = "Incorrect."

        total_similarity_score += similarity

        # Move to the next question or finish the interview
        current_question_index += 1
        if current_question_index < total_questions:
            next_question = questions[current_question_index]
            return jsonify({
                'similarity': round(similarity, 2),
                'result': result,
                'next_question': next_question['Question'],
                'question_index': current_question_index,
                'timer': QUESTION_TIMER
            })
        else:
            # Calculate total similarity score for the entire interview
            similarity_score = total_similarity_score / total_questions
            return jsonify({
                'similarity': round(similarity_score, 2),
                'result': result,
                'finished': True,
                'score': score,
                'total': total_questions,
                'total_similarity_score': round(similarity_score, 2)
            })
    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500


@app.route('/show_scores')
def show_scores():
    final_score = score
    average_similarity_score = round(total_similarity_score / total_questions, 2) if total_questions > 0 else 0
    return render_template('scores.html', final_score=final_score, average_similarity_score=average_similarity_score, user_answers=user_answers)

if __name__ == '__main__':
    app.run(debug=True)
