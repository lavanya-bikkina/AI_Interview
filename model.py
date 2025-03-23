import pandas as pd
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import random

# Initialize the Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

name=input("enter your name:")
# Function to calculate semantic similarity between user and correct answers
def calculate_semantic_similarity(user_answer, correct_answer):
    embeddings = model.encode([user_answer, correct_answer])
    cosine_sim = cosine_similarity([embeddings[0]], [embeddings[1]])
    return cosine_sim[0][0]

# Function to load the dataset based on the domain
def get_domain_dataset(domain, uploaded_files):
    domain_normalized = domain.replace(" ", "_").lower()
    for filename in uploaded_files:
        filename_normalized = filename.replace(" ", "_").lower()
        if domain_normalized in filename_normalized:
            try:
                return pd.read_csv(filename)
            except Exception as e:
                print(f"Error loading file {filename}: {e}")
                return None
    print(f"No dataset found for domain: {domain}")
    return None

# Function to ask questions and compare answers
def ask_questions_and_score(selected_dataset):
    total_questions = selected_dataset.shape[0]
    score = 0
    total_similarity_score = 0
    asked_questions = set()

    print("Answer the following questions:")

    def ask_next_question():
        nonlocal score, total_similarity_score, asked_questions

        # Check if we have asked 5 questions or run out of questions
        if len(asked_questions) >= 5 or len(asked_questions) >= total_questions:
            print("name of the interviewee:",name)
            print(f"\nYour total similarity score: {total_similarity_score:.2f}/5")
            rating = (score / 5) * 10
            print(f"Your final rating: {rating:.2f}/10")
            return

        available_questions = selected_dataset.drop(index=asked_questions)
        if available_questions.empty:
            print("No more questions available.")
            return

        # Randomly pick one question
        row = available_questions.sample(1).iloc[0]
        index = row.name
        asked_questions.add(index)

        # Display question and ask for the answer
        print(f"\nQuestion: {row['Question']}")
        user_answer = input("Your Answer: ")

        # Compare with correct answer
        correct_answer = row['Answer']
        similarity = calculate_semantic_similarity(user_answer, correct_answer)
        print(f"Similarity: {similarity:.2f}")

        if similarity >= 0.6:
            print("Correct!")
            score += 1
        else:
            print(f"Incorrect. Correct answer: {correct_answer}")

        total_similarity_score += similarity

        # Proceed to the next question
        ask_next_question()

    # Start the interview with the first question
    ask_next_question()

def main():
    # Directory containing uploaded files (modify this path accordingly)
    uploaded_files = os.listdir()  # Or specify a path like `os.listdir('path_to_folder')`

    print(f"Available files: {uploaded_files}")

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
    
    print("Available domains:")
    for i, domain in enumerate(domain_options, 1):
        print(f"{i}. {domain}")

    # Get the domain selection from the user
    domain_choice = int(input("Select a domain by number: "))
    selected_domain = domain_options[domain_choice - 1]

    # Load the dataset for the selected domain
    dataset = get_domain_dataset(selected_domain, uploaded_files)
    if dataset is not None:
        # Clean the dataset
        if {'Domain', 'Question', 'Answer'}.issubset(dataset.columns):
            dataset_cleaned = dataset.dropna(subset=['Question', 'Answer'])
            selected_dataset = dataset_cleaned[['Domain', 'Question', 'Answer']]
            print(f"Loaded dataset for domain: {selected_domain}")
            print(selected_dataset.head())  # Display the first few rows
            # Start the interview
            ask_questions_and_score(selected_dataset)
        else:
            print(f"The dataset for {selected_domain} is missing required columns: 'Domain', 'Question', 'Answer'.")
    else:
        print(f"Dataset for domain {selected_domain} not found.")

if __name__ == "__main__":
    main()
