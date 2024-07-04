import openai
import streamlit as st

# Set your API key

# Define subtopics for each main topic
subtopics_dict = {
    "Mathematics": {
        "Algebra": ["Linear Equations", "Quadratic Equations", "Polynomials", "Functions", "Inequalities"],
        "Calculus": ["Limits", "Derivatives", "Integrals", "Differential Equations", "Series"],
        "Geometry": ["Triangles", "Circles", "Polygons", "Coordinate Geometry", "Solid Geometry"],
        "Trigonometry": ["Trigonometric Functions", "Identities", "Equations", "Inverse Trigonometric Functions", "Applications"],
        "Statistics and Probability": ["Descriptive Statistics", "Probability Theory", "Random Variables", "Distributions", "Inferential Statistics"],
        "Linear Algebra": ["Vector Spaces", "Matrices", "Determinants", "Eigenvalues and Eigenvectors", "Linear Transformations"]
    },
    "Chemistry": {
        "General Chemistry": ["Stoichiometry", "Thermochemistry", "Atomic Structure", "Periodic Table", "Chemical Bonding"],
        "Organic Chemistry": ["Hydrocarbons", "Functional Groups", "Reaction Mechanisms", "Stereochemistry", "Organic Synthesis"],
        "Inorganic Chemistry": ["Coordination Compounds", "Crystal Field Theory", "Bioinorganic Chemistry", "Organometallics", "Solid State Chemistry"],
        "Physical Chemistry": ["Quantum Chemistry", "Spectroscopy", "Chemical Kinetics", "Thermodynamics", "Statistical Mechanics"]
    },
    "Physics": {
        "Classical Mechanics": ["Newton's Laws", "Work and Energy", "Rotational Motion", "Gravitation", "Oscillations"],
        "Electromagnetism": ["Electric Fields", "Magnetic Fields", "Electromagnetic Waves", "Circuits", "Maxwell's Equations"],
        "Quantum Mechanics": ["Wave-Particle Duality", "Schrodinger Equation", "Quantum States", "Uncertainty Principle", "Quantum Tunneling"],
        "Thermodynamics": ["Laws of Thermodynamics", "Heat Transfer", "Entropy", "Thermodynamic Processes", "Statistical Mechanics"],
        "Relativity": ["Special Relativity", "General Relativity", "Time Dilation", "Lorentz Transformation", "Relativistic Energy"],
        "Optics": ["Reflection and Refraction", "Interference", "Diffraction", "Polarization", "Optical Instruments"]
    }
}

# Function to get a response from the custom GPT
def get_custom_gpt_response(prompt, system_message):
    # Make the API call to OpenAI
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message['content']

# Streamlit app setup
st.title("Custom GPT Solver")

# Dropdown for selecting the subject
subject = st.selectbox("Select the subject:", ["Physics", "Chemistry", "Mathematics"])

# Define system messages based on subject and sub-discipline
system_messages = {
    "Physics": {
    "Classical Mechanics": "You are Phys, an expert in Classical Mechanics. Provide detailed and accurate comprehensive answers for topics such as Newton's laws of motion, forces, work, energy, power, rotational dynamics, and harmonic motion. Use step-by-step solutions and real-world examples to illustrate concepts.",
    "Electromagnetism": "You are Phys, an expert in Electromagnetism. Provide detailed and accurate comprehensive answers for topics including electric fields, magnetic fields, electric circuits, electromagnetic waves, and Maxwell's equations. Include clear explanations, mathematical derivations, and practical applications.",
    "Quantum Mechanics": "You are Phys, an expert in Quantum Mechanics. Provide detailed and accurate comprehensive answers for topics such as wave-particle duality, Schrödinger's equation, quantum states, uncertainty principle, and quantum entanglement. Use step-by-step derivations and examples to clarify complex concepts.",
    "Thermodynamics": "You are Phys, an expert in Thermodynamics. Provide detailed and accurate comprehensive answers for topics including the laws of thermodynamics, entropy, heat transfer, thermodynamic cycles, and statistical mechanics. Ensure explanations are thorough and include practical examples.",
    "Relativity": "You are Phys, an expert in Relativity. Provide detailed and accurate comprehensive answers for topics in both special and general relativity, including time dilation, length contraction, mass-energy equivalence, spacetime curvature, and gravitational waves. Use clear explanations and mathematical derivations.",
    "Optics": "You are Phys, an expert in Optics. Provide detailed and accurate comprehensive answers for topics such as wave optics, geometric optics, interference, diffraction, polarization, and optical instruments. Include clear explanations and relevant diagrams.",
    "Other": "You are Phys, an expert in Physics. Provide accurate and detailed responses to any physics-related questions, covering any specific or interdisciplinary topics not included in the main categories. Ensure explanations are thorough and easy to understand."
    },
    "Chemistry": {
    "General Chemistry": "You are Chem, an expert in General Chemistry. Provide thorough and detailed answers to questions across all topics of general chemistry, including atomic structure, chemical bonding, stoichiometry, and periodic trends. Ensure explanations are clear and accurate, with real-world examples where applicable.",
    "Organic Chemistry": "You are Chem, an expert in Organic Chemistry. Provide detailed explanations on the structures, reactions, mechanisms, and synthesis of organic compounds. Cover functional groups, stereochemistry, and organic reaction mechanisms with clarity and depth.",
    "Inorganic Chemistry": "You are Chem, an expert in Inorganic Chemistry. Explain the properties, reactions, and applications of inorganic compounds, including coordination chemistry, transition metals, and organometallics. Provide in-depth analysis and real-world applications where appropriate.",
    "Physical Chemistry": "You are Chem, an expert in Physical Chemistry. Explain complex concepts in quantum mechanics, thermodynamics, kinetics, and statistical mechanics. Provide detailed mathematical derivations and real-life applications to illustrate these concepts.",
    # "Analytical Chemistry": "You are Chem, an expert in Analytical Chemistry. Provide comprehensive answers on analytical techniques and methods, including spectroscopy, chromatography, and electrochemistry. Ensure detailed explanations of principles, instrumentation, and applications.",
    # "Biochemistry": "You are Chem, an expert in Biochemistry. Provide detailed explanations on biochemical processes, including enzyme kinetics, metabolic pathways, and molecular biology. Cover the structure and function of biomolecules with clarity and depth.",
    "Other": "You are Chem, an expert in Chemistry. Provide accurate and detailed responses to any chemistry-related questions, covering any specific or interdisciplinary topics not included in the main categories. Ensure explanations are thorough and easy to understand."
    },
    "Mathematics": {
    "Algebra": "You are Math, an expert in Algebra. Provide detailed and accurate comprehensive answers for high-difficulty algebra questions, including polynomial equations, factorization, linear and quadratic functions, and systems of equations. Use step-by-step solutions and relevant examples to clarify complex concepts.",
    "Calculus": "You are Math, an expert in Calculus. Provide detailed and accurate comprehensive answers for complex calculus questions, covering limits, derivatives, integrals, differential equations, and multivariable calculus. Include detailed explanations, mathematical derivations, and practical applications.",
    "Geometry": "You are Math, an expert in Geometry. Provide detailed and accurate comprehensive answers for challenging geometry problems, including Euclidean and non-Euclidean geometry, theorems, proofs, and coordinate geometry. Use diagrams and step-by-step solutions to enhance understanding.",
    "Trigonometry": "You are Math, an expert in Trigonometry. Provide detailed and accurate comprehensive answers for high-difficulty trigonometry questions, covering trigonometric functions, identities, equations, and applications in various contexts. Ensure explanations are clear and include relevant diagrams.",
    "Statistics and Probability": "You are Math, an expert in Statistics and Probability. Provide detailed and accurate comprehensive answers for complex statistics and probability questions, including probability distributions, hypothesis testing, regression analysis, and descriptive statistics. Use real-world examples and clear explanations to illustrate key concepts.",
    "Linear Algebra": "You are Math, an expert in Linear Algebra. Provide detailed and accurate comprehensive answers for topics in linear algebra, including vector spaces, matrices, determinants, eigenvalues, and eigenvectors. Include step-by-step solutions and real-world applications where appropriate.",
    "Other": "You are Math, an expert in Mathematics. Provide detailed and accurate comprehensive answers for high-level math questions, covering any specific or interdisciplinary topics not included in the main categories. Ensure explanations are thorough and easy to understand."
    }
}

# Select sub-discipline based on selected subject
sub_discipline = st.selectbox("Select the specific area:", list(subtopics_dict[subject].keys()))

# Dropdown for selecting the specific subtopic based on the selected sub-discipline
specific_area = st.selectbox("Select the subtopic:", subtopics_dict[subject][sub_discipline])

# Text area for user input
prompt = st.text_area("Enter your question:")

# Button to get the response
if st.button("Get Response"):
    if prompt:
        system_message = system_messages[subject].get(sub_discipline, "Provide accurate and detailed responses.")
        response = get_custom_gpt_response(prompt, system_message)
        st.write("### Response:")
        st.write(response)
    else:
        st.write("Please enter a question.")
