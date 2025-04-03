from flask import Flask, request, jsonify, render_template
from transformers import pipeline
import importlib

app = Flask(__name__)

# Function to check if PyTorch is installed
def check_framework():
    if not importlib.util.find_spec("torch"):
        raise ImportError("PyTorch is not installed. Please install it using 'pip install torch'.")

# Check PyTorch and load the AI model for Q&A
check_framework()
qa_pipeline = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')

# ✅ Enhanced Gardening Knowledge Base (More Detailed Context)
gardening_context = (
    "Gardening involves growing plants, including flowers, vegetables, and fruit trees. "
    "Roses require well-drained soil and at least six hours of direct sunlight daily. "
    "Vegetable gardens need organic compost and balanced fertilizers like NPK 10-10-10. "
    "Fruit trees should be watered deeply once a week and pruned regularly. "
    "Composting enriches soil fertility, and crop rotation helps prevent soil depletion. "
    "Seasonal planting enhances growth, and understanding plant compatibility reduces pests. "
    "Tomatoes grow best in full sunlight with well-draining soil and require regular watering. "
    "Carrots prefer loose, sandy soil with deep watering to encourage root growth. "
    "Lettuce grows best in cooler temperatures and needs regular moisture. "
    "Cucumbers require trellising for best growth and need frequent watering. "
    "Potted plants need well-draining soil and should be watered when the topsoil is dry. "
    "Mulching helps retain soil moisture and suppresses weeds. "
    "Organic fertilizers such as composted manure and bone meal improve soil fertility. "
    "Overwatering can cause root rot, while underwatering leads to wilting. "
    "Pruning fruit trees in winter encourages healthy growth. "
    "Pollinators like bees and butterflies help plants produce fruit. "
    "Marigolds can help repel pests in vegetable gardens. "
    "Indoor plants such as succulents need indirect sunlight and minimal watering. "
    "Hydroponic gardening allows plants to grow without soil, using nutrient-rich water. "
    "Gardening in raised beds helps improve soil drainage and makes weeding easier. "
    "Companion planting, such as growing basil with tomatoes, helps improve plant health. "
    "Drip irrigation is an efficient way to water plants while conserving water."
)

# List of allowed gardening-related keywords
gardening_keywords = ["garden", "plant", "soil", "water", "sunlight", "compost", "fertilizer", 
                      "pruning", "roses", "vegetables", "fruit", "mulch", "pollination", "seeds",
                      "hydroponic", "irrigation", "weeds", "drainage", "pests", "indoor plants","rose","carrots"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_question = data.get("question", "").lower()

    # Check if the question is related to gardening
    if not any(keyword in user_question for keyword in gardening_keywords):
        return jsonify({"response": "🌿 This chatbot only answers gardening-related questions!"})

    # Use the AI model to generate a meaningful answer from the gardening context
    response = qa_pipeline(question=user_question, context=gardening_context)

    return jsonify({"response": response["answer"]})

if __name__ == "__main__":
    app.run(debug=True)
