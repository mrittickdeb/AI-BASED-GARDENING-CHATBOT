from flask import Flask, request, jsonify, render_template
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import importlib
import re

app = Flask(__name__)

# ✅ Check PyTorch is installed
if not importlib.util.find_spec("torch"):
    raise ImportError("Please install PyTorch: pip install torch")

# ✅ Load transformer model
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# ✅ Gardening context and keywords
gardening_context = """
Gardening involves growing plants, including flowers, vegetables, herbs, and fruit trees.

=== FLOWERS ===
Roses require well-drained soil (pH 6.0-6.5) and at least six hours of direct sunlight daily. Prune in early spring and deadhead regularly.
Lilies prefer well-drained, slightly acidic soil and grow best in full to partial sunlight. Plant bulbs 4-6 inches deep in fall or spring.
Sunflowers need full sun and should be planted after last frost. They grow quickly and can reach 6-12 feet tall.
Tulips are spring-blooming bulbs that should be planted in fall, 6-8 inches deep in well-drained soil.
Hydrangeas change color based on soil pH - blue in acidic soil (pH <5.5), pink in alkaline (pH >6.5).

=== VEGETABLES ===
Vegetables are the unsung heroes of nutrition and flavor! They come in a dazzling variety, from leafy greens like spinach and kale, to root vegetables like carrots and radishes, to fruiting vegetables like tomatoes and bell peppers
Tomatoes grow best in full sunlight with well-draining soil (pH 6.0-6.8). Water 1-2 inches weekly and support with stakes or cages.
Carrots prefer loose, sandy soil with deep watering to encourage root growth. Sow seeds 1/4 inch deep, 2 inches apart.
Peppers need warm soil (70°F+) and consistent moisture. Harvest when fruits reach full size and color.
Zucchini is a prolific summer squash that needs plenty of space (plant 3 feet apart) and regular harvesting.
Leafy greens like lettuce and spinach prefer cooler temperatures (60-70°F) and moist, fertile soil.

=== FRUITS ===
Strawberries grow best in full sun with rich, well-drained soil. Plant in spring and mulch around plants.
Blueberries need acidic soil (pH 4.0-5.5) and consistent moisture. Prune old canes in winter.
Apple trees require cross-pollination between different varieties. Prune in late winter to maintain shape.
Citrus trees (lemons, oranges) need warm climates (or containers in cold areas) and well-drained soil.

=== HERBS ===
Basil grows best in warm weather and full sun. Pinch off flowers to encourage leaf growth.
Mint prefers partial shade and moist soil. It spreads aggressively, so contain in pots.
Rosemary needs well-drained soil and full sun. It's drought-tolerant once established.
Parsley grows well in both sun and partial shade. Soak seeds overnight before planting.

=== GARDENING TECHNIQUES ===
Composting requires a balance of green (nitrogen-rich) and brown (carbon-rich) materials. Turn pile every 2-4 weeks.
Mulching with 2-4 inches of organic material helps retain moisture and suppress weeds.
Crop rotation prevents soil depletion and reduces pest/disease problems. Rotate plant families annually.
Raised beds improve drainage, reduce weeds, and allow better soil control. Ideal height is 12-18 inches.
Drip irrigation conserves water by delivering it directly to plant roots. Reduces leaf diseases.
Container gardening is ideal for small spaces. Use pots with drainage holes and quality potting mix.
"""

# ✅ Expanded Gardening Keywords
gardening_keywords = [
    # Flowers
    "rose", "lily", "sunflower", "tulip", "daffodil", "hydrangea", "peony", 
    "dahlia", "marigold", "petunia", "zinnia", "geranium", "lavender",
    
    # Vegetables
    "vegetables", "tomato", "carrot", "pepper", "zucchini", "cucumber", "lettuce", "spinach",
    "kale", "broccoli", "cauliflower", "bean", "pea", "radish", "beet", "onion",
    
    # Fruits
    "strawberry", "blueberry", "raspberry", "apple", "pear", "peach", "plum",
    "cherry", "citrus", "lemon", "orange", "grape", "melon", "watermelon",
    
    # Herbs
    "basil", "mint", "rosemary", "thyme", "oregano", "parsley", "cilantro",
    "dill", "chive", "sage", "lemongrass",
    
    # Techniques
    "compost", "mulch", "prune", "fertilize", "water", "irrigate", "soil",
    "raised bed", "container", "seed", "transplant", "harvest", "pollinate",
    "germinate", "propagate", "hardening off", "crop rotation", "companion planting",
    
    # General
    "gardening", "organic", "bloom", "flowering", "vegetable", "fruit", "herb",
    "perennial", "annual", "zone", "hardiness", "sunlight", "shade", "drainage",
    "beginner", "beginners", "easy", "simple"
]

# ✅ Expanded Fallback Answers
fallback_answers = {
    # Flowers
    "rose": "Roses need well-drained soil (pH 6.0-6.5), 6+ hours of sun, and deep watering 2-3 times weekly. Prune in early spring.",
    "lily": "Lilies prefer well-drained soil, full to partial sun. Plant bulbs 4-6\" deep in fall/spring. Water regularly but avoid soggy soil.",
    "sunflower": "Sunflowers need full sun and well-drained soil. Plant after last frost, 1\" deep. Water deeply but infrequently.",
    "tulip": "Tulip bulbs should be planted 6-8\" deep in fall. They need well-drained soil and prefer cool winters.",
    
    # Vegetables
    "vegetables": "Vegetables are the unsung heroes of nutrition and flavor! They come in a dazzling variety, from leafy greens like spinach and kale, to root vegetables like carrots and radishes, to fruiting vegetables like tomatoes and bell peppers.",
    "tomato": "Tomatoes require full sun (6-8 hours), well-drained soil, and 1-2\" water weekly. Support with stakes/cages.",
    "carrot": "Carrots grow best in loose, sandy soil. Sow seeds 1/4\" deep, thin to 2\" apart. Keep soil consistently moist.",
    "pepper": "Peppers need warm soil (70°F+), full sun, and consistent moisture. Fertilize lightly every 2-3 weeks.",
    "zucchini": "Zucchini plants need 3\' spacing, full sun, and regular watering. Harvest when 6-8\" long.",
    
    # Fruits
    "strawberry": "Strawberries need full sun, well-drained soil, and regular watering. Mulch around plants to keep fruit clean.",
    "blueberry": "Blueberries require acidic soil (pH 4.0-5.5), full sun, and consistent moisture. Prune old canes in winter.",
    "apple": "Apple trees need full sun, well-drained soil, and cross-pollination between varieties. Prune in late winter.",
    
    # Herbs
    "basil": "Basil grows best in warm weather (70-80°F) and full sun. Pinch off flowers to encourage leaf production.",
    "mint": "Mint prefers partial shade and moist soil. It spreads aggressively - best grown in containers.",
    "rosemary": "Rosemary needs full sun and well-drained soil. It's drought-tolerant once established.",
    "best herbs for beginners": "Great beginner herbs include:\n- Basil (easy to grow from seed)\n- Mint (hardy but grows aggressively)\n- Chives (perennial and low-maintenance)\n- Parsley (biennial and versatile)\n- Thyme (drought-tolerant once established)",
    
    # Techniques
    "compost": "Good compost needs 2 parts brown (carbon) to 1 part green (nitrogen) materials. Turn pile every 2-4 weeks.",
    "mulch": "Apply 2-4\" of organic mulch (straw, wood chips) around plants. Keep mulch a few inches from stems.",
    "prune": "Use clean, sharp tools. Prune at 45° angle just above outward-facing buds. Remove dead/diseased wood first.",
    "raised bed": "Raised beds should be 12-18\" deep. Use mixture of topsoil, compost, and organic matter.",
    "container": "Use pots with drainage holes. Water when top 1\" of soil is dry. Fertilize regularly as nutrients leach out."
}

# ✅ Break context into chunks
context_chunks = [chunk.strip() for chunk in gardening_context.split("\n\n") if chunk.strip()]

def contains_gardening_keywords(text):
    words = set(re.findall(r"\b\w+\b", text.lower()))
    return any(kw.lower() in words for kw in gardening_keywords)

# ✅ TF-IDF function
def find_relevant_context(question, top_n=3):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2)).fit(context_chunks + [question])
    vectors = vectorizer.transform(context_chunks + [question])
    sims = cosine_similarity(vectors[-1], vectors[:-1])
    top_indices = sims[0].argsort()[-top_n:][::-1]
    return " ".join([context_chunks[i] for i in top_indices])

# ✅ Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_question = data.get("question", "").lower().strip()

    if not user_question:
        return jsonify({"response": "Please enter a gardening question!"})

    # ✅ Use updated keyword detection
    if not contains_gardening_keywords(user_question):
        return jsonify({
            "response": (
                "🌿 I specialize in gardening advice. Try asking about:\n"
                "- How to grow tomatoes\n"
                "- Rose care tips\n"
                "- When to plant strawberries\n"
                "- Best herbs for beginners"
            )
        })

    # ✅ Check for exact matches in fallback answers first
    for keyword, answer in fallback_answers.items():
        if keyword in user_question:
            print(f"Using fallback for keyword: {keyword}")
            return jsonify({"response": answer})

    # ✅ Find relevant context
    relevant_context = find_relevant_context(user_question)

    # ✅ Transformer model
    try:
        result = qa_pipeline(
            question=user_question,
            context=relevant_context,
            max_answer_len=150,
            handle_impossible_answer=True
        )

        if result["score"] > 0.3 and result["answer"].strip():
            print("Model gave a confident answer.")
            return jsonify({"response": result["answer"]})

    except Exception as e:
        print(f"Model error: {e}")

    # ✅ Final fallback (if gardening-related but no answer)
    if contains_gardening_keywords(user_question):
        print("Gardening-related fallback: no confident answer from model.")
        advice = [
            "Most plants need 6+ hours of sunlight daily",
            "Water when the top inch of soil is dry",
            "Improve soil with compost before planting",
            "Rotate crops annually to prevent disease"
        ]
        return jsonify({
            "response": "Here's general gardening advice that might help:\n- " + "\n- ".join(advice)
        })
    else:
        print("Not gardening-related. Offering suggested questions.")
        return jsonify({
            "response": (
                "🌿 I specialize in gardening advice. Try asking about:\n"
                "- How to grow tomatoes\n"
                "- Rose care tips\n"
                "- When to plant strawberries\n"
                "- Best herbs for beginners"
            )
        })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)