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
    "Gardening involves the cultivation of plants, including flowers, vegetables, fruits, and herbs, in various environments such as backyards, balconies, and greenhouses."  
"Healthy soil is the foundation of a thriving garden, requiring a balanced mix of sand, silt, and clay, along with organic matter like compost and mulch."  
"Organic gardening avoids synthetic chemicals, relying on natural fertilizers, compost, and companion planting to enhance soil health and pest resistance."  
"Raised garden beds improve drainage, reduce soil compaction, and allow better control over soil quality, making them ideal for urban gardening."  
"Drip irrigation systems conserve water by delivering it directly to plant roots, reducing evaporation and promoting efficient hydration."  
"Compost enriches soil with essential nutrients, improving plant health and promoting microbial activity necessary for root development."  
"Mulching helps retain soil moisture, suppress weeds, regulate soil temperature, and prevent erosion, benefiting plant growth."  
"Pollinators like bees, butterflies, and hummingbirds play a vital role in plant reproduction, ensuring fruit and vegetable production."  
"Vegetables such as tomatoes, peppers, and cucumbers thrive in full sunlight, requiring at least six to eight hours of direct exposure daily."  
"Hydroponic gardening allows plants to grow without soil, using nutrient-rich water solutions to sustain root systems efficiently."  
"Companion planting, such as growing basil near tomatoes, enhances growth, deters pests, and improves overall plant health."  
"Crop rotation prevents soil nutrient depletion and reduces the risk of plant diseases and pest infestations in vegetable gardens."  
"Indoor plants, like snake plants and peace lilies, improve air quality by absorbing toxins and releasing oxygen into the environment."  
"Perennials return year after year, making them a low-maintenance choice for gardeners looking for sustainable plant options."  
"Seasonal gardening ensures that crops are grown at the right time of the year, maximizing yields and minimizing plant stress."  
"Pruning fruit trees during the dormant season encourages healthier growth, better fruit production, and improved tree structure."  
"Microgreens are nutrient-dense young vegetable greens that can be harvested within weeks, making them ideal for small-space gardening."  
"Organic fertilizers such as fish emulsion, bone meal, and worm castings improve soil fertility and enhance plant health naturally."  
"Self-watering planters help maintain consistent moisture levels, reducing the risk of overwatering or underwatering plants."  
"Greenhouse gardening extends the growing season by providing a controlled environment that protects plants from extreme weather."  
"Native plants require less maintenance, adapt well to local climates, and provide food and habitat for local wildlife and pollinators."  
"Succulents and cacti thrive in dry conditions, requiring minimal watering and well-draining soil to prevent root rot."  
"Vertical gardening techniques, such as trellises and hanging planters, maximize space and allow for higher crop yields in limited areas."  
"Organic pest control methods, like neem oil and introducing beneficial insects, help manage garden pests without harming the environment."  
"Permaculture gardening focuses on sustainable and regenerative agricultural practices to create self-sufficient ecosystems."  
"Edible landscaping integrates fruits, vegetables, and herbs into ornamental garden designs, blending beauty with functionality."  
"Rainwater harvesting systems collect and store rainwater for irrigation, reducing reliance on municipal water supplies."  
"Frost protection techniques, such as using row covers and cold frames, help protect tender plants from freezing temperatures."  
"Garden biodiversity improves soil health and ecosystem stability by incorporating a variety of plant species and encouraging beneficial organisms."  
"Tomatoes benefit from staking or caging to support their growth and prevent fruits from touching the soil, reducing disease risks."  
"Herb gardens thrive in both indoor and outdoor settings, providing fresh culinary ingredients and natural remedies for health and wellness."  
"Butterfly gardens attract pollinators by incorporating nectar-rich flowers such as lavender, zinnias, and milkweed."  
"Leaf mold, created from decomposed leaves, improves soil structure and water retention, making it a valuable natural amendment."  
"Worm composting, or vermicomposting, uses earthworms to break down organic waste into nutrient-rich worm castings for garden use."  
"Seed starting indoors allows gardeners to get a head start on the growing season before transplanting seedlings outdoors."  
"Landscaping with drought-tolerant plants, such as lavender and yucca, conserves water and reduces the need for frequent irrigation."  
"Growing herbs like mint, rosemary, and thyme in containers provides easy access to fresh ingredients for cooking and teas."  
"Eco-friendly gardening reduces environmental impact by using natural fertilizers, composting, and practicing water conservation."  
"Aeroponics is a soil-free gardening method where plants receive nutrients through misted water solutions, promoting rapid growth."  
"Growing fruit trees in home gardens provides a sustainable source of fresh produce and enhances the local ecosystem."  
"Green manure crops, such as clover and rye, are grown and tilled back into the soil to improve fertility and organic content."  
"Container gardening allows urban gardeners to grow plants in pots and planters, making efficient use of small spaces."  
"Sunflowers are excellent companion plants that attract beneficial insects and provide shade for smaller crops."  
"Garden therapy promotes mental well-being by engaging in planting, weeding, and caring for plants as a form of relaxation."  
"Hardening off seedlings before transplanting helps them adjust to outdoor conditions, reducing transplant shock."  
"Using natural insect repellents like garlic spray and diatomaceous earth keeps pests at bay without harming beneficial insects."  
"Wildflowers provide habitat for bees and butterflies while adding vibrant colors to gardens and natural landscapes."  
"Biodiversity in a garden supports healthy ecosystems by attracting a variety of beneficial insects, birds, and microorganisms."  
"Permaculture design principles focus on sustainable agriculture, reducing waste, and creating self-sufficient food systems."  
"Growing medicinal plants such as chamomile, aloe vera, and echinacea provides natural remedies for common ailments."  
"Sun-loving plants like lavender, marigolds, and petunias thrive in direct sunlight and require minimal shade."  
"Watering deeply but less frequently encourages deep root growth and increases plant resilience during dry periods."  
"Using mulch made from organic materials like straw, bark, or leaves helps retain soil moisture and suppress weeds."  
"Rock gardens are ideal for drought-prone areas, featuring succulents, cacti, and other low-water plants."  
"Integrated pest management combines biological, mechanical, and organic controls to maintain a healthy, pest-free garden."  
"Rooftop gardens help reduce urban heat and provide fresh produce in city environments with limited green spaces."  
"Using slow-release fertilizers provides plants with a steady supply of nutrients over an extended period."  
"Crop diversity in gardens helps prevent disease spread and creates a more resilient growing environment."  
"Garden edging with bricks, stones, or wooden borders defines planting areas and enhances garden aesthetics."  
"Shade-loving plants like ferns, hostas, and begonias thrive in areas with limited direct sunlight."  
"Growing nutrient-rich crops like kale, spinach, and Swiss chard ensures a healthy and vitamin-packed diet."  
"Community gardens bring people together to share gardening knowledge, cultivate crops, and support local food systems."  
"Using cover crops like alfalfa and mustard greens improves soil structure and reduces erosion during the off-season."  
"Garden composting reduces household waste by turning kitchen scraps and yard debris into nutrient-rich soil amendments."  
"Drip irrigation conserves water and prevents fungal diseases by keeping foliage dry and delivering moisture directly to roots."  
"Building trellises for climbing plants like peas, beans, and cucumbers maximizes vertical space in small gardens."  
"Microclimates within a garden, such as shaded areas or sunny spots, influence plant growth and should be considered in garden planning."  
"Using recycled materials for garden structures, like repurposed wood and old containers, promotes sustainability."  
"Designing a sensory garden with fragrant plants, textured leaves, and colorful flowers creates an engaging experience for all senses."  
"Growing a diverse range of crops in a home garden ensures year-round harvests and reduces reliance on store-bought produce."  
"Native wildflowers and grasses provide habitat for wildlife while requiring minimal maintenance and water."  
"Utilizing companion planting techniques, such as pairing onions with carrots, deters pests and enhances growth."  
"Selecting disease-resistant plant varieties helps prevent common garden problems and reduces the need for chemical treatments."
"Soil pH affects nutrient availability, with most plants thriving in a range of 6.0 to 7.0."

"Aerated compost tea enhances soil microbial life and provides essential nutrients."

"Shade gardens use plants that thrive in limited sunlight, such as ferns and hostas."

"Hügelkultur is a gardening method that involves creating raised beds with decaying wood."

"Espalier is a technique of training trees to grow flat against a structure."

"Square foot gardening maximizes small spaces by dividing raised beds into sections."

"No-dig gardening reduces soil disturbance, preserving beneficial microbes and fungi."

"Deep watering encourages plants to develop strong and deep root systems."

"Lawn alternatives like clover and ground covers reduce water usage and maintenance."

"Xeriscaping is a landscaping method that conserves water using drought-tolerant plants."

"Native plants attract pollinators and require less maintenance than non-native species."

"Sunflowers can be used as natural trellises for climbing plants like beans."

"Clover lawns improve soil health by fixing nitrogen and reducing erosion."

"Trellising vining plants increases air circulation and prevents fungal diseases."

"Drip irrigation prevents water waste by delivering moisture directly to roots."

"Herb spirals provide microclimates, allowing multiple herbs to thrive in one space."

"Vertical gardens are ideal for urban gardening, maximizing limited space."

"Rain gardens help manage stormwater runoff and filter pollutants from soil."

"Compost bins reduce waste and create nutrient-rich organic matter for plants."

"Wicking beds are self-watering garden beds that use a reservoir system."

"Winter gardening involves using cold frames, greenhouses, or hardy plants."

"Biodynamic gardening follows lunar cycles and organic farming techniques."

"Grafted fruit trees combine desirable rootstocks with high-yielding varieties."

"Hydroponic systems can be soil-free and allow faster plant growth."

"Straw bale gardening is an alternative to soil-based gardening with minimal weeds."

"Urban farming brings food production to city environments using rooftops and balconies."

"Gutter gardens make use of vertical space by growing plants in repurposed gutters."

"Edible flowers such as nasturtiums and violets can be used in culinary dishes."

"Chickens in gardens provide natural pest control and organic fertilizer."

"Soil amendment with biochar improves carbon retention and soil fertility."

"Perlite in potting mixes improves drainage and aeration for healthy roots."

"Seed stratification mimics winter conditions to encourage germination in certain seeds."

"Aromatic herbs like lavender and rosemary naturally repel mosquitoes."

"Tomatoes and basil grow well together due to mutual pest-repelling properties."

"Moon gardens feature white or pale-colored flowers that glow in moonlight."

"Planting in layers, such as canopy trees with understory shrubs, maximizes space."

"Windbreaks made of trees or shrubs protect plants from strong winds."

"Raised beds improve drainage and warm up soil faster in the spring."

"Self-seeding plants like calendula and dill come back every season naturally."

"Alley cropping integrates trees and crops to improve biodiversity and soil health."

"Diatomaceous earth is a natural insecticide that controls soft-bodied pests."

"Intercropping different plants reduces pest outbreaks and improves soil nutrients."

"Butterflies prefer plants like milkweed, verbena, and butterfly bush for nectar."

"Hedges provide privacy, wind protection, and habitat for birds and insects."

"Microgreens grow quickly and provide high concentrations of vitamins and minerals."

"Germination heat mats speed up seed sprouting by providing consistent warmth."

"Companion planting onions and carrots helps deter pests like aphids and carrot flies."

"Bird feeders attract birds that help control insect populations in the garden."

"Wood chips are an excellent mulch that retains moisture and improves soil structure."

"Frost blankets protect sensitive plants from sudden temperature drops."

"Using fish emulsion as a liquid fertilizer boosts plant health naturally."

"Grafting citrus trees allows multiple fruit varieties on a single tree."

"Solarization uses plastic sheets to kill weeds and pests through heat exposure."

"Aromatic herbs like thyme, sage, and oregano deter deer and rabbits."

"Phosphorus is essential for root development and flower production in plants."

"Crop spacing ensures proper airflow, reducing fungal diseases in gardens."

"Floating row covers protect crops from pests while allowing sunlight and rain."

"Pine needles make an acidic mulch, ideal for blueberries and azaleas."

"Self-fertile fruit trees can produce fruit without needing a pollination partner."

"Planting drought-resistant varieties reduces the need for frequent watering."

"Beneficial nematodes help control soil-borne pests like grubs and larvae."

"Citrus peels can be used as a natural pest deterrent in gardens."

"Garden ponds attract frogs and dragonflies, which help control pests naturally."

"Planting deep-rooted cover crops like alfalfa improves soil aeration."

"Using crushed eggshells around plants deters snails and slugs."

"French intensive gardening maximizes yield in small garden spaces."

"Using coffee grounds in compost adds nitrogen and repels certain pests."

"Soaking seeds before planting improves germination rates for certain crops."

"Drying herbs like mint and basil extends their shelf life for later use."

"Creating wildflower strips promotes biodiversity and attracts pollinators."

"Using old newspapers as mulch suppresses weeds and retains soil moisture."

"Greenhouses extend the growing season and protect delicate plants from frost."

"Encouraging beneficial insects like ladybugs helps reduce aphid populations."

"Applying neem oil helps prevent fungal infections and insect infestations."

"Air layering is a propagation technique used for woody plants like figs."

"Tilling soil disrupts weeds but can also harm beneficial organisms."

"Planting hedge rows around crops creates wind barriers and natural habitats."

"Aeroponic gardening is an advanced soilless method using misted nutrients."

"Edging gardens with bricks or stones prevents soil erosion and weed invasion."

"Marigolds are known to deter root-knot nematodes in vegetable gardens."

"Fencing protects gardens from deer, rabbits, and other wildlife damage."

"Winter rye as a cover crop improves soil organic matter."

"Planting crops at the right time reduces the risk of disease and pest damage."

"Using bamboo stakes supports climbing vegetables like beans and peas."

"Soaking wood ash in water creates potassium-rich fertilizer for plants."

"Peppermint oil deters rodents and insects when used as a garden spray."

"Using leftover cooking water cools and adds nutrients to garden soil."

"A garden journal helps track planting schedules and seasonal observations."

"Solar-powered garden lights create an eco-friendly nighttime landscape."

"Rain chains guide rainwater into barrels for later irrigation use."

"Leaf mold increases soil water retention and enhances plant health."

"Indoor vertical farms use LED lighting for year-round food production."

"Homemade insecticidal soap protects plants from soft-bodied pests."

"Espaliered fruit trees make use of small garden spaces effectively."

"Strategically placed rocks in gardens provide shelter for beneficial insects."

"Planting garlic near roses helps deter aphids naturally."

"Growing berries like raspberries and blackberries requires trellising."

"Shade cloth protects delicate plants from excessive sun exposure."

"Using straw mulch prevents soil-borne diseases from splashing onto leaves."

"Terracing prevents soil erosion on sloped gardens and retains moisture."
)

# List of allowed gardening-related keywords
gardening_keywords = ["garden", "plant", "soil", "water", "sunlight", "compost", "fertilizer", 
                      "pruning", "roses", "vegetables", "fruit", "mulch", "pollination", "seeds",
                      "hydroponic", "irrigation", "weeds", "drainage", "pests", "indoor plants","rose","carrots",
                      "garden", "gardening", "plant", "plants", "soil", "compost", "fertilizer", "pruning", "watering", "irrigation", "sunlight", "mulch", "pollination", "seeds", "hydroponics", "aquaponics", "vertical gardening", "raised beds", "drainage", "pests", "organic gardening", "indoor plants", "herbs", "vegetables", "fruit trees", "flowers", "crop rotation", "pollinators", "greenhouse", "vermicomposting", "aeroponics", "rooftop gardening", "organic pest control", "companion planting", "trellis", "compost tea", "potting soil", "nitrogen", "phosphorus", "potassium", "loamy soil", "pH balance", "cover crops", "drought-resistant plants", "hanging gardens", "worm composting", "edible flowers", "leaf mold", "sphagnum moss", "self-watering planters", "bonsai", "succulent", "cactus", "air plants", "orchids", "bird feeder", "garden edging", "greenhouse gardening", "biodegradable pots", "garden tools", "raised rows", "misting", "grow lights", "microgreens", "espaliers", "beneficial insects", "seed starting", "propagation", "container gardening", "biochar", "frost protection", "garden trellises", "tomato hornworm", "beneficial nematodes", "vertical farming", "LED grow lights", "self-sufficiency", "irrigation system", "biodiversity", "wildflowers", "soil aeration", "food forest", "agroforestry", "perennials", "annuals", "permaculture", "rewilding", "native plants", "flower pressing", "garden therapy", "drip irrigation", "landscape design", "companion flowers", "natural insecticides", "eco-friendly gardening", "garden composting", "green manure", "fruit thinning", "hardening off", "sustainable gardening", "moon gardening", "worm castings", "raised garden beds", "desert plants", "rock garden", "butterfly garden", "garden mulch", "pruning shears", "seed catalog", "soil amendments", "garden biodiversity", "rooftop farming", "water conservation", "garden planning", "organic farming", "gardening gloves", "trellis netting", "espalier pruning", "microclimate gardening","soil", "pH", "nutrients", "compost", "shade garden", "ferns", "hostas", "Hügelkultur", "raised beds", "decaying wood", "Espalier", "tree training", "square foot gardening", 
"small space gardening", "no-dig gardening", "aeration", "microbes", "deep watering", 
"root systems", "lawn alternatives", "clover", "xeriscaping", "drought-tolerant", 
"native plants", "pollinators", "sunflowers", "trellises", "vining plants", "drip irrigation", 
"herb spirals", "vertical gardening", "urban gardening", "rain gardens", "stormwater runoff", 
"compost bins", "organic matter", "wicking beds", "self-watering", "winter gardening", 
"cold frames", "greenhouses", "biodynamic gardening", "lunar cycles", "grafted trees", 
"hydroponics", "soilless gardening", "straw bale gardening", "weed control", "urban farming", 
"rooftop gardens", "gutter gardens", "vertical space", "edible flowers", "culinary gardening", 
"chickens", "pest control", "biochar", "perlite", "potting mix", "seed stratification", 
"germination", "aromatic herbs", "mosquito repellent plants", "companion planting", 
"moon gardens", "night-blooming plants", "layered planting", "windbreaks", "privacy plants", 
"raised beds", "self-seeding plants", "cover crops", "diatomaceous earth", "intercropping", 
"butterflies", "hedges", "wildlife habitat", "microgreens", "nutrient-dense plants", 
"germination heat mats", "carrot fly prevention", "bird feeders", "pest control birds", 
"wood chips", "mulching", "frost blankets", "temperature protection", "fish emulsion", 
"liquid fertilizer", "grafting citrus", "solarization", "plastic mulch", "deer-resistant plants", 
"phosphorus", "flowering plants", "crop spacing", "fungal disease prevention", 
"floating row covers", "pest barriers", "acidic mulch", "pine needles", "blueberry soil", 
"self-fertile trees", "pollination", "drought-resistant", "low-water gardening", 
"beneficial nematodes", "pest control soil", "citrus peels", "natural pest deterrent", 
"garden ponds", "frog habitat", "dragonflies", "deep-rooted plants", "alfalfa", 
"eggshell mulch", "French intensive gardening", "high-yield gardening", "coffee grounds", 
"organic fertilizer", "seed soaking", "germination boost", "drying herbs", "herbal preservation", 
"wildflower strips", "pollinator gardens", "newspaper mulch", "weed suppression", 
"greenhouses", "season extension", "beneficial insects", "ladybugs", "neem oil", 
"organic pesticides", "air layering", "plant propagation", "no-till gardening", 
"reduced soil disruption", "hedgerows", "wind barriers", "aeroponics", "mist irrigation", 
"garden edging", "border plants", "marigolds", "nematode control", "fencing", "deer fencing", 
"winter rye", "soil enrichment", "seasonal planting", "bamboo stakes", "plant support", 
"wood ash fertilizer", "natural potassium", "peppermint oil", "rodent repellent", 
"cooking water fertilizer", "garden journal", "plant tracking", "solar garden lights", 
"eco-friendly lighting", "rain chains", "rainwater harvesting", "leaf mold", 
"moisture retention", "indoor vertical farms", "LED plant growth", "insecticidal soap", 
"natural pest control", "espalier fruit trees", "space-saving gardening", "garden rocks", 
"insect shelter", "garlic companion planting", "aphid repellent", "berry trellising", 
"fruit supports", "shade cloth", "sun protection", "straw mulch", "disease prevention", 
"terracing", "soil erosion prevention", "slope gardening"]

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
