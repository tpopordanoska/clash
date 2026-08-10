import os
from pathlib import Path

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_DIR = Path("path/to/coco/dataset") # Update this path to your COCO dataset directory
CONTRADICTION_TEXT = "Conflicting information – cannot answer."

OBJECT_CATEGORIES_NAMES = ["animals", "transportation", "food", "sports", "household", "other"]
ATTRIBUTE_CATEGORIES_NAMES = ["colors", "numbers", "materials", "physical", "environmental", "other"]

OBJECT_CONDITIONS = {
    "change_is_object": "Yes",
    "change_is_synonym": "No",
    "change_is_ambiguous": "No",
    "change_is_relevant": "Yes",
}

ATTRIBUTE_CONDITIONS = {"change_is_attribute": "Yes", "change_is_objective": "Yes"}

ANSWERS_CONDITIONS = {
    "ans_is_synonym": "No",
    "ans_is_ambiguous": "No",
    "ans_is_relevant": "Yes",
    "ans_is_visible": "Yes",
}

QUESTION_CONDITIONS = {
    "question_is_clear": "Yes",
    "question_is_focused": "Yes",
    "question_is_answerable": "Yes",
}

MODELS_MAPPER = {
    "models/gemini-2.0-flash-lite": "Gemini 2.0 Flash Lite",
    "gemini-2.5-flash-lite-preview-06-17": "Gemini 2.5 Flash Lite",
    "gemini-1.5-flash-8b": "Gemini 1.5 Flash 8B",
    "gemini-1.5-pro": "Gemini 1.5 Pro",
    "gemini-2.0-flash-exp": "Gemini 2.0 Flash",
    "gemini-1.5-flash-latest": "Gemini 1.5 Flash",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gemini-2.5-flash-lite": "Gemini 2.5 Flash Lite",
    "gpt-5": "GPT 5",
    "gpt-4.1-mini": "GPT 4.1 Mini",
    ### open-source ###
    "instruct_blip_t5xxl": "InstructBlip-T5xxl",
    "instruct_blip_vicuna7b": "InstructBlip-7b",
    "blip2_t5xxl": "Blip2-T5xxl",
    "internvl_chatv1.5": "InternVL1.5",
    "Phi3_vision_128k_instruct": "Phi3-vision-128k-instruct",
    ### minigpt4 ###
    "minigpt4": "MiniGPT4-7b",
    "minigpt4_lrv": "MiniGPT4-7b-LRV",
    "minigpt4_ft_mc": "MiniGPT4-7b-ft",
    "minigpt4_ft_open_ended": "MiniGPT4-7b-ft", 
    ### mplug_owl ###
    "mplug_owl1": "mPLUG-Owl-1",
    "mplug_owl2": "mPLUG-Owl-2",
    "mplug_owl1_lrv": "mPLUG-Owl-LRV",
    "mplug_owl1_lora_open_ended_single": "mPLUG-Owl-ft-single-round", 
    "mplug_owl1_lora_mc_single": "mPLUG-Owl-ft-single-round",
    "mplug_owl1_lora_open_ended_single_wo_contr": "mPLUG-Owl-ft-single-round-wo-contr",
    "mplug_owl1_lora_mc_single_wo_contr": "mPLUG-Owl-ft-single-round-wo-contr",
    ### qwen ###
    "Qwenvl_7b_instruct": "QwenVL-7b",
    "Qwen2vl_7b_instruct": "Qwen2VL-instruct-7b",
    "Qwen2_5_vl_7b_instruct": "Qwen2.5VL-instruct-7b",
    "Qwen2_5_vl_32b_instruct": "Qwen2.5VL-instruct-32b",
    "Qwen3vl_8b_instruct": "Qwen3VL-instruct-8b",
    "Qwen3vl_30b_instruct": "Qwen3VL-instruct-30b",
    ### llava ###
    "llava-onevision-7b": "LLaVa-onevision-7b",
    "llava-v1.5-7b": "LLaVa1.5-7b",
    "llava-v1.5-7b_wo_contr": "LLaVa1.5-7b-wo-contr",
    "llava-v1.5-7b_lora_mc_single": "LLaVa1.5-7b-ft-single-round",
    "llava-v1.5-7b_lora_mc_multi": "LLaVa1.5-7b-ft-multi-round",
    "llava-v1.5-7b_lora_open_ended_single": "LLaVa1.5-7b-ft-single-round",
    "llava-v1.5-7b_lora_open_ended_multi": "LLaVa1.5-7b-ft-multi-round",
    "llava-v1.5-7b_lora_mc_30k": "LLaVa1.5-7b-ft-30k",
    "llava-v1.5-7b_lora_open_ended_30k": "LLaVa1.5-7b-ft-30k",
    "llava-v1.5-7b_lora_mc_single_wo_contr": "LLaVa1.5-7b-ft-wo-contr-single-round",
    "llava-v1.5-7b_lora_mc_multi_wo_contr": "LLaVa1.5-7b-ft-wo-contr-multi-round",
    "llava-v1.5-7b_lora_open_ended_single_wo_contr": "LLaVa1.5-7b-ft-wo-contr-single-round",
    "llava-v1.5-7b_lora_open_ended_multi_wo_contr": "LLaVa1.5-7b-ft-wo-contr-multi-round",
    "llava-v1.5-7b_lora_open_ended_single_30k": "LLaVa1.5-7b-ft-single-round-30k",

}

OBJECT_CATEGORIES = {
    "animals": [
        # Domestic/farm animals
        "dog", "cat", "sheep", "cow", "cows", "horse", "horses", "pig", "goat", "goats",
        "cattle", "donkeys", "chicken", "bull",

        # Wild animals
        "elephant", "zebra", "zebras", "giraffe", "giraffes", "bear", "rhino", "rhinoceros",
        "rhinoceroses", "rhinos", "bird", "birds", "monkey", "camel", "antelope", "deer",
        "bison", "wildebeest", "hippopotamus", "hippos", "polar bear", "brown bear",
        "parrot", "owl", "crow", "pigeon", "butterfly", "octopus", "shark", "fish", "worm",

        # Multiple/general
        "elephants", "dogs", "cats", "ducks", "animals", "kitten", "puppy",
    ],

    "transportation": [
        # Land vehicles
        "skateboard", "skate board", "bicycle", "car", "bus", "scooter", "motorcycle",
        "train", "truck", "cars", "tractor", "automobile", "motorcycles",
        "bicycles", "scooters", "fire truck", "police car", "snowmobile", "tow truck",
        "pickup truck", "train car", "tour buses", "bullet train",

        # Air vehicles
        "plane", "airplane", "air plane", "helicopter", "fighter jet", "commercial plane",
        "fighter jets", "commercial jets",

        # Water vehicles
        "boat", "boats", "surf board", "surfboard", "surf boards", "kayak", "wakeboard",

        "bike", "bikes", "commercial jet", "engine", "trunk", "trucks", "road", "tracks",
        "track", "windsail", "parking meter",
    ],

    "food": [
        # Fruits
        "apples", "bananas", "banana", "fruits", "fruit", "oranges", "apple", "pear",
        "strawberries", "blueberries", "strawberry", "banana peel", "apple core",

        # Vegetables
        "vegetables", "carrots", "potatoes", "broccoli", "olives", "tomatoes", "carrot",
        "cauliflower", "onion rings", "mushrooms", "peppers", "peas", "spinach",

        # Prepared food
        "pizza", "burger", "burgers", "cake", "hot dog", "hot dogs", "pastry", "sandwich",
        "donuts", "cookies", "food", "noodles", "hamburgers", "hotdogs", "cheese", "sauce",
        "sandwiches", "quiche", "meat", "mead", "beef", "eggs", "pasta", "french fries",
        "bread", "hamburger", "rice", "cheeses", "meats", "fries", "rice cake", "cookie",
        "pickle", "piece of cake", "slice of pizza", "sausage",

        # Drinks
        "wine", "beer", "coffee", "wine bottle", "beer bottle", "coffees", "drinks", "milk",

        # Food descriptors/toppings
        "toppings", "sauces", "greens", "pepperoni",

        "bar b que", "blender", "bottle", "bottles", "bowls", "chicken", "dishes",
        "fork", "knife", "olive", "oysters", "pears", "pie", "spoon", "water", "snails"
    ],

    "sports": [
        # Sports equipment
        "ball", "frisbee", "snowboard", "snow board", "skis", "ski", "snowboards",
        "baseball bat", "baseball", "tennis racket", "tennis racquet", "basketball",
        "kite", "kites", "tennis ball", "football", "glove", "bat", "tennis", "surfboard",
        "surf board", "surf boards", "soccer", "soccer ball", "soccer balll", "golf club",
        "golf", "golf ball", "racket", "racquet", "shuttlecock", "a snowboard",
        "skateboard", "skate board", "bicycle",

        # Sports people/participants
        "snowboarder", "skier", "skiier", "batter", "catcher", "snowboarders", "skiers",
        "skateboarder", "cyclist", "surfer", "kayaker", "tennis players", "basketball players",
        "tennis player", "basketball player", "baseball player", "football player",
        "baseball players", "football players", "umpire", "skateboarders", "cyclists",

        # Sports venues/areas
        "tennis court", "basketball court", "skate park", "ski lift"

        "base ball", "cricket", "pitch", "pitcher", "ski lift", "skis", "surfer", "track", "wakeboard",
    ],

    "household": [
        # Furniture
        "chair", "table", "bed", "sofa", "couch", "bench", "coffee table", "dining table",
        "computer desk", "nightstand", "shelf", "counter",

        # Room/space identifiers
        "kitchen", "bathroom", "bedroom", "living room", "dining room",

        # Bathroom items
        "toilet", "sink", "bathtub", "shower", "toothbrush", "tooth brush", "toilet tissue",
        "soap", "mirror", "toilet bowl", "shower curtain", "towel rack", "handicap bar",

        # Kitchen items
        "fork", "bowl", "bowls", "spoon", "plate", "plates", "knife", "knif",
        "cup", "trays", "pots",

        # Storage/containers
        "bag", "trash can", "laundry basket", "baskets", "mason jar", "coffee cup", "bottle",
        "bottles", "vases",

        # Technology/Electronics (moved from previous category)
        "phone", "cell phone", "laptop", "laptops", "keyboard", "mouse", "tablet", "tablets",
        "television", "camera", "phones", "cellphones", "wii", "wii console", "playstation",
        "playstation console", "xbox", "remote", "game remote", "controller", "refrigerator",
        "oven", "stove", "microwave", "dishwasher", "washer", "dryer", "clothes washer",
        "clothes dryer", "blender", "ice machine", "coffee machine", "printer", "monitors",
        "screen", "clock", "bell", "ipod", "microphones", "speakers", "equipment",

        # Decor/furnishing
        "lamp", "paintings", "painting", "picture frame", "pillow", "carpet", "rug",
        "sculptures", "sculpture", "statues", "statutes", "crosses", "flags", "flag",

        # Household items
        "furniture", "window", "door", "towels", "dishes", "appliances", "comb", "rope",
        "chain", "scarf", "mask", "ties", "scarves", "backpack", "coat", "shirt", "belt",
        "hairbrush", "aluminum foil", "plastic wrap", "stand", "cart", "books",
        "book", "glasses", "handle", "backrest", "toys", "toy", "doll",
        "accessories", "clothing", "swimsuit", "dress", "skirts", "pants", "roses",
        "tulips", "flowers", "plants", "leaves", "branches", "umbrella", "umbrellas", "hat",

        "changing table", "fire place", "fireplace", "pacifier", "refrigerator magnet", "urinal",
    ]
}


ATTRIBUTE_CATEGORIES = {
    "colors": [
        # Single colors
        "blue", "white", "red", "black", "brown", "green", "yellow", "orange",
        "pink", "grey", "gray", "purple", "silver", "tan", "beige", "cream", "gold",

        # Color combinations
        "black and white", "blue and white", "black and yellow", "green and yellow",
        "brown and white", "black and red", "black and gray", "white and gray",

        # Color descriptors
        "light blue", "dark red", "mint green", "colorful", "rainbow colored",
        "monochrome", "dark", "light", "color", "colored", "different colors",
        "colors", "browns", "whites", "rosy",
        
        "colorfully", "red-haired", "blonde-haired", "ginger", "creamy"
    ],

    "numbers": [
        # Basic numbers
        "one", "two", "three", "four", "five", "six", "seven",

        # Written numbers
        "2", "3", "25", "50",

        # Ordinals
        "first", "second", "third",

        # Quantities
        "a", "another", "solo", "whole",

        # Prices
        "11.98", "10.99"
    ],

    "materials": [
        # Materials
        "wooden", "wood", "metal", "plastic", "glass", "ceramic", "concrete",
        "stainless steel", "tile", "brick", "cement", "marble", "leather",
        "fabric", "steel", "granite", "stone", "plywood", "paper",

        # Surface qualities and textures
        "striped", "polka dot", "polka-dotted", "polka dotted", "tiled", "plain",
        "painted", "polished", "scratched", "printed", "stripped",
    ],

    "physical": [
        # Shapes
        "square", "round", "circular", "oval", "rectangular", "triangular",

        # Physical descriptors
        "thick", "thin", "stuffed", "sliced", "ripe", "unripe", "wet", "dry",
        "clean", "muddy", "squares", "wedges", "opaque", "clear", "edge", "back",
        
        "duck shaped", "fish shaped", "horned", "antlered",
    ],

    "environmental": [
        # Weather
        "sunny", "snowy", "cloudy", "overcast", "stormy", "wet",

        # Landscape/terrain
        "grassy", "grass covered", "snow covered", "rocky", "sandy", "lush", "dry",
        "desert", "tropical", "remote", "green", "fenced",

        # Water depth
        "knee deep", "ankle deep",

        # Light conditions
        "dim", "bright",
    ]
}