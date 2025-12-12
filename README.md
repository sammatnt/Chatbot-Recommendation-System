# Recommendation Engine & Visual Search System

This project is a multimodal fashion recommendation engine that combines:

- **LightFM collaborative filtering** for returning users  
- **Rule-based filtering** for new users  
- **CLIP-based image similarity search** for visual recommendations  
- **A conversational chatbot interface** powered by OpenAI  
- **A Streamlit UI** for interactive product exploration  

The system allows users to:
1. Receive personalized recommendations (if they are returning customers).
2. Discover products using filters (for new users).
3. Upload an image and get visually similar product matches.
4. Interact through a chatbot that guides the recommendation flow.

---

## Features

### **1. Collaborative Filtering (Returning Users)**
- Uses **LightFM** with `logistic` loss.
- Learns user and product embeddings from past purchase interactions.
- Generates ranked product recommendations using: score = dot(user_vector, item_vector)
- No explicit product attributes needed — purely behavioral.

### **2. Cold-Start Filtering (New Users)**
- For users without history, recommendations come from:
- Product Type  
- Color  
- Prints  
- Sleeves  
- Operates via standard DataFrame filtering.

### **3. Image-Based Similarity Search (CLIP)**
- Uses **CLIP ViT-B/32** from HuggingFace.
- Extracts embeddings for:
- User-uploaded images  
- Combined product metadata (Color, Neck, Prints, Sleeves)
- Computes cosine similarity to return visually similar items.
- Independent of the LightFM recommender.

### **4. Chatbot Interface**
- Conversational guidance powered by **GPT-4o-mini**.
- Helps users navigate:
- Old-user recommendations
- New-user filtering
- Image similarity search
- Manages multi-step conversational state with `session_state`.

---

## Environment Variables

This application uses environment variables for configuration. You will need to create a `.env` file in the root of the project based on the provided `.env.example` file.

The following environment variables are required:

* `OPENAI_API_KEY`: Your API key for accessing the OpenAI service.
* `FEATURES_DATASET_PATH`: The absolute or relative path to the CSV file containing the features data for image search.
* `FULL_DATASET_PATH`: The absolute or relative path to the main CSV file containing product information.
* `BACKGROUND_IMAGE_PATH`: The absolute or relative path to the background image file for the Streamlit application.

**To set up the environment:**

1.  Make a copy of the `.env.example` file and rename it to `.env`.
2.  Replace the placeholder values in the `.env` file with your actual API key and file paths.

## Running the App
streamlit run chatbottest.py

