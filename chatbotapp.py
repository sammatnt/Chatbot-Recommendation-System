import os
import openai
import streamlit as st
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import base64
import logging
import requests
from bs4 import BeautifulSoup  # For scraping if needed
from recommendation_system import RecommendationSystem  # Import the class
from dotenv import load_dotenv


load_dotenv()
file_path = os.getenv("FULL_DATASET_PATH")
print(f"The value of FULL_DATASET_PATH is: '{file_path}'")
df = pd.read_csv(file_path)


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load CLIP model and processor for image search
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load and clean features dataset for image search (no UI output)
@st.cache_data
def load_features():
    path = os.getenv("FEATURES_DATASET_PATH")

    try:
        df = pd.read_csv(path, encoding='utf-8', on_bad_lines='warn')
        df["Variant ID"] = df["Variant ID"].astype(str).str.strip()  # Strip whitespace
        df = df[["Variant ID", "Color", "Neck_inventory", "Prints_inventory", "Sleeves_inventory"]]
        for col in ["Color", "Neck_inventory", "Sleeves_inventory"]:
            df[col] = df[col].fillna("")
        df["combined_features"] = (
            df["Color"] + " " +
            df["Neck_inventory"].replace("", "round neck") * 15 + " " +
            df["Sleeves_inventory"].replace("", "short sleeve") * 15
        )
        df = df.drop_duplicates(subset=["Variant ID"], keep='first')  # Ensure unique rows
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

features_df = load_features()

if features_df is not None:
    # Extract text features using CLIP for image search
    def extract_text_features(texts):
        inputs = clip_processor(text=list(texts), return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            features = clip_model.get_text_features(**inputs)
        return features.squeeze().numpy()

    # Extract image features using CLIP for image search
    def extract_image_features(image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        inputs = clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = clip_model.get_image_features(**inputs)
        return features.squeeze().numpy()

    # Generate text feature vectors for the dataset
    product_feature_vectors = extract_text_features(features_df["combined_features"])

# Initialize OpenAI API

openai.api_key = os.getenv("OPENAI_API_KEY")

if openai.api_key is None:
    st.error("API key is missing. Please check your .env file.")
    st.stop()

# Initialize conversation history, user input, state, and filters
if 'conversation_history' not in st.session_state:
    st.session_state['conversation_history'] = []
if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ''
if 'user_state' not in st.session_state:
    st.session_state['user_state'] = 'start'
if 'user_filters' not in st.session_state:
    st.session_state['user_filters'] = {}

file_path =  os.getenv("FULL_DATASET_PATH")
df = pd.read_csv(file_path)

# Initialize the recommendation system
recommender = RecommendationSystem(df)

# Function to display recommendations with images and hover effect, with enhanced error handling
def display_recommendations_with_images(recommendations):
    # Inject custom CSS for hover effect
    st.markdown("""
    <style>
        .highlight-on-hover {
            transition: transform 0.2s ease-in-out;
            cursor: pointer;
        }
        .highlight-on-hover:hover {
            transform: scale(1.05);
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }
    </style>
    """, unsafe_allow_html=True)

    # Create two columns layout
    cols = st.columns(2)
    
    # Loop through the recommendations and display them in the columns, limiting to top 3
    for idx, product in enumerate(recommendations[:3]):  # Limit to top 3 unique recommendations
        col = cols[idx % 2]
        
        with col:
            try:
                # Use product_url as the image source (assuming it links to an image or a page with an image)
                image_url = product['product_url']
                logger.info(f"Attempting to load image for Product ID {product['Product ID']} from URL: {image_url}")
                
                # Check if the URL is the provided valid URL or placeholder
                if image_url == 'https://cdn.shopify.com/s/files/1/2503/4628/products/PhoneNumberPage.png?v=1633607510':
                    # Use this URL directly, assuming it's valid as per your confirmation
                    pass
                elif not image_url.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    try:
                        response = requests.get(image_url, timeout=5)
                        soup = BeautifulSoup(response.text, 'html.parser')
                        img = soup.find('img')
                        if img and 'src' in img.attrs:
                            image_url = img['src']
                            if not image_url.startswith('http'):
                                image_url = requests.compat.urljoin(image_url, img['src'])
                        else:
                            logger.warning(f"No image found in webpage for URL: {image_url}")
                            image_url = 'https://cdn.shopify.com/s/files/1/2503/4628/products/PhoneNumberPage.png?v=1633607510'  # Use the valid URL you provided
                    except Exception as e:
                        logger.error(f"Failed to extract image from {image_url}: {str(e)}")
                        image_url = 'https://cdn.shopify.com/s/files/1/2503/4628/products/PhoneNumberPage.png?v=1633607510'  # Fallback to the valid URL

                if image_url:
                    # Verify the URL is accessible before displaying
                    try:
                        response = requests.head(image_url, timeout=5)
                        response.raise_for_status()
                    except requests.RequestException as e:
                        logger.error(f"Image URL {image_url} is not accessible: {str(e)}")
                        image_url = 'https://via.placeholder.com/250'  # Final fallback to placeholder

                    st.markdown(f'<a href="{product["product_url"]}" target="_blank">'
                                f'<img src="{image_url}" class="highlight-on-hover" style="width: 250px; height: auto;" />'
                                f'</a>', unsafe_allow_html=True)
                    st.caption(f"Product ID: {int(product['Product ID'])}")
                else:
                    st.write(f"Unable to load image for Product ID {int(product['Product ID'])}")
                    st.caption(f"Product ID: {int(product['Product ID'])}")
                    logger.warning(f"No valid image URL available for Product ID {product['Product ID']}")
            except Exception as e:
                st.write(f"Error loading image for Product ID {int(product['Product ID'])}: {str(e)}")
                st.caption(f"Product ID: {int(product['Product ID'])}")
                logger.error(f"Error in display_recommendations_with_images for Product ID {product['Product ID']}: {str(e)}")

# New function to prepare unique image recommendations from product URLs and Product IDs for Image Search mode
def prepare_unique_image_recommendations(similar_products, df):
    # Ensure unique recommendations by Variant ID
    unique_variants = similar_products.drop_duplicates(subset=['Variant ID'])
    product_recommendations = []
    for _, product in unique_variants.iterrows():
        variant_id = product['Variant ID']
        product_data = df[df['Variant ID'] == variant_id].iloc[0] if not df[df['Variant ID'] == variant_id].empty else None
        if product_data is not None and 'product_url' in product_data and 'url' in product_data:
            product_recommendations.append({
                'Product ID': variant_id,  # Use Variant ID as Product ID for consistency
                'product_url': product_data['product_url'],
                'url': product_data['url']  # Use url from df
            })
        else:
            logger.warning(f"Missing image-related fields in df for Variant ID: {variant_id}")
            # Fallback with the valid URL you provided
            product_recommendations.append({
                'Product ID': variant_id,
                'product_url': 'https://cdn.shopify.com/s/files/1/2503/4628/products/PhoneNumberPage.png?v=1633607510',
                'url': 'https://cdn.shopify.com/s/files/1/2503/4628/products/PhoneNumberPage.png?v=1633607510'
            })
    return product_recommendations

# Function to get response from OpenAI API
def get_openai_response(messages):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=150,
        temperature=0.9,
    )
    return response.choices[0].message['content'].strip()

def match_closest_option(user_input, options):
    prompt = f"User input: '{user_input}'\nOptions: {options}\nWhich options are the closest matches to the user's input? Please return them as a comma-separated list."
    response = get_openai_response([
        {"role": "system", "content": "You are an assistant that helps match user input to a list of options."},
        {"role": "user", "content": prompt}
    ])
    
    # Normalize response and options for comparison
    def normalize(text):
        return text.lower().replace('/', ' ').replace('-', ' ').replace(',', '').replace('.', '').replace("'", '').strip()
    
    normalized_response = [normalize(match) for match in response.split(",")]
    normalized_options = {normalize(option): option for option in options}
    
    closest_matches = [normalized_options[match] for match in normalized_response if match in normalized_options]

    return closest_matches

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded_string = base64.b64encode(image.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


background_image_path = os.getenv("BACKGROUND_IMAGE_PATH")
if background_image_path:
    try:
        add_bg_from_local(background_image_path)
    except FileNotFoundError:
        logger.error(f"Background image not found at path: {background_image_path}")
else:
    logger.warning("BACKGROUND_IMAGE_PATH environment variable not set in .env file.")

# Title of the app
st.title("Fashom Chatbot")

# Toggle between image upload and chatbot
mode = st.sidebar.radio("Choose Mode", ["Chatbot", "Image Search"])

if mode == "Image Search":
    st.header("Image Search")
    uploaded_file = st.file_uploader("Upload an image (JPG, PNG, JPEG)", type=['jpg', 'png', 'jpeg'])

    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Extract image features with preprocessing
            image_features = extract_image_features(image)

            # Find similar products
            similarities = cosine_similarity([image_features], product_feature_vectors)[0]
            min_similarity = 0.22
            top_indices = np.where(similarities > min_similarity)[0]
            if len(top_indices) > 0:
                top_indices = top_indices[np.argsort(-similarities[top_indices])]  # Sort by similarity
            else:
                top_indices = np.argsort(-similarities)  # Fallback to all sorted by similarity

            # Ensure top_indices are within bounds
            max_index = len(features_df) - 1
            top_indices = [i for i in top_indices if 0 <= i <= max_index]
            if not top_indices:
                st.error("No valid indices found for similar products.")
                st.stop()

            # Get similar products and ensure uniqueness by Variant ID
            similar_products = features_df.iloc[top_indices][['Variant ID', 'Color', 'Neck_inventory', 'Prints_inventory', 'Sleeves_inventory']]
            # Drop duplicates and take top 3 unique Variant IDs
            similar_products = similar_products.drop_duplicates(subset=['Variant ID']).head(3)
            similar_products['Variant ID'] = similar_products['Variant ID'].astype(str)

            # If fewer than 3 unique products, pad with additional unique ones
            if len(similar_products) < 3:
                used_variant_ids = set(similar_products['Variant ID'].values)
                remaining_indices = [i for i in top_indices if features_df.iloc[i]['Variant ID'] not in used_variant_ids]
                remaining_products = features_df.iloc[remaining_indices][['Variant ID', 'Color', 'Neck_inventory', 'Prints_inventory', 'Sleeves_inventory']]
                remaining_products = remaining_products.drop_duplicates(subset=['Variant ID']).head(3 - len(similar_products))
                similar_products = pd.concat([similar_products, remaining_products]).head(3)

            # Merge with df to include product_url and url, ensuring uniqueness
            df_merge = df[['Variant ID', 'product_url', 'url']].drop_duplicates(subset=['Variant ID']).copy()
            df_merge['Variant ID'] = df_merge['Variant ID'].astype(str)
            similar_products_with_urls = pd.merge(similar_products, df_merge, on='Variant ID', how='left')

            # Ensure exactly 3 unique products are displayed
            similar_products_with_urls = similar_products_with_urls.drop_duplicates(subset=['Variant ID']).head(3)

            # Display the top 3 unique products
            st.subheader("Top 3 Unique Similar Products:")
            st.dataframe(similar_products_with_urls)

            # Log for debugging
            for _, product in similar_products_with_urls.iterrows():
                variant_id = product['Variant ID']
                product_url = product['product_url']
                url = product['url']
                logger.info(f"Variant ID: {variant_id}, Product URL: {product_url}, URL: {url}")
                if product_url and not product_url.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    logger.warning(f"Product URL for Variant ID {variant_id} may not be an image: {product_url}")
                if url and not url.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    logger.warning(f"URL for Variant ID {variant_id} may not be an image: {url}")

            # Check for specific ideal match
            ideal_match_id = "1938244179932"
            ideal_match_row = features_df[features_df["Variant ID"] == ideal_match_id]
            if not ideal_match_row.empty:
                ideal_match_index = ideal_match_row.index[0]
                if 0 <= ideal_match_index < len(features_df):
                    ideal_match_similarity = similarities[ideal_match_index]
                    if ideal_match_index in top_indices[:3]:
                        st.success(f"Ideal Match Found! (Similarity: {ideal_match_similarity:.4f})")

            # Prepare and display image recommendations
            product_recommendations = prepare_unique_image_recommendations(similar_products_with_urls, df)
            if product_recommendations:
                st.subheader("Recommended Products (Images):")
                display_recommendations_with_images(product_recommendations)

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            logger.error(f"Image processing error: {str(e)}")

else:
    st.header("Chatbot")
    # Display conversation history (the chat log)
    for chat in st.session_state['conversation_history']:
        if chat['role'] == 'user':
            st.markdown(f"**You:** {chat['content']}")
        else:
            st.markdown(f"**Assistant:** {chat['content']}")

    # Define options for new user filters
    product_types = df['Product Type'].unique().tolist()
    colors = df['Color'].unique().tolist()
    prints_inventory = df['Prints_order'].unique().tolist()
    sleeves_inventory = df['Sleeves_order'].unique().tolist()

    # Ensure 'my_text' and 'widget' are in session state
    if "my_text" not in st.session_state:
        st.session_state.my_text = ""

    def submit():
        # Process the input and determine the next state
        user_input = st.session_state.widget
        if user_input:
            st.session_state['conversation_history'].append({"role": "user", "content": user_input})

            if st.session_state['user_state'] == 'start':
                st.session_state['conversation_history'].append({"role": "assistant", "content": "How can I help you today? Would you like me to recommend new products or look at similar products to your previous purchase?"})
                st.session_state['user_state'] = 'waiting_for_user_type'

            elif st.session_state['user_state'] == 'waiting_for_user_type':
                if 'new' in user_input.lower() or 'recommend new' in user_input.lower():
                    st.session_state['conversation_history'].append({"role": "assistant", "content": "What color are you looking for?"})
                    st.session_state['user_state'] = 'waiting_for_color'
                elif 'old' in user_input.lower() or 'similar products' in user_input.lower() or 'previous purchase' in user_input.lower():
                    st.session_state['conversation_history'].append({"role": "assistant", "content": "Please provide your requester ID."})
                    st.session_state['user_state'] = 'waiting_for_requester_id'
                else:
                    st.session_state['conversation_history'].append({"role": "assistant", "content": "I didn't understand that. Please specify if you are a new user or an old user."})

            elif st.session_state['user_state'] == 'waiting_for_requester_id':
                print(f"User ID being processed: {user_input}")
                print(f"Type of user_id: {type(user_input)}")
                recommendations = recommender.recommend(user_id=user_input)
                print(f"Recommendations returned: {recommendations}")
                # Format recommendations to include image-related fields for old users
                formatted_recommendations = []
                for rec in recommendations['recommendations']:
                    variant_id = rec.get('product_id', rec.get('Variant ID', None))  # Handle both possible keys
                    if variant_id:
                        product_data = df[df['Variant ID'] == variant_id].iloc[0] if not df[df['Variant ID'] == variant_id].empty else None
                        if product_data is not None:
                            formatted_recommendations.append({
                                'Product ID': product_data['Product ID'],
                                'product_url': product_data['product_url'],
                                'url': product_data['product_url']  # Use product_url as the image source
                            })
                st.session_state['conversation_history'].append({"role": "assistant", "content": f"Recommendations for old user {user_input}: {formatted_recommendations}"})
                if formatted_recommendations:
                    display_recommendations_with_images(formatted_recommendations)
                st.session_state['user_state'] = 'start'

            elif st.session_state['user_state'] == 'waiting_for_color':
                closest_color = match_closest_option(user_input, colors)
                if closest_color:
                    st.session_state['user_filters']['Color'] = closest_color
                    st.session_state['conversation_history'].append({"role": "assistant", "content": f"Got it. You chose color: {closest_color}. Would you like to add more filters? (yes/no)"})
                    st.session_state['user_state'] = 'waiting_for_more_filters'
                else:
                    st.session_state['conversation_history'].append({"role": "assistant", "content": f"I couldn't find a matching color. Available options are: {', '.join(colors)}. Please try again."})

            elif st.session_state['user_state'] == 'waiting_for_more_filters':
                if 'yes' in user_input.lower():
                    st.session_state['conversation_history'].append({"role": "assistant", "content": "What sleeve length are you looking for?"})
                    st.session_state['user_state'] = 'waiting_for_sleeve_length'
                else:
                    recommendations = recommender.recommend(user_filters=st.session_state['user_filters'])
                    print(f"Recommendations returned for filters: {recommendations}")
                    # Format recommendations to include image-related fields for new users
                    formatted_recommendations = []
                    for rec in recommendations['recommendations']:
                        variant_id = rec.get('Variant ID', None)  # Use 'Variant ID' as the key for new user recommendations
                        if variant_id:
                            product_data = df[df['Variant ID'] == variant_id].iloc[0] if not df[df['Variant ID'] == variant_id].empty else None
                            if product_data is not None:
                                formatted_recommendations.append({
                                    'Product ID': product_data['Product ID'],
                                    'product_url': product_data['product_url'],
                                    'url': product_data['product_url']  # Use product_url as the image source
                                })
                    st.session_state['conversation_history'].append({"role": "assistant", "content": f"Recommendations based on your filters: {formatted_recommendations}"})
                    if formatted_recommendations:
                        display_recommendations_with_images(formatted_recommendations)
                    st.session_state['user_state'] = 'start'
                    st.session_state['user_filters'] = {}  # Clear filters for next session

            elif st.session_state['user_state'] == 'waiting_for_sleeve_length':
                closest_sleeve = match_closest_option(user_input, sleeves_inventory)
                if closest_sleeve:
                    st.session_state['user_filters']['Sleeves_order'] = closest_sleeve
                    st.session_state['conversation_history'].append({"role": "assistant", "content": f"Got it. You chose sleeve length: {closest_sleeve}. Would you like to add more filters? (yes/no)"})
                    st.session_state['user_state'] = 'waiting_for_more_filters_again'
                else:
                    st.session_state['conversation_history'].append({"role": "assistant", "content": f"I couldn't find a matching sleeve length. Available options are: {', '.join(sleeves_inventory)}. Please try again."})

            elif st.session_state['user_state'] == 'waiting_for_more_filters_again':
                if 'yes' in user_input.lower():
                    st.session_state['conversation_history'].append({"role": "assistant", "content": "What print are you looking for?"})
                    st.session_state['user_state'] = 'waiting_for_print'
                else:
                    recommendations = recommender.recommend(user_filters=st.session_state['user_filters'])
                    print(f"Recommendations returned for filters: {recommendations}")
                    # Format recommendations to include image-related fields for new users
                    formatted_recommendations = []
                    for rec in recommendations['recommendations']:
                        variant_id = rec.get('Variant ID', None)  # Use 'Variant ID' as the key for new user recommendations
                        if variant_id:
                            product_data = df[df['Variant ID'] == variant_id].iloc[0] if not df[df['Variant ID'] == variant_id].empty else None
                            if product_data is not None:
                                formatted_recommendations.append({
                                    'Product ID': product_data['Product ID'],
                                    'product_url': product_data['product_url'],
                                    'url': product_data['product_url']  # Use product_url as the image source
                                })
                    st.session_state['conversation_history'].append({"role": "assistant", "content": f"Recommendations based on your filters: {formatted_recommendations}"})
                    if formatted_recommendations:
                        display_recommendations_with_images(formatted_recommendations)
                    st.session_state['user_state'] = 'start'
                    st.session_state['user_filters'] = {}  # Clear filters for next session

            elif st.session_state['user_state'] == 'waiting_for_print':
                closest_print = match_closest_option(user_input, prints_inventory)
                if closest_print:
                    st.session_state['user_filters']['Prints_order'] = closest_print
                    st.session_state['conversation_history'].append({"role": "assistant", "content": f"Got it. You chose print: {closest_print}. Would you like to add more filters? (yes/no)"})
                    st.session_state['user_state'] = 'waiting_for_more_filters_final'
                else:
                    st.session_state['conversation_history'].append({"role": "assistant", "content": f"I couldn't find a matching print. Available options are: {', '.join(prints_inventory)}. Please try again."})

            elif st.session_state['user_state'] == 'waiting_for_more_filters_final':
                if 'yes' in user_input.lower():
                    st.session_state['conversation_history'].append({"role": "assistant", "content": "What product type are you looking for?"})
                    st.session_state['user_state'] = 'waiting_for_product_type'
                else:
                    recommendations = recommender.recommend(user_filters=st.session_state['user_filters'])
                    print(f"Recommendations returned for filters: {recommendations}")
                    # Format recommendations to include image-related fields for new users
                    formatted_recommendations = []
                    for rec in recommendations['recommendations']:
                        variant_id = rec.get('Variant ID', None)  # Use 'Variant ID' as the key for new user recommendations
                        if variant_id:
                            product_data = df[df['Variant ID'] == variant_id].iloc[0] if not df[df['Variant ID'] == variant_id].empty else None
                            if product_data is not None:
                                formatted_recommendations.append({
                                    'Product ID': product_data['Product ID'],
                                    'product_url': product_data['product_url'],
                                    'url': product_data['product_url']  # Use product_url as the image source
                                })
                    st.session_state['conversation_history'].append({"role": "assistant", "content": f"Recommendations based on your filters: {formatted_recommendations}"})
                    if formatted_recommendations:
                        display_recommendations_with_images(formatted_recommendations)
                    st.session_state['user_state'] = 'start'
                    st.session_state['user_filters'] = {}  # Clear filters for next session

            elif st.session_state['user_state'] == 'waiting_for_product_type':
                closest_product_type = match_closest_option(user_input, product_types)
                if closest_product_type:
                    st.session_state['user_filters']['Product Type'] = closest_product_type
                    st.session_state['conversation_history'].append({"role": "assistant", "content": f"Got it. You chose product type: {closest_product_type}. Would you like to add more filters? (yes/no)"})
                    st.session_state['user_state'] = 'waiting_for_more_filters_final'
                else:
                    st.session_state['conversation_history'].append({"role": "assistant", "content": f"I couldn't find a matching product type. Available options are: {', '.join(product_types)}. Please try again."})

        # Clear the input field
        st.session_state.widget = ""

    # Input box with the submit function
    st.text_input("You:", key="widget", on_change=submit)

# Debug: Print dataframe info to console
logger.info("Dataframe columns: " + ", ".join(df.columns))
logger.info("Unique Product IDs: " + str(df['Product ID'].unique()))
logger.info("Unique Product URLs: " + str(df['product_url'].unique()[:5]))  # Show first 5 unique product URLs
