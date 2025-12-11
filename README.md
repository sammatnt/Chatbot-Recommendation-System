# UC-Davis

### 📁 Sample Images

This repository includes sample images in the `/sample_images` folder to **demonstrate and test the image upload functionality** of the system.

These images are used to:
- Validate the **image feature extraction** process.
- Test the **image similarity search** using **cosine similarity**.
- Ensure the end-to-end functionality of the **image-based recommendation engine**.

> 🔍 When a sample image is uploaded, its visual features are extracted and matched against existing entries in the database to return the most similar results.

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
