import streamlit as st
from PIL import Image
import io
import requests
import base64

def call_api(image):
    url = "http://localhost:5000/predict"
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)

    response = requests.post(url, files={"file": image_bytes})
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception("Error calling API: " + response.text)


st.title("LuzIA - Retina Image Classifier")

uploaded_file = st.file_uploader("Upload a Retina Image", type=["png", "jpg", "jpeg"])

print_dict = {
    'cataract': {'name': 'Cataract', 0: 'Cataract', 1: 'Normal'},
    'public_excavation': {'name': 'Abnormal excavation', 0: 'Abnormal', 1: 'Normal'},
    'public_dr': {'name': 'Referable DR', 0: 'Referable', 1: 'Normal'},
    'vessel': {'name': 'Abnormal vessels', 0: 'Abnormal', 1: 'Normal'},
}

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Retina Image", use_column_width=True)
    
    try:
        result = call_api(image)

        # Display the JSON prediction results
        st.subheader("Prediction Results")

        for disease in result:
            st.write(f"{print_dict[disease]['name']} Prediction:")
            prediction = result[disease]['prediction']
            mapped_prediction = {print_dict[disease][key]: str(round(value * 100, 2))+"%" 
                                 for key, value in enumerate(prediction)}
            st.write(mapped_prediction)

            if result[disease]['cam'] is not None:
                # Display the returned Class Activation Map (CAM) image
                cam_image = Image.open(io.BytesIO(base64.b64decode(result[disease]['cam'])))
                st.image(cam_image, caption=f"{print_dict[disease]['name']} CAM", use_column_width=True)
            else:
                st.write(f"No {print_dict[disease]['name']} CAM available.")

    except Exception as e:
        st.error(str(e))
