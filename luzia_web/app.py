import streamlit as st
from PIL import Image
import io
import requests
import base64

def call_api(image, model_type):
    url = "https://luziaapi.gabrielaranha.com/predict"
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)

    response = requests.post(url, files={"file": image_bytes}, data={"model": model_type})
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception("Error calling API: " + response.text)


st.title("LuzIA - Retina Image Classifier")

model_type = st.selectbox("Select a Model Type", ["rd", "excavation", "cataract"])

uploaded_file = st.file_uploader("Upload a Retina Image", type=["png", "jpg", "jpeg"])

print_dict = {
    'rd': {'name': 'Referable DR', 1: 'Referable', 0: 'Normal'},
    'excavation': {'name': 'Optic Disc Excavation', 1: 'Abnormal', 0: 'Normal'},
    'cataract': {'name': 'Cataract', 1: 'Present', 0: 'Absent'}
}

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Retina Image", use_column_width=True)
    
    try:
        result = call_api(image, model_type)

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
