<h1  align=center  >Wildfire Prediction and Visualization Using CAM</h1>

<h2>Overview</h2>
<p> 
This project implements a wildfire prediction model using Convolutional Neural Networks (CNNs) and Class Activation Maps (CAMs). It is designed to classify satellite images into two categories: Wildfire and No Wildfire. The project also provides visual insights into predictions using CAMs, helping users understand which areas of an image contribute most to the classification.
</p>

<!-- <li><a href='https://open.canada.ca/data/en/dataset/9d8f219c-4df0-4481-926f-8a2a532ca003'>Refer to Canada's Website for the Original Wildfires Data</a></li> -->


<h2>Features</h2>
<ul>
<li><strong>Image Classification:</strong> Classifies satellite images into wildfire or no wildfire.</li>
<li><strong>Class Activation Maps (CAMs):</strong> Visualize the regions influencing the predictions.</li>
<li><strong>Custom CNN Architecture:</strong> Includes adjustable convolutional layers for performance tuning.</li>
<li><strong>Interactive GUI:</strong> A simple GUI to upload and classify images interactively.</li>
<li><strong>Data Visualization:</strong> Plots for training accuracy, loss, and class distributions.</li>
</ul>

<h2>Key Functions</h2>
<ol>
  <li>Model Training:
    <ul>
      <li>Adjust model parameters such as input size, batch size, and number of epochs.</li>
      <li>Visualize accuracy and loss during training.</li>
    </ul>
  </li>

  <li>Prediction:
    <ul>      
      <li>Generate predictions for test images and save results in CSV format.</li>
      <li>Visualize Class Activation Maps for deeper insights.</li>
    </ul>
  </li>

  <li>Interactive GUI:
    <ul>
      <li>Simple file upload interface to classify images and visualize CAMs.</li>
    </ul>
  </li>
</ol>

<h2>Dependencies</h2>
 <ul>
    <li>Python 3.12
    <li>TensorFlow
    <li>OpenCV
    <li>NumPy
    <li>Pandas
    <li>Matplotlib
    <li>Seaborn
    <li>Pillow
    <li>Streamlit
  </ul>