function uploadImage() {
    const fileInput = document.getElementById('imageInput');
    const file = fileInput.files[0];
    
    const formData = new FormData();
    formData.append('image', file);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        const predictionResult = document.getElementById('predictionResult');
        predictionResult.innerHTML = `Predicted Class: ${data.prediction}`;
    })
    .catch(error => console.error('Error:', error));
}
