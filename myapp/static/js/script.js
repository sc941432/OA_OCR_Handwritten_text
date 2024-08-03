// static/js/script.js

document.addEventListener("DOMContentLoaded", function () {
    console.log("Page loaded successfully!");
    
    const modelForm = document.querySelector('form[action=""]');
    if (modelForm) {
        modelForm.addEventListener('submit', function (event) {
            event.preventDefault();
            const selectedModel = document.querySelector('input[name="model"]:checked').value;
            if (selectedModel === "google") {
                window.location.href = '/google-vision/';
            } else if (selectedModel === "custom") {
                window.location.href = '/custom-model/';
            }
        });
    }
    
    const selectElement = document.querySelector('.image-select');
    if (selectElement) {
        selectElement.addEventListener('change', function (event) {
            const selectedImage = event.target.value;
            if (selectedImage) {
                alert('You have selected: ' + selectedImage);
            }
        });
    }
});
