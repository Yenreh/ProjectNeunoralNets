// JavaScript para la aplicación de demo de modelos MLP

document.addEventListener('DOMContentLoaded', function() {
    // Elementos del DOM
    const predictionForm = document.getElementById('prediction-form');
    const textInput = document.getElementById('text-input');
    const predictBtn = document.getElementById('predict-btn');
    const randomSampleBtn = document.getElementById('random-sample-btn');
    const predictRandomBtn = document.getElementById('predict-random-btn');
    const loadingSpinner = document.getElementById('loading-spinner');
    const predictionResults = document.getElementById('prediction-results');
    const noResults = document.getElementById('no-results');

    let currentRandomSample = null;

    // Manejar envío del formulario de predicción manual
    if (predictionForm) {
        predictionForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const text = textInput.value.trim();
            
            if (!text) {
                alert('Por favor, ingresa un texto para predecir.');
                return;
            }
            
            makePrediction(text, false);
        });
    }

    // Manejar botón de muestra aleatoria
    if (randomSampleBtn) {
        randomSampleBtn.addEventListener('click', function() {
            getRandomSample();
        });
    }

    // Manejar predicción de muestra aleatoria
    if (predictRandomBtn) {
        predictRandomBtn.addEventListener('click', function() {
            if (currentRandomSample) {
                makePrediction(currentRandomSample.text, true, currentRandomSample.true_stars);
            }
        });
    }

    // Función para hacer predicciones
    function makePrediction(text, isRandom = false, trueStars = null) {
        showLoading();
        
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model_name: MODEL_NAME,
                text_input: text
            })
        })
        .then(response => response.json())
        .then(data => {
            hideLoading();
            
            if (data.error) {
                alert('Error: ' + data.error);
                return;
            }
            
            displayResults(data, isRandom, trueStars);
        })
        .catch(error => {
            hideLoading();
            console.error('Error:', error);
            alert('Error al hacer la predicción. Por favor, intenta de nuevo.');
        });
    }

    // Función para obtener muestra aleatoria
    function getRandomSample() {
        fetch(`/random_sample/${MODEL_NAME}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert('Error: ' + data.error);
                return;
            }
            
            currentRandomSample = data;
            displayRandomSample(data);
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error al obtener muestra aleatoria.');
        });
    }

    // Función para mostrar muestra aleatoria
    function displayRandomSample(data) {
        const randomText = document.getElementById('random-text');
        const trueStars = document.getElementById('true-stars');
        const randomContent = document.getElementById('random-sample-content');
        
        if (randomText && trueStars && randomContent) {
            randomText.textContent = data.text;
            trueStars.innerHTML = generateStarsHTML(data.true_stars) + ` (${data.true_stars} estrellas)`;
            randomContent.style.display = 'block';
        }
    }

    // Función para mostrar resultados
    function displayResults(data, isRandom = false, trueStars = null) {
        const predictedStarsEl = document.getElementById('predicted-stars');
        const confidenceBar = document.getElementById('confidence-bar');
        const confidenceText = document.getElementById('confidence-text');
        const classProbabilities = document.getElementById('class-probabilities');
        const comparisonSection = document.getElementById('comparison-section');
        
        if (predictedStarsEl) {
            predictedStarsEl.innerHTML = generateStarsHTML(data.predicted_stars);
        }
        
        if (confidenceBar && confidenceText) {
            const confidencePercent = (data.confidence * 100).toFixed(1);
            confidenceBar.style.width = confidencePercent + '%';
            confidenceText.textContent = confidencePercent + '%';
            
            // Colorear barra según confianza
            confidenceBar.className = 'progress-bar';
            if (data.confidence > 0.8) {
                confidenceBar.classList.add('bg-success');
            } else if (data.confidence > 0.6) {
                confidenceBar.classList.add('bg-warning');
            } else {
                confidenceBar.classList.add('bg-danger');
            }
        }
        
        // Mostrar probabilidades por clase
        if (classProbabilities) {
            classProbabilities.innerHTML = '';
            Object.entries(data.class_probabilities).forEach(([className, probability]) => {
                const item = document.createElement('div');
                item.className = 'probability-item';
                
                const stars = parseInt(className.split(' ')[0]);
                const probabilityPercent = (probability * 100).toFixed(1);
                
                item.innerHTML = `
                    <span>${generateStarsHTML(stars)}</span>
                    <span class="text-muted">${probabilityPercent}%</span>
                `;
                classProbabilities.appendChild(item);
            });
        }
        
        // Mostrar comparación si es muestra aleatoria
        if (isRandom && trueStars && comparisonSection) {
            const predStarsComparison = document.getElementById('pred-stars-comparison');
            const trueStarsComparison = document.getElementById('true-stars-comparison');
            const accuracyIndicator = document.getElementById('accuracy-indicator');
            
            if (predStarsComparison && trueStarsComparison && accuracyIndicator) {
                predStarsComparison.innerHTML = generateStarsHTML(data.predicted_stars);
                trueStarsComparison.innerHTML = generateStarsHTML(trueStars);
                
                const isCorrect = data.predicted_stars === trueStars;
                accuracyIndicator.textContent = isCorrect ? 'Correcto' : 'Incorrecto';
                accuracyIndicator.className = isCorrect ? 'badge accuracy-correct' : 'badge accuracy-incorrect';
                
                comparisonSection.style.display = 'block';
            }
        } else if (comparisonSection) {
            comparisonSection.style.display = 'none';
        }
        
        // Mostrar resultados con animación
        if (predictionResults) {
            predictionResults.style.display = 'block';
            predictionResults.classList.add('fade-in');
        }
        
        if (noResults) {
            noResults.style.display = 'none';
        }
    }

    // Función para generar HTML de estrellas
    function generateStarsHTML(stars) {
        let html = '<span class="stars">';
        for (let i = 1; i <= 5; i++) {
            if (i <= stars) {
                html += '<i class="fas fa-star"></i>';
            } else {
                html += '<i class="far fa-star"></i>';
            }
        }
        html += '</span>';
        return html;
    }

    // Función para mostrar loading
    function showLoading() {
        if (loadingSpinner) loadingSpinner.style.display = 'block';
        if (predictionResults) predictionResults.style.display = 'none';
        if (noResults) noResults.style.display = 'none';
        if (predictBtn) predictBtn.disabled = true;
        if (predictRandomBtn) predictRandomBtn.disabled = true;
    }

    // Función para ocultar loading
    function hideLoading() {
        if (loadingSpinner) loadingSpinner.style.display = 'none';
        if (predictBtn) predictBtn.disabled = false;
        if (predictRandomBtn) predictRandomBtn.disabled = false;
    }

    // Ejemplo de textos útiles
    const exampleTexts = [
        "Este producto es excelente, muy buena calidad y llegó rápido. Lo recomiendo totalmente.",
        "Terrible producto, no funciona como se describe. Muy decepcionante y mala calidad.",
        "El producto está bien, cumple con lo básico. Nada extraordinario pero aceptable.",
        "Increíble calidad, superó mis expectativas completamente. Cinco estrellas sin duda.",
        "No recomiendo este producto, muy mala experiencia de compra. Perdí mi dinero."
    ];

    // Botón para ejemplos rápidos (si existe)
    const exampleBtn = document.getElementById('example-btn');
    if (exampleBtn && textInput) {
        exampleBtn.addEventListener('click', function() {
            const randomExample = exampleTexts[Math.floor(Math.random() * exampleTexts.length)];
            textInput.value = randomExample;
            textInput.focus();
        });
    }
});