import axios from 'axios';
document.getElementById("prediction-form").addEventListener("submit", async function(event) {
    event.preventDefault();
  
    // Collect the form data
    const formData = new FormData(this);
    const formObject = Object.fromEntries(formData.entries());
  
    // Get the model type
    const modelType = formObject["model-type"];
    
    // Prepare the data to be sent to the backend API
    const apiUrl = "http://127.0.0.1:8000/predict/";  // Update with your backend API URL
    const response = await axios.post(apiUrl, {
   
        model_type: modelType,
        Store: parseInt(formObject["Store"]),
        DayOfWeek: parseInt(formObject["DayOfWeek"]),
        Open: parseInt(formObject["Open"]),
        Promo: parseInt(formObject["Promo"]),
        SchoolHoliday: parseInt(formObject["SchoolHoliday"]),
        CompetitionDistance: parseFloat(formObject["CompetitionDistance"]),
        Promo2: parseInt(formObject["Promo2"]),
        Promo2SinceWeek: parseInt(formObject["Promo2SinceWeek"]),
        Promo2SinceYear: parseInt(formObject["Promo2SinceYear"]),
        Year: parseInt(formObject["Year"]),
        Month: parseInt(formObject["Month"]),
        Day: parseInt(formObject["Day"]),
        WeekOfYear: parseInt(formObject["WeekOfYear"]),
        CompetitionOpenSince: parseInt(formObject["CompetitionOpenSince"]),
        Promo2ActiveMonths: parseInt(formObject["Promo2ActiveMonths"]),
        StateHoliday_0: parseInt(formObject["StateHoliday_0"]),
        StateHoliday_a: parseInt(formObject["StateHoliday_a"]),
        StoreType_b: parseInt(formObject["StoreType_b"]),
        StoreType_c: parseInt(formObject["StoreType_c"]),
        StoreType_d: parseInt(formObject["StoreType_d"]),
        Assortment_b: parseInt(formObject["Assortment_b"]),
        Assortment_c: parseInt(formObject["Assortment_c"]),
        PromoInterval_Jan_Apr_Jul_Oct: parseInt(formObject["PromoInterval_Jan_Apr_Jul_Oct"]),
        PromoInterval_Mar_Jun_Sept_Dec: parseInt(formObject["PromoInterval_Mar_Jun_Sept_Dec"]),
        PromoInterval_None: parseInt(formObject["PromoInterval_None"]),
        model_type: modelType,

  

    });
  
    if (response.ok) {
      const result = await response.json();
      displayPrediction(result);
    } else {
      alert("Error: Unable to get prediction from the model.");
    }
  });
  
  function displayPrediction(result) {
    const predictionElement = document.getElementById("prediction-result");
    predictionElement.style.display = "block";
    predictionElement.innerHTML = `<strong>Prediction for ${result.model_type.toUpperCase()}:</strong> ${result.prediction}`;
  }
  