var socket = io();
var questionNum = 1; //Starts at two because question 1 is already present

function updateDatabase() {
    var questions = [];
    var name = document.getElementById('name').value;
    for (var i = 1; i <= questionNum; i++) {
        var question = document.getElementById('q' + i).value;
        var answer1 = document.getElementById(i + 'a1').value;
        var answer2 = document.getElementById(i + 'a2').value;
        var answer3 = document.getElementById(i + 'a3').value;
        var answer4 = document.getElementById(i + 'a4').value;
        var correct = document.getElementById('correct' + i).value;
        var answers = [answer1, answer2, answer3, answer4];
        const urlImage = document.getElementById('urlImage' + i).value;
        questions.push({ "question": question, "answers": answers, "correct": correct, urlImage })
    }

    var quiz = { id: 0, "name": name, "questions": questions };
    socket.emit('newQuiz', quiz);
}

function addQuestion() {
    questionNum += 1;

    var questionsDiv = document.getElementById('allQuestions');

    var newQuestionDiv = document.createElement("div");

    var questionLabel = document.createElement('label');
    var questionField = document.createElement('input');

    var fileForm = document.createElement('form');
    fileForm.setAttribute('onsubmit', 'predict(event)');

    var fileInput = document.createElement('input');
    fileInput.setAttribute('type', 'file');
    fileInput.setAttribute('id', 'fileInput' + questionNum); // Give a unique ID

    var predictButton = document.createElement('button');
    predictButton.innerHTML = 'predict';

    let img = document.createElement("img")
    img.src = ""
    img.setAttribute('id','uploadedImage'+questionNum)
    img.style.display = 'none' // "display:none; width: 200px; height: auto;"
    img.style.width = '200px' // "display:none; width: 200px; height: auto;"
    img.style.height = 'auto' // "display:none; width: 200px; height: auto;"

    let imageInput = document.createElement('input');
    imageInput.setAttribute('type', 'text');
    imageInput.setAttribute('id', 'urlImage' + questionNum); // Give a unique ID
    imageInput.setAttribute("hidden","true")

    var answer1Label = document.createElement('label');
    var answer1Field = document.createElement('input');

    var answer2Label = document.createElement('label');
    var answer2Field = document.createElement('input');

    var answer3Label = document.createElement('label');
    var answer3Field = document.createElement('input');

    var answer4Label = document.createElement('label');
    var answer4Field = document.createElement('input');

    var correctLabel = document.createElement('label');
    var correctField = document.createElement('input');

    questionLabel.innerHTML = "Question " + String(questionNum) + ": ";
    questionField.setAttribute('class', 'question');
    questionField.setAttribute('id', 'q' + String(questionNum));
    questionField.setAttribute('type', 'text');

    answer1Label.innerHTML = "Answer 1: ";
    answer2Label.innerHTML = " Answer 2: ";
    answer3Label.innerHTML = "Answer 3: ";
    answer4Label.innerHTML = " Answer 4: ";
    correctLabel.innerHTML = "Correct Answer (1-4): ";

    answer1Field.setAttribute('id', String(questionNum) + "a1");
    answer1Field.setAttribute('type', 'text');
    answer2Field.setAttribute('id', String(questionNum) + "a2");
    answer2Field.setAttribute('type', 'text');
    answer3Field.setAttribute('id', String(questionNum) + "a3");
    answer3Field.setAttribute('type', 'text');
    answer4Field.setAttribute('id', String(questionNum) + "a4");
    answer4Field.setAttribute('type', 'text');
    correctField.setAttribute('id', 'correct' + String(questionNum));
    correctField.setAttribute('type', 'number');

    newQuestionDiv.setAttribute('id', 'question-field');

    // Append question label and field
    newQuestionDiv.appendChild(questionLabel);
    newQuestionDiv.appendChild(questionField);
    newQuestionDiv.appendChild(document.createElement('br'));
    newQuestionDiv.appendChild(document.createElement('br'));

    // Append file input form and predict button
    fileForm.appendChild(fileInput);
    fileForm.appendChild(predictButton);
    newQuestionDiv.appendChild(fileForm);
    newQuestionDiv.appendChild(img)
    newQuestionDiv.appendChild(imageInput)
    newQuestionDiv.appendChild(document.createElement('br'));
    newQuestionDiv.appendChild(document.createElement('br'));

    // Append answer labels and fields
    newQuestionDiv.appendChild(answer1Label);
    newQuestionDiv.appendChild(answer1Field);
    newQuestionDiv.appendChild(answer2Label);
    newQuestionDiv.appendChild(answer2Field);
    newQuestionDiv.appendChild(document.createElement('br'));
    newQuestionDiv.appendChild(document.createElement('br'));
    newQuestionDiv.appendChild(answer3Label);
    newQuestionDiv.appendChild(answer3Field);
    newQuestionDiv.appendChild(answer4Label);
    newQuestionDiv.appendChild(answer4Field);
    newQuestionDiv.appendChild(document.createElement('br'));
    newQuestionDiv.appendChild(document.createElement('br'));

    // Append correct answer label and field
    newQuestionDiv.appendChild(correctLabel);
    newQuestionDiv.appendChild(correctField);

    questionsDiv.appendChild(document.createElement('br')); // Creates a break between each question
    questionsDiv.appendChild(newQuestionDiv); // Adds the question div to the screen

    newQuestionDiv.style.backgroundColor = randomColor();
}


//Called when user wants to exit quiz creator
function cancelQuiz() {
    if (confirm("Are you sure you want to exit? All work will be DELETED!")) {
        window.location.href = "../";
    }
}

socket.on('startGameFromCreator', function (data) {
    window.location.href = "../../host/?id=" + data;
});

function randomColor() {

    var colors = ['#4CAF50', '#f94a1e', '#3399ff', '#ff9933'];
    var randomNum = Math.floor(Math.random() * 4);
    return colors[randomNum];
}

function setBGColor() {
    var randColor = randomColor();
    document.getElementById('question-field').style.backgroundColor = randColor;
}

async function predict(event) {
    event.preventDefault(); // Prevent form submission and page reload

    const fileInput = document.getElementById(`fileInput${questionNum}`);
    const file = fileInput.files[0]; // Get the selected file

    if (!file) {
        alert('Please select a file to upload.');
        return;
    }

    // Prepare the form data with the selected file
    const formData = new FormData();
    formData.append('file', file); // 'file' is the key expected by the API

    try {
        //save image
        const uploadResp = await fetch('http://localhost:3000/upload', {
            method: 'POST',
            body: formData, // Send the form data with the file
            // No need for headers like 'Content-Type', as FormData automatically sets the correct multipart headers
        });

        if (!uploadResp.ok) {
            throw new Error('API request failed');
        }
        const fileResp = await uploadResp.json()

        // Simulate an API call (replace this with the actual API URL)
        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            body: formData, // Send the form data with the file
            // No need for headers like 'Content-Type', as FormData automatically sets the correct multipart headers
        });

        if (!response.ok) {
            throw new Error('API request failed');
        }

        const data = await response.json(); // Assuming the API returns JSON data
        //แปลงเป็นไทยอีกที
        const predictions = `${data.pred_gender} - ${data.pred_race} - อายุ ${getRange(data.pred_age)} ปี `; // Assuming API returns an array of predictions

        // Shuffle predictions randomly and assign them to answer fields
        const random = Math.floor(Math.random() * 4 + 1)

        const replace = `${questionNum}a${random}`

        //reset ค่า
        document.getElementById(`${questionNum}a1`).value = ""
        document.getElementById(`${questionNum}a2`).value = ""
        document.getElementById(`${questionNum}a3`).value = ""
        document.getElementById(`${questionNum}a4`).value = ""

        console.log(fileResp.fileName)

        //replace question
        document.getElementById(`q${questionNum}`).value = "เกมทายเพศชาติ และอายุ สุดหรรษา"

        //แปะ path รูป
        document.getElementById(`urlImage${questionNum}`).value = fileResp.fileName

        //แปะคำตอบของ AI
        document.getElementById(replace).value = predictions
        document.getElementById('correct' + questionNum).value = random
        for (let i = 1; i < 5; i++) {
            if (i != random) {
                document.getElementById(`${questionNum}a${i}`).value = generateRandomData()
            }

        }

        // Display the uploaded image
        const imageUrl = URL.createObjectURL(file); // Use URL.createObjectURL to create a URL for the file
        const imgElement = document.getElementById(`uploadedImage${questionNum}`);
        imgElement.src = imageUrl;
        imgElement.style.display = 'block'; // Show the image
        // Assign the shuffled predictions to answer fields

    } catch (error) {
        console.error('Error while predicting:', error);
        alert('Failed to fetch predictions');
    }
}



function getRange(number) {
    // Determine the start of the range
    let start = Math.floor((number - 1) / 10) * 10 + 1;
    // Determine the end of the range
    let end = start + 9;

    return `${start}-${end}`;
}

// Gender and race dictionaries
const gender_dict = { 0: "Male", 1: "Female" };
const race_dict = { 0: "White", 1: "Black", 2: "Asian", 3: "Indian", 4: "Others" };

// Random number generator function
function getRandomInt(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

// Function to get range (1-10, 11-20, etc.)
function getRange(age) {
    let start = Math.floor((age - 1) / 10) * 10 + 1;
    let end = start + 9;
    return `${start}-${end}`;
}

// Function to generate random data and format the output
function generateRandomData() {
    let pred_gender = gender_dict[getRandomInt(0, 1)];
    let pred_race = race_dict[getRandomInt(0, 4)];
    let pred_age = getRandomInt(1, 100);

    return `${pred_gender} - ${pred_race} - อายุ ${getRange(pred_age)} ปี`;
}
console.log(generateRandomData())