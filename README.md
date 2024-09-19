# Movie Recommendation System

## Overview
This project is a Streamlit-based web application that provides personalized movie recommendations. It uses collaborative filtering techniques to suggest movies based on user preferences and movie similarities.

## Features
- User-friendly web interface built with Streamlit
- Personalized movie recommendations based on user ID and a liked movie
- Uses the MovieLens 100K dataset for a diverse range of movies
- Implements a KNN (K-Nearest Neighbors) algorithm for collaborative filtering

## Installation

### Prerequisites
- Python 3.7+
- pip (Python package manager)

### Steps
1. Clone the repository:
   ```
   git clone https://github.com/Nne85/SuggestToMe.git
   cd SuggestToMe
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and go to `http://localhost:8501`

3. Enter your user ID (1-943) and select a movie you like from the dropdown menu.

4. Click "Get Recommendations" to see a list of recommended movies based on your selection.

## Data
This project uses the MovieLens 100K dataset. The dataset files should be placed in the `data/ml-100k/` directory.

## File Structure
```
SuggestToMe/
│
├── app_ui/
│   └── streamlit_app.py
│
├── data/
│   ├── ml-100k/
│   │   ├── u.data
│   │   ├── u.item
│   │   └── u.user
│   └── processed/
│       └── ratings_data.csv
│
├── src/
│   ├── data_processing.py
│   └── recommendations.py
│
├── requirements.txt
└── README.md
```

## How It Works
1. The app loads and processes the MovieLens 100K dataset.
2. It uses a KNN algorithm to find movies similar to the one the user likes.
3. It then filters these similar movies based on genre similarity.
4. Finally, it presents the top recommendations to the user.

## Contributing
Contributions to improve the project are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes and commit them (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a new Pull Request

## License
[MIT License](https://opensource.org/licenses/MIT)

## Contact
Nnenna Kehinde - nneukamaka@gmail.com

Project Link: [https://github.com/Nne85/SuggestToMe(https://github.com/Nne85/SuggestToMe