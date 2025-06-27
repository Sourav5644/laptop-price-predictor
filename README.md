# 💻 Laptop Price Predictor — End-to-End ML Project with MLOps 🚀

An end-to-end Machine Learning project to predict laptop prices based on hardware configurations like RAM, CPU, GPU, and more. This project is production-ready, fully integrated with **MongoDB Atlas**, **AWS S3**, **CI/CD pipeline via GitHub Actions**, **Docker**, and **FastAPI** for serving the model. Perfect example of how MLOps can take a data science model from notebook to a scalable, cloud-deployed app.

---

## 📂 Project Structure

```

├── src/
│   ├── components/              # All ML components (ingestion, validation, training, etc.)
│   ├── configuration/          # MongoDB and AWS configurations
│   ├── entity/                 # Entity classes for configs and artifacts
│   ├── exception/              # Custom exception handling
│   ├── logger/                 # Logging setup
│   ├── pipeline/               # Training pipeline
│   ├── prediction/             # FastAPI prediction pipeline
│   ├── aws\_storage/            # AWS S3 interaction
│   ├── utils/                  # Utility functions
│   └── constants/              # Constant values and thresholds
│
├── templates/                  # HTML templates for the frontend
├── static/                     # Static files (CSS, JS)
├── notebook/                   # EDA & MongoDB integration notebooks
├── .github/workflows/          # GitHub Actions CI/CD setup
├── Dockerfile                  # Docker config for containerization
├── .dockerignore               # Files to ignore in Docker build
├── requirements.txt            # Python dependencies
├── setup.py                    # Project packaging
├── pyproject.toml              # Python build system config
├── demo.py                     # Initial test script
└── app.py                      # FastAPI backend entry point

````

---

## 📌 Step-by-Step Setup Instructions

### 1. 🔧 Project Initialization
```bash
python template.py             # Generate folder structure
````

### 2. 📦 Packaging

Set up `setup.py` and `pyproject.toml` to install local modules.

> 📘 Refer to `crashcourse.txt` to understand packaging.

---

### 3. 🐍 Create Virtual Environment

```bash
conda create -n laptop python=3.10 -y
conda activate laptop
pip install -r requirements.txt
pip list     # Confirm packages are installed
```

---

## 🌐 MongoDB Atlas Integration

1. Sign up on [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)

2. Create a cluster (M0 Free Tier)

3. Create a user and allow IP `0.0.0.0/0`

4. Copy connection string for Python driver

5. Use the connection string in `.env` or environment variable:

   ```bash
   export MONGODB_URL="mongodb+srv://<user>:<password>@cluster.mongodb.net/db"
   ```

6. Load your dataset into MongoDB using the `mongoDB_demo.ipynb` notebook.

7. Verify data in the MongoDB collection (key-value format).

---

## 🧰 Logging & Exception Handling

* Create logger in `src/logger/logging.py` and test via `demo.py`
* Create custom exception class in `src/exception/exception.py`

---

## 📊 Data Ingestion

* Define configs in `constants/`, `config_entity.py`, and `artifact_entity.py`
* Connect MongoDB and convert data to Pandas DataFrame
* Run Data Ingestion pipeline via `demo.py`

---

## ✅ Data Validation & Transformation

* Validate dataset against schema (`schema.yaml`)
* Apply transformations: missing values, feature encoding, scaling
* Save transformation object using `estimator.py` for future reuse

---

## 🧠 Model Training

* Train regression model on transformed data
* Save trained model and metrics

---

## 🧪 Model Evaluation (with AWS S3 Integration)

* Set up AWS account, IAM user, and configure credentials:

```bash
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
```

* Push models to S3 bucket: `my-model-mlopsproj`
* Track performance threshold to avoid downgrading model accuracy

---

## 🚚 Model Deployment (Model Pusher)

* Push best-performing model to production directory or S3
* Serve the model through FastAPI app (`app.py`)

---

## 🌐 Prediction API with FastAPI

* Frontend with HTML form (`templates/index.html`)
* Backend using FastAPI for form input → prediction
* Hosted at: `http://<EC2-IP>:5000`

---

## 🐳 Docker Integration

* Create `Dockerfile` and `.dockerignore`
* Build Docker image:

```bash
docker build -t laptop-price-predictor .
```

---

## 🔁 CI/CD with GitHub Actions & AWS

### ✅ Steps:

* Setup GitHub Actions workflow in `.github/workflows/aws.yaml`
* Build and push Docker image to AWS ECR
* Deploy image to EC2 using a self-hosted runner

### ⚙ GitHub Secrets to Set:

* `AWS_ACCESS_KEY_ID`
* `AWS_SECRET_ACCESS_KEY`
* `AWS_DEFAULT_REGION`
* `ECR_REPO`

---

## ☁️ Deploy on EC2

1. Launch EC2 instance (Ubuntu)
2. Install Docker:

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu
newgrp docker
```

3. Set up EC2 as a GitHub self-hosted runner
4. Expose port 5000 (security group inbound rule)

---

## ✅ Model Training & Testing Endpoints

| Route      | Description                    |
| ---------- | ------------------------------ |
| `/`        | Home page with form            |
| `/predict` | Submit form and get prediction |
| `/train`   | Train and save new model       |

---

## 📈 Technologies Used

| Domain               | Stack                                    |
| -------------------- | ---------------------------------------- |
| Language             | Python 3.10                              |
| Data                 | Pandas, NumPy                            |
| ML                   | scikit-learn                             |
| Logging & Exceptions | Custom logger & exception classes        |
| Database             | MongoDB Atlas                            |
| Backend              | FastAPI                                  |
| DevOps               | Docker, GitHub Actions, AWS EC2, S3, ECR |
| CI/CD                | GitHub Actions with self-hosted runner   |
| Deployment           | EC2 Ubuntu + Docker container            |

---

## 🌟 Highlighted Features

* ✅ Real-world MLOps practices
* ✅ MongoDB Atlas cloud database integration
* ✅ Full modular architecture with reusable components
* ✅ GitHub Actions for CI/CD pipeline
* ✅ Containerized deployment via Docker
* ✅ Fully deployed prediction system on AWS



---------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------

## 📥 How to Run This Project on Your System

Follow the steps below to run the **Laptop Price Predictor** project from scratch on your own machine:

---

### ✅ 1. **Clone the Repository**

```bash
git clone https://github.com/Sourav5644/laptop-price-predictor
cd laptop-price-predictor
```

---

### ✅ 2. **Create and Activate a Python Environment**

Using **conda**:

```bash
conda create -n laptop python=3.10 -y
conda activate laptop
```

Using **virtualenv** (alternative):

```bash
python -m venv laptop
source laptop/bin/activate  # On Windows: laptop\Scripts\activate
```

---

### ✅ 3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

---

### ✅ 4. **Set MongoDB Connection URL**

You need a free MongoDB Atlas account.

* Go to [MongoDB Atlas](https://www.mongodb.com/cloud/atlas) and:

  * Create a free cluster
  * Add IP address: `0.0.0.0/0`
  * Create a user and get the connection string (driver: Python)
  * Replace `<username>` and `<password>` in your connection string

**Set the environment variable:**

**Linux/macOS (bash):**

```bash
export MONGODB_URL="your_connection_string"
```

**Windows (PowerShell):**

```powershell
$env:MONGODB_URL="your_connection_string"
```

---

### ✅ 5. **Push Your Dataset to MongoDB**

1. Open `notebook/mongoDB_demo.ipynb`
2. Follow the code to upload your dataset to MongoDB Atlas
3. Check uploaded data in "Browse Collections"

---

### ✅ 6. **Run the Full Training Pipeline**

```bash
python demo.py
```

This will perform:

* Data Ingestion
* Data Validation
* Data Transformation
* Model Training
* Model Evaluation
* Model Upload to AWS (if configured)

---

### ✅ 7. **Launch the Web Application**

```bash
python app.py
```

Then open your browser and go to:

```
http://localhost:5000
```

* Fill the form and get laptop price predictions instantly.
* Use `/train` route to retrain the model manually.

---

### ✅ 8. **Docker Deployment (Optional)**

To run with Docker:

```bash
docker build -t laptop-predictor .
docker run -p 5000:5000 laptop-predictor
```

Then visit: `http://localhost:5000`

---

### ✅ 9. **CI/CD (Optional)**

CI/CD is pre-configured using:

* GitHub Actions
* AWS ECR, EC2, and S3

You can update your GitHub repo and it will auto-deploy to your EC2 instance.

---
---

## 🙌 Author

**Sourav Bhardwaj**
*Machine Learning | MLOps | Full Stack Projects Enthusiast*
[LinkedIn](www.linkedin.com/in/sourav-bhardwaj-88b9b7212) | [GitHub](https://github.com/Sourav5644/laptop-price-predictor)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📌 Pro Tip for Recruiters

This project is **not just a notebook**. It reflects a **production-grade ML system**, showcasing:

* Software engineering practices
* Cloud deployment
* Real CI/CD pipeline
* Reusable ML components
* Full lifecycle from data to prediction

👀 Perfect for roles in ML Engineering, MLOps, or Full Stack Data Science!


