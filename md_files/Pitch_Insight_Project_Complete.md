# Pitch Insight — Complete Project Idea, Flow & Working

## 1. Project Overview

**Pitch Insight** is a cricket-pitch analysis system that takes a **manually cropped image of a cricket pitch** and combines:

1. **Computer Vision** — extracts visual pitch characteristics.
2. **Weather Information** — obtains environmental conditions using **WeatherAPI.com**.
3. **Pitch Classification** — uses an existing trained **ResNet-18 ONNX model** as a supporting/baseline component.
4. **Feature-based reasoning** — combines visual and weather features to estimate how the pitch is likely to behave for batting, seam bowling, and spin bowling.
5. **Web application** — React frontend + FastAPI backend.
6. **Database** — MongoDB stores analysis results.

The objective is **not simply image classification**, but a more explainable **pitch-condition and pitch-behaviour analysis** by combining image-derived features with environmental conditions.

---

## 2. Core Idea

A simple classifier may say: **“This pitch is batting-friendly.”**

Pitch behaviour, however, depends on multiple factors such as:

- grass coverage
- visible dryness
- soil/brownness
- cracks
- moisture-related appearance
- surface texture
- temperature
- humidity
- rainfall
- UV index
- wind
- other environmental conditions

Therefore:

```text
                 MANUALLY CROPPED PITCH IMAGE
                              |
                              v
                   Computer Vision Pipeline
                              |
                              v
                  Visual Pitch Features
        +---------------------+---------------------+
        |          |           |          |         |
      Grass     Dryness     Brownness   Cracks   Texture
        |          |           |          |         |
        +----------+-----------+----------+---------+
                              |
                              |
              +---------------+----------------+
              |                                |
              v                                v
        ResNet-18 ONNX                    WeatherAPI.com
        4-class baseline                 Environmental data
              |                                |
              +---------------+----------------+
                              |
                              v
                       Feature Fusion
                              |
                              v
                     Analysis / Reasoning
                              |
                +-------------+-------------+
                |             |             |
                v             v             v
             Condition     Behaviour    Confidence
                |             |             |
                +-------------+-------------+
                              |
                              v
                         Explanation
                              |
                    +---------+---------+
                    |                   |
                    v                   v
                 MongoDB            React UI
```

---

## 3. Manual Cropping Decision

The application assumes the user will **manually crop/select the pitch region** before analysis.

The current system does **not need automatic pitch detection/segmentation**.

This avoids irrelevant regions such as:

- players
- boundary
- advertisements
- sky
- stands
- spectators
- grass outside the pitch
- shadows

Current pipeline:

```text
Original Image
      |
      v
User manually selects pitch
      |
      v
Cropped Pitch Image
      |
      v
Feature Extraction + Classification
```

The system therefore starts its computer-vision analysis **after manual cropping**.

---

## 4. System Architecture

```text
                         USER
                          |
                          v
                +-------------------+
                |   React Frontend  |
                +---------+---------+
                          |
                   Upload / Crop
                          |
                          v
                +-------------------+
                |   FastAPI Backend |
                +---------+---------+
                          |
              +-----------+-----------+
              |                       |
              v                       v
       Image Processing        WeatherAPI.com
              |                       |
              v                       v
       Visual Features         Weather Features
              |                       |
              +-----------+-----------+
                          |
                          v
                +-------------------+
                | Analysis / Fusion  |
                |      Engine        |
                +---------+---------+
                          |
                          v
                +-------------------+
                | Pitch Insight      |
                +---------+---------+
                          |
                          v
                +-------------------+
                |      MongoDB       |
                +-------------------+
                          |
                          v
                +-------------------+
                |  React Dashboard  |
                +-------------------+
```

---

## 5. Frontend — React

### Responsibilities

- Upload pitch image
- Display uploaded image
- Allow manual cropping
- Ask for location if required
- Send cropped image to backend
- Display analysis results
- Display visual feature values
- Display weather information
- Display final pitch insight
- Display confidence and explanation

### UI Flow

```text
PITCH INSIGHT
--------------------------------
Upload Pitch Image
[ Choose Image ]

Crop Your Pitch
+-----------------------+
|                       |
|      PITCH IMAGE      |
|                       |
+-----------------------+

[ Analyze ]
```

Result page can contain:

```text
Pitch Condition
    DRY

Batting Suitability
    80%

Seam Suitability
    50%

Spin Suitability
    70%

Visual Features
    Grass Coverage   61.2%
    Dryness          72%
    Brown Surface    68%
    Crack Presence   Low

Weather
    Temperature      XX°C
    Humidity         XX%
    Rainfall         XX mm
    UV Index         X
    Wind             XX km/h

Insight
    The pitch appears relatively dry...
```

---

## 6. Backend — FastAPI

FastAPI is the central processing layer.

Possible main endpoint:

```text
POST /analyze
```

Input:

```text
image
location
```

Processing:

```text
Receive Image
      |
Validate Image
      |
Preprocess
      |
Extract Visual Features
      |
Get Weather
      |
Run ResNet-18 ONNX
      |
Combine Features
      |
Generate Analysis
      |
Store Result in MongoDB
      |
Return JSON
```

Example response:

```json
{
  "pitch_condition": "Dry",
  "grass_coverage_percent": 61.22,
  "dryness_score": 72.4,
  "brown_surface_percent": 68.1,
  "crack_score": 18.5,
  "weather": {
    "temperature": 31,
    "humidity": 45,
    "rainfall": 0,
    "wind_speed": 12,
    "uv_index": 8
  },
  "prediction": {
    "batting": 0.80,
    "seam": 0.50,
    "spin": 0.70
  },
  "insight": "The pitch appears relatively dry..."
}
```

---

## 7. Computer Vision Pipeline

The key technical component is extracting meaningful information from the **cropped pitch image**.

```text
Pitch Image
     |
     v
Image Preprocessing
     |
     +-- Resize
     +-- Normalize
     +-- Noise handling
     +-- Color-space conversion
     |
     v
Feature Extraction
     |
     +-- Grass
     +-- Dryness
     +-- Brown/soil surface
     +-- Cracks
     +-- Texture
```

The system should not depend only on the neural-network classification.

---

## 8. Grass Coverage

A feature already being extracted is:

```text
grass_coverage_percent
```

Current example:

```text
grass_coverage_percent: 61.2248
```

Interpretation:

> Approximately 61.22% of the analysed pitch region has visual characteristics associated with grass.

Grass presence can influence:

- seam movement
- moisture retention
- early bowling assistance
- surface characteristics

**Important:** grass percentage alone must not determine the final pitch type; it is one feature among several.

---

## 9. Dryness Analysis

Dryness can be represented as:

```text
dryness_score = 0 -> very moist
dryness_score = 100 -> very dry
```

Potential visual indicators:

- lighter/brown surface
- reduced dark/moist appearance
- exposed soil
- surface texture
- cracks
- colour distribution

Dryness becomes one of the major inputs to the reasoning layer.

---

## 10. Brown / Soil Surface

A useful visual feature is the percentage of pixels associated with:

- exposed soil
- brown surface
- dry-surface characteristics

Example:

```text
brown_surface_percent = 68%
```

Higher brown/soil proportion may correlate with a visibly dry surface, but lighting and camera conditions can affect it. Therefore, it should be combined with other indicators.

---

## 11. Crack Detection

Cracks can help identify an older or dry pitch.

Potential processing:

```text
Linear structures
      +
Dark/light contrast
      +
Surface discontinuities
      |
      v
Crack score
```

Example scale:

```text
crack_score = 0   -> no visible cracks
crack_score = 100 -> strong crack presence
```

Not every dark line is a crack, so this remains an estimated feature.

---

## 12. Texture Analysis

Texture can provide additional information about the pitch surface.

Possible techniques:

- Local Binary Patterns (LBP)
- GLCM features
- edge density
- gradient statistics
- variance
- entropy

Possible interpretation:

```text
Texture
  |
  +-- Smooth
  +-- Moderate
  +-- Rough
```

Texture can help distinguish surfaces that simple colour segmentation cannot.

---

## 13. Existing ResNet-18 ONNX Model

The project already has a trained **ResNet-18 model converted from PyTorch to ONNX**.

It has four expert-manually-labelled classes:

```text
1. batting_friendly
2. bowling_friendly
3. seam_friendly
4. spin_friendly
```

The model should be treated as a **baseline/supporting component**, not the sole research contribution.

Pipeline:

```text
Cropped Pitch Image
        |
        v
ResNet-18 ONNX
        |
        +-- batting_friendly
        +-- bowling_friendly
        +-- seam_friendly
        +-- spin_friendly
```

Example:

```text
batting_friendly : 0.62
bowling_friendly : 0.14
seam_friendly    : 0.10
spin_friendly    : 0.14
```

---

## 14. Why Not Depend Only on ResNet?

Pure classification:

```text
Image -> CNN -> Class
```

Pitch Insight:

```text
Image
 |
 +-- Visual features
 |
 +-- CNN prediction
 |
 +-- Weather data
 |
 v
Feature fusion
 |
 v
Explainable insight
```

Instead of only returning:

> Spin-friendly

the system can explain:

> The surface appears relatively dry with high exposed-soil characteristics. Current environmental conditions indicate low humidity and strong UV exposure, which may promote further drying. These factors increase the likelihood of spin assistance as the pitch dries.

This makes the system more useful and explainable.

---

## 15. WeatherAPI.com Integration

Weather data will be obtained from **WeatherAPI.com** using the pitch location.

Potential data:

```text
Temperature
Humidity
Rainfall
Wind speed
Cloud cover
UV index
Weather condition
Pressure
```

Flow:

```text
User
 |
 +-- Pitch Image
 |
 +-- Location
      |
      v
FastAPI
      |
      v
WeatherAPI.com
      |
      v
Weather Data
```

---

## 16. UV Index — Important Feature

UV index should be included because it can contribute to understanding **drying conditions**.

Conceptually:

```text
Higher UV
    |
    v
Greater solar-radiation exposure
    |
    v
Potential contribution to surface drying
    |
    v
Change in pitch moisture
    |
    v
Potential effect on pitch behaviour
```

However, the system should **not claim that UV directly determines pitch behaviour**.

Correct interpretation:

> UV index is an environmental factor that can contribute to drying and should be interpreted together with humidity, rainfall, temperature, wind, sunlight exposure and visual pitch evidence.

---

## 17. Weather Feature Interpretation

### Temperature

Higher temperature can contribute to faster evaporation:

```text
Temperature ↑
     |
     v
Potential evaporation ↑
     |
     v
Potential drying ↑
```

### Humidity

Higher humidity generally reduces evaporation potential:

```text
Humidity ↑
    |
    v
Evaporation tendency ↓
    |
    v
Drying tendency ↓
```

### Rainfall

Recent rainfall can increase surface moisture:

```text
Rainfall ↑
    |
    v
Surface moisture ↑
```

### Wind

Wind can increase evaporation by removing moist air near the surface:

```text
Wind ↑
   |
   v
Evaporation potential ↑
```

### UV Index

Higher UV indicates stronger UV radiation exposure and can be incorporated as an indicator of solar drying conditions:

```text
UV ↑
 |
 v
Solar exposure ↑
 |
 v
Potential drying contribution ↑
```

These are **contributing factors, not deterministic rules**.

---

## 18. Image + Weather Fusion

This is a key part of the project.

Example:

```text
IMAGE:
Dryness = 75
Grass = 30%
Cracks = 65%

WEATHER:
Temperature = high
Humidity = low
Rainfall = 0
Wind = moderate
UV = high
```

Both sources support drying:

```text
Visual evidence
      +
Weather evidence
      |
      v
Strong drying indication
```

But consider:

```text
IMAGE:
Dry-looking surface

WEATHER:
Recent heavy rainfall
High humidity
Cloudy conditions
```

The evidence conflicts.

Therefore, the system should not blindly trust one source:

```text
Image evidence ------+
                      |
                      v
                 Fusion Layer
                      ^
                      |
Weather evidence ----+
```

Confidence should decrease when sources strongly disagree.

---

## 19. Feature Fusion Strategy

A practical initial approach is weighted scoring:

```text
Final Score =
    Visual Score  × Visual Weight
  + Weather Score × Weather Weight
  + CNN Score     × CNN Weight
```

Example starting weights:

```text
Visual Weight  = 0.50
Weather Weight = 0.25
CNN Weight     = 0.25
```

These are **initial design values only**. They should eventually be selected/validated through experiments and labelled data.

---

## 20. Example Fusion

Suppose:

```text
Visual:
    batting = 0.65
    seam    = 0.20
    spin    = 0.15

CNN:
    batting = 0.60
    seam    = 0.15
    spin    = 0.25

Weather:
    strong drying conditions
```

The analysis may increase the importance of:

```text
dryness
spin potential
```

and produce:

> **Batting-friendly initially, with potential spin assistance as the surface dries.**

The system should avoid forcing everything into one rigid class when the evidence supports a more nuanced result.

---

## 21. Pitch Condition vs Pitch Behaviour

These should be treated as two different concepts.

### Pitch Condition

What the surface looks/appears like:

```text
Moist
Moderate
Dry
Very Dry
Cracked
Grass-covered
```

### Pitch Behaviour

How it may affect gameplay:

```text
Batting
Seam
Swing
Spin
Bowling
```

Therefore:

```text
Image + Weather
       |
       v
Pitch Condition
       |
       v
Expected Behaviour
```

This separation improves explainability.

---

## 22. Example Decision Logic

### Green + Moist

```text
Grass high
Moisture high
Humidity high
Recent rainfall
        |
        v
Likely early seam assistance
```

### Dry + Hard

```text
Grass low
Dryness high
Temperature high
UV high
Humidity low
        |
        v
Potentially good batting surface initially
```

### Dry + Cracked

```text
Dryness high
Cracks high
Humidity low
No recent rainfall
        |
        v
Increasing possibility of spin assistance
```

### Wet Conditions

```text
Recent rain
Humidity high
Surface visually dark/moist
        |
        v
Lower drying confidence
Potentially slower/moister surface
```

These are **decision hypotheses** and should ultimately be validated against expert labels or real match/pitch observations.

---

## 23. Confidence Score

The system should produce a confidence estimate.

Example:

```text
Prediction:
Spin-friendly

Confidence:
78%
```

Confidence should represent **system/model agreement**, not guaranteed real-world accuracy.

Confidence can be reduced when:

- image quality is poor
- weather data is unavailable
- image and weather evidence conflict
- visual features are ambiguous

Example:

```text
Strong agreement:
Image -> dry
Weather -> drying
CNN -> spin
        |
        v
Confidence increases
```

Whereas:

```text
Image -> dry
Weather -> heavy rain
CNN -> batting
        |
        v
Confidence decreases
```

---

## 24. Explainability Layer

A strong project feature is an explanation of the result.

Instead of:

```text
Prediction = Spin
```

return:

```text
Why?

• Low grass coverage
• High visible dryness
• Significant surface cracking
• Low humidity
• No recent rainfall
• High UV conditions
```

The user can therefore understand **why** the system reached its conclusion.

---

## 25. MongoDB

MongoDB stores analysis results and can later support history, comparison, analytics and model improvement.

Example document:

```json
{
  "image_id": "abc123",
  "location": {
    "city": "Hyderabad"
  },
  "visual_features": {
    "grass_coverage": 61.22,
    "dryness_score": 72.4,
    "brown_surface": 68.1,
    "crack_score": 18.5
  },
  "weather": {
    "temperature": 31,
    "humidity": 45,
    "rainfall": 0,
    "wind_speed": 12,
    "uv_index": 8
  },
  "prediction": {
    "batting": 0.80,
    "seam": 0.50,
    "spin": 0.70
  },
  "final_prediction": "Batting Friendly",
  "confidence": 0.78,
  "created_at": "..."
}
```

Potential future uses:

- analysis history
- pitch comparison
- dashboard analytics
- model improvement
- historical pitch studies

---

## 26. Suggested Backend Structure

```text
backend/
│
├── main.py
│
├── routes/
│   ├── analysis.py
│   └── weather.py
│
├── services/
│   ├── image_processing.py
│   ├── feature_extraction.py
│   ├── weather_service.py
│   ├── pitch_analysis.py
│   └── explanation.py
│
├── models/
│   ├── request_models.py
│   └── response_models.py
│
├── ml/
│   └── pitch_classifier.onnx
│
├── database/
│   └── mongodb.py
│
└── utils/
    └── preprocessing.py
```

---

## 27. Suggested Frontend Structure

```text
frontend/
│
├── src/
│   ├── components/
│   │   ├── ImageUpload.jsx
│   │   ├── ImageCropper.jsx
│   │   ├── WeatherCard.jsx
│   │   ├── FeatureCard.jsx
│   │   └── PitchResult.jsx
│   │
│   ├── pages/
│   │   ├── Home.jsx
│   │   └── Analysis.jsx
│   │
│   ├── services/
│   │   └── api.js
│   │
│   └── App.jsx
│
└── package.json
```

---

## 28. Development Roadmap

### Phase 1 — Image Input

```text
React
  |
Upload
  |
Manual Crop
  |
Send Cropped Image
```

### Phase 2 — Basic Computer Vision

Implement:

```text
Grass coverage
Dryness
Brown surface
Crack detection
Texture
```

The current development has already reached the stage where pitch visual features such as grass coverage are being extracted.

### Phase 3 — WeatherAPI

Integrate WeatherAPI.com and retrieve:

```text
Temperature
Humidity
Rainfall
Wind
UV
```

### Phase 4 — Existing ResNet-18

```text
Cropped image
     |
ResNet-18 ONNX
     |
4 class probabilities
```

### Phase 5 — Fusion Engine

Combine:

```text
CV features
+
Weather
+
CNN
```

### Phase 6 — Explanation

Generate:

```text
Prediction
+
Confidence
+
Reasons
```

### Phase 7 — MongoDB

Save analysis results.

### Phase 8 — Dashboard

Display:

```text
Image
Features
Weather
Prediction
Confidence
Explanation
```

### Phase 9 — Validation

Compare system predictions against:

- expert pitch classifications
- historical pitch reports
- match observations
- additional manually labelled data

This validation phase is essential for research credibility.

---

## 29. Research Contribution

The project should **not primarily be presented as**:

> “We trained ResNet-18 to classify cricket pitches.”

A stronger framing is:

> **An explainable multimodal cricket pitch analysis system that combines visual surface characteristics, environmental/weather conditions, and learned pitch classification to estimate pitch condition and expected playing behaviour.**

Therefore:

```text
Supporting ML component:
    ResNet-18 ONNX

Larger system:
    Computer Vision
        +
    Weather Intelligence
        +
    Machine Learning
        +
    Feature Fusion
        +
    Explainable Decision Making
```

---

## 30. Scientific Limitation

The system should **not claim to predict the exact outcome of a cricket match**.

It should estimate:

```text
Pitch condition
      +
Likely pitch behaviour
```

Actual pitch behaviour also depends on:

- pitch preparation
- rolling
- watering
- soil composition
- match duration
- ball condition
- sunlight
- maintenance
- player/bowler characteristics
- stadium conditions

Correct wording:

> “The system estimates likely pitch characteristics and playing behaviour based on available visual and environmental evidence.”

Avoid:

> “The system guarantees how the pitch will behave.”

---

## 31. Complete Runtime Flow

```text
                         USER
                          |
                          v
                Upload pitch image
                          |
                          v
                  Manual crop image
                          |
                          v
                    React frontend
                          |
                          v
                   FastAPI /analyze
                          |
             +------------+------------+
             |                         |
             v                         v
      Image Processing           WeatherAPI.com
             |                         |
             v                         v
      Visual Features            Weather Features
             |                         |
             |                         |
             +------------+------------+
                          |
                          v
                    Feature Fusion
                          |
             +------------+------------+
             |            |            |
             v            v            v
          Vision         CNN        Weather
          features    ResNet-18     features
             |            |            |
             +------------+------------+
                          |
                          v
                   Decision Engine
                          |
                          v
                   Pitch Condition
                          |
                          v
                   Pitch Behaviour
                          |
                          v
                      Confidence
                          |
                          v
                     Explanation
                          |
             +------------+------------+
             |                         |
             v                         v
          MongoDB                 React UI
             |                         |
             +------------+------------+
                          |
                          v
                    FINAL RESULT
```

---

## 32. Final One-Sentence Definition

> **Pitch Insight takes a manually cropped cricket-pitch image, extracts measurable visual surface characteristics, combines them with WeatherAPI environmental data and the existing ResNet-18 ONNX classification, and produces an explainable estimate of pitch condition and likely batting/seam/spin behaviour.**

---

## 33. Final Concept Summary

```text
             +----------------------+
             |   Pitch Image        |
             +----------+-----------+
                        |
                  Manual Crop
                        |
                        v
             +----------------------+
             |  Computer Vision     |
             +----------+-----------+
                        |
          +-------------+-------------+
          |             |             |
        Grass        Dryness       Cracks
          |             |             |
          +-------------+-------------+
                        |
                        v
               Visual Representation
                        |
            +-----------+-----------+
            |                       |
            v                       v
       ResNet-18 ONNX          WeatherAPI.com
            |                       |
            v                       v
      Pitch probabilities    Environmental data
            |                       |
            +-----------+-----------+
                        |
                        v
                  Feature Fusion
                        |
                        v
               Analysis / Reasoning
                        |
             +----------+----------+
             |          |          |
             v          v          v
          Condition  Behaviour  Confidence
             |          |          |
             +----------+----------+
                        |
                        v
                   Explanation
                        |
                        v
                     MongoDB
                        |
                        v
                    React UI
```

**Core principle:** the system should use **visual evidence + environmental evidence + learned classification**, rather than relying on a single image classifier. The goal is an **interpretable pitch insight**, not an absolute prediction of match outcome.
