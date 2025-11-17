- Our goal is to build a model that normal people can use to estimate their dementia risk using information they already know about themselves.

In this hackathon, medical data means things only doctors or medical staff would measure or interpret (for example detailed cognitive test scores, scans, lab results). Do not use these as features.

Simple diagnoses that people usually know about themselves, like having had a heart attack or a stroke, are allowed as features.

Please read the data dictionary, choose which columns you think are non-medical and usable, and explain your choices in your report, especially for any “borderline” features.

You are allowed to use AI tools (for example ChatGPT, Copilot) to help you read and understand the data dictionary, but use them carefully and make sure you understand and agree with the final decisions.

Introduction Dementia is a major and growing global health issue. Many risk factors are non-medical, such as lifestyle, education and social context. In this hackathon, you will explore how well non-medical information alone can help predict dementia risk. SCENARIO Your goal: build a binary classification model that predicts whether a person is at risk of dementia or not, expressed as a probability (0–100%), using only non medical variables from the dataset. INTRODUCTION TO THE DATASET Dictionary Dataset Curated subset of the NACC cohort. Each row is one participant visit, with both medical and non-medical features plus a binary label indicating dementia vs. no dementia. Companion document describing every column: variable names, meanings, value encodings and special codes . Use it to understand which fields are allowed and which must be excluded. WHAT YOU NEED TO DO Explore the non-medical data Select a valid feature set Feature engineering and Preprocessing Build a binary classification model Evaluate and improve the model Explain what the model has learned Develop your solution in a python notebook INSTRUCTIONS We are not only judging final model performance but also the overall quality and thoughtfulness of your approach, including preprocessing, feature engineering and hyperparameter tuning. During the hackathon, we encourage you to use version control (for example Git) and to build and compare multiple models rather than stopping at the first one that works, as this will help you achieve a higher score. Use your preferred ML stack You must not use medical/diagnostic variables as model inputs. Do not merge with any external patient-level data sources Use the given template for the report Justify your every step Provide comprehensive Exploratory Dat
- 📢 Use case clarification – Problem and what you should do

Hey everyone! 👋 Here’s a clearer explanation of the task.

🧠 What is the problem?

The main goal of this hackathon is to build a machine learning model that can estimate whether a person is at risk of having dementia or not, using information that normal people already know about themselves (not detailed hospital or lab data).

You can imagine a simple website or app where someone answers questions like:

- How old are you? 🎂
- What is your highest level of education? 🎓
- Who do you live with? 🏠
- Do you smoke or drink alcohol? 🚬🍷
- Have you ever had a heart attack or stroke? ❤‍🩹

After answering, the system would show something like:

- “Your estimated risk of having dementia is X%  and
-  label them as “At risk” or “Not at risk”.

In this hackathon, you are building the model behind that system.

In technical terms:

The inputs to the model are non-medical features about a person (age, education, lifestyle, social factors, simple known diagnoses, etc.).

Your model should learn from the dataset which people have dementia and which don’t, then use those patterns to predict the risk for new people, using only the allowed non-medical information.

👉 In one sentence:

Given a person’s non-medical information, your model should estimate how likely it is that they have dementia and classify them as “at risk” or “not at risk”.

📊 What data can we use?
You should use non-medical information that people usually know about themselves, such as:

Age, sex, education, lifestyle and social factors
Simple diagnoses they already know (for example “I had a heart attack” or “I had a stroke”) Plus many other similar non-medical features in the dataset.

Do not use things that only doctors measure or interpret, like detailed cognitive test scores, scans, lab results or specialist clinical scales.

Use the data dictionary to find suitable non-medical features and explain your choices in the report, especially for any borderline features.

🛠 What should we do? (high level)

Read the data dictionary and pick the non-medical features you will use.

Build and compare more than one model (with preprocessing, feature engineering and hyperparameter tuning).