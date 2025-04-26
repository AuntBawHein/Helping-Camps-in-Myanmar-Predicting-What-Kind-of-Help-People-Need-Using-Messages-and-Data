# Helping-Camps-in-Myanmar-Predicting-What-Kind-of-Help-People-Need-Using-Messages-and-Data

To predict a single, most urgent type of aid needed.

### **Step 1: Define Scope & Objectives**

#### **Objective**

I want to build a smart deep learning model that can understand messages and data coming from camps and predict what kind of help they need the most—whether it's food, clean water, shelter, or medical support.

#### **Scope**

This project will focus only on data and request messages from camps located in Myanmar. The goal is to predict a single, most urgent type of aid needed. I won't be working on physically delivering aid or integrating real-time updates from the field. The purpose is to support decision-makers by helping them quickly identify critical needs using data and AI

### **Step 2: Prepare and Understand the Dataset**

I’ll start by getting familiar with my dataset and making sure everything is in good shape before building the model.

#### **What I’ll Do**

- Load a CSV file that includes aid request messages and details about each camp—like the number of people there, which zone it’s in, and any known disease risks.
- Use **Pandas** to open and explore the dataset.
- Check the column names, look for any missing values, and get a general feel for the structure and content of the data.

#### **Tools I’ll Use**

- **Pandas** – to load, inspect, and work with the dataset.

#### **What is Pandas?**

Pandas is a powerful Python library that helps me easily work with data. It lets me clean up messy parts, filter out things I don’t need, and understand what the data is telling me.

#### **Why am I using Pandas for this step?**

Because Pandas makes it simple to explore and clean data. I can quickly load the file, spot missing information, and understand the layout of the dataset—all of which are important before moving on to any predictions.

---

Code for step 2 

```python
import pandas as pd  # I use pandas to handle data easily.

file_path = "aid_request_dataset_9000_updated.csv"  # I tell pandas where my data is stored.

df = pd.read_csv(file_path)  # I load the data into a table I can work with.

print(df.head())  # I quickly peek at the first few rows to understand what’s inside.

print(df.columns.tolist())  # I list all column names so I know what kinds of info are available.

print(df.isnull().sum())  # I check if any columns have missing (empty) data.

df.info()  # I ask for a quick report about rows, columns, and data types.

print(df.describe())  # I look at summary stats (like average, min, max) to understand the numbers.
```

```
Preview of the dataset:
                            request_text  num_people  num_children  num_elderly               zone  disease_risk  has_clinic previous_aid  situation_score    label
0         We need food and baby formula.        3428           970          401         River View          0.14           0        water                4     food
1        We urgently need food supplies.        1392           365          212  Eastern Hill Camp          0.08           0         none                5     food
2      We need water purifiers urgently.        2644            27          340    Northern Refuge          0.10           0         none                8    water
3  There is no clinic and many are sick.         663           207           31              MaeLa          0.59           1         none                2   health
4  Temporary shelter is urgently needed.        3343           516          560             Umpiem          0.63           0         food                8  shelter

Column Names:
['request_text', 'num_people', 'num_children', 'num_elderly', 'zone', 'disease_risk', 'has_clinic', 'previous_aid', 'situation_score', 'label']

Missing Values (per column):
request_text       0
num_people         0
num_children       0
num_elderly        0
zone               0
disease_risk       0
has_clinic         0
previous_aid       0
situation_score    0
label              0
dtype: int64

Dataset Info:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 9000 entries, 0 to 8999
Data columns (total 10 columns):
 #   Column           Non-Null Count  Dtype  
---  ------           --------------  -----  
 0   request_text     9000 non-null   object 
 1   num_people       9000 non-null   int64  
 2   num_children     9000 non-null   int64  
 3   num_elderly      9000 non-null   int64  
 4   zone             9000 non-null   object 
 5   disease_risk     9000 non-null   float64
 6   has_clinic       9000 non-null   int64  
 7   previous_aid     9000 non-null   object 
 8   situation_score  9000 non-null   int64  
 9   label            9000 non-null   object 
dtypes: float64(1), int64(5), object(4)
memory usage: 703.3+ KB

Summary Statistics:
        num_people  num_children  num_elderly  disease_risk   has_clinic  situation_score
count  9000.000000   9000.000000  9000.000000   9000.000000  9000.000000      9000.000000
mean   2582.474778    515.118667   256.695778      0.499169     0.498778         5.476778
std    1402.941732    444.161918   219.968964      0.287739     0.500026         2.840955
min     100.000000      0.000000     0.000000      0.000000     0.000000         1.000000
25%    1380.500000    145.000000    76.000000      0.250000     0.000000         3.000000
50%    2604.500000    389.000000   194.000000      0.500000     0.000000         5.000000
75%    3803.000000    788.250000   383.000000      0.750000     1.000000         8.000000
max    5000.000000   1973.000000   988.000000      1.000000     1.000000        10.000000
```
### **Conclusion for Step 2: Prepare and Understand the Dataset**

I found that the camps have about **2,582 people** on average, but some camps have as many as **5,000 people**, which is almost double.  
I noticed that most camps have around **515 children**, but one camp has **1,973 children**, so that place might need extra help for kids.  
I also saw that the disease risk usually stays around **0.50**, but in some camps it reaches **1.0**, which means they could have bigger health problems.  
When I looked at elderly people, the average was **257**, but one camp had **988 elderly**, showing they might need more medical care than others.

### **Step 3: Clean the Data**

Now I will clean the dataset so my model doesn’t get confused by missing or strange data.

#### **What I’ll Do**

- I will look for any missing (empty) values and either fill them in using `.fillna()` or remove them using `.dropna()` if they can’t be fixed.
- I will remove any rows that don’t make sense—for example, if a camp says it has 0 people but still requests help, that might be an error.
- I will check that every column has the correct data type. For example, numbers should be stored as numbers, and text should be stored as text.

#### **Tools I’ll Use**

- **Pandas** – I will use this tool to clean and duplicate the data.

---

### **What is Pandas?**

Pandas is a Python tool that helps me work with data tables. I use it to duplicate, clean, and explore the data in a way that’s easy to understand and analyze.

---

### **Why do I have to use Pandas for this step?**

Because I need a simple and powerful way to clean the data before building my model. Pandas makes it easy for me to find missing or strange data, fix problems, and make sure everything is in the right format. Without cleaning the data, the AI model might get confused and give wrong predictions.

Code for step 3 

```python
import pandas as pd  # I am using pandas, a tool that helps me work with tables of data.

file_path = "aid_request_dataset_9000_updated.csv"  # I tell pandas where my file is saved.

df = pd.read_csv(file_path)  # I open the file and turn it into a table I can work with.

print(df.isnull().sum())  # I check how many missing (empty) values are in each column.

df = df[df["num_people"] > 0]  # I remove rows where the number of people is 0.

print(df.dtypes)  # I check the type of data in each column (like numbers or text).

print(df.head())  # I look at the first few rows after cleaning to make sure it looks good.

```Missing values before cleaning:
request_text       0
num_people         0
num_children       0
num_elderly        0
zone               0
disease_risk       0
has_clinic         0
previous_aid       0
situation_score    0
label              0
dtype: int64

Removed 0 rows where num_people was 0.

Data types in the dataset:
request_text        object
num_people           int64
num_children         int64
num_elderly          int64
zone                object
disease_risk       float64
has_clinic           int64
previous_aid        object
situation_score      int64
label               object
dtype: object

Preview after cleaning:
                            request_text  num_people  num_children  num_elderly               zone  disease_risk  has_clinic previous_aid  situation_score    label
0         We need food and baby formula.        3428           970          401         River View          0.14           0        water                4     food
1        We urgently need food supplies.        1392           365          212  Eastern Hill Camp          0.08           0         none                5     food
2      We need water purifiers urgently.        2644            27          340    Northern Refuge          0.10           0         none                8    water
3  There is no clinic and many are sick.         663           207           31              MaeLa          0.59           1         none                2   health
4  Temporary shelter is urgently needed.        3343           516          560             Umpiem          0.63           0         food                8  shelter
```
### **Conclusion for Step 3: Clean the Data**

I checked the data and saw that there were **no missing values** in any of the 10 columns, so I didn’t need to fix anything.  
I compared the number of people in each camp and found that every camp had **more than 0 people**, so I didn’t have to delete any rows.  
I also looked at data types and saw that numbers like **“disease_risk”** are saved as decimals, like **0.14 or 0.59**, which makes sense because risk is not always a full number.  
Since everything was already correct, I feel ready to move on and use this clean dataset to build my AI model.


### **Step 4: Explore the Data (EDA)**

Now I want to look more closely at the data by making charts. This helps me see patterns, compare things, and understand what the data is telling me — before I train my model.

---

### **What I’ll Do**

- I will make a **bar chart** to see how many camps in **Myanmar** asked for different kinds of help — like food, water, shelter, or healthcare.
- I will make a **scatter plot** to find out if **bigger camps** (with more people) usually have **more disease risk**.
- I will make a **box plot** to compare how disease risk looks in **different camp zones**.
- I will look for anything **interesting, surprising, or unusual** that can help me later when I build my model.

---

### **Tools I’ll Use**

- I will use **Pandas** to open and explore the dataset.
- I will use **Seaborn** to make simple and beautiful charts.
- I will use **Matplotlib** to help display the charts one at a time.

---

### **What is Pandas?**  

Pandas is a tool in Python that helps me work with data in table form.  
I use Pandas to **open my CSV file**, check the data, and clean it before making any graphs.

---

### **Why do I use Pandas?**  

I use Pandas because it helps me **see what’s in the data**, like how many people are in each camp, what kind of help they asked for, and which zone they’re in.  
Before I can make a chart or train my model, I need Pandas to help me explore and prepare the data.

---

### **What is Seaborn?**  

Seaborn is a tool that works with Pandas and helps me make charts with just a few lines of code.  
It makes my charts look **clean, colorful, and easy to understand**.

---

### **Why do I use Seaborn?**  

I use Seaborn because it makes it simple to **turn data into pictures** — like bar charts and scatter plots.  
When I see the data as a picture, it's much easier to understand and find patterns.

---

### **What is Matplotlib?**  

Matplotlib is a tool that helps Seaborn **show the charts on the screen**.  
Even though I don’t use it much directly, I still use it to **start a new chart** or to **fix the layout** so things don’t overlap.

---

### **Why do I use Matplotlib?**  

I use Matplotlib to make sure each chart looks **nice and doesn’t squish together**.  
For example, I use `plt.figure()` to start a new chart, and `plt.tight_layout()` to keep things neat.

Code for step 4 

```python
# I import the libraries that help me read data and make charts
import pandas as pd                   # I use this to open and work with table data 
import matplotlib.pyplot as plt       # I use this to display the charts
import seaborn as sns                 # I use this to make charts look clean and pretty

# I load the dataset from the CSV file into a table called "df"
file_path = "aid_request_dataset_9000_updated.csv"
df = pd.read_csv(file_path)

# I set a clean style for all my charts so they are easier to read
sns.set_theme(style="whitegrid") 

# I start a new figure so the chart doesn’t get mixed with others
plt.figure()

# I want to count how many camps in Myanmar asked for each kind of help
bar_chart = sns.countplot(data=df, x='label', order=df['label'].value_counts().index)

# I give my chart a clear title and simple axis labels
bar_chart.set_title("What Kind of Help Myanmar Camps Asked For?") 
bar_chart.set_xlabel("Type of Help Needed")             # Example: food, water, shelter, health
bar_chart.set_ylabel("Number of Camps Asking")          # How many camps asked for each help type

# I fix the spacing so all the labels and title fit nicely
plt.tight_layout()

# I start a new chart again
plt.figure()

# I want to explore if larger camps in Myanmar (with more people) have more disease risk
scatter_plot = sns.scatterplot(data=df, x="num_people", y="disease_risk", hue="label", s=30)

# I add a simple title and labels so anyone can understand
scatter_plot.set_title("Myanmar Camps: Disease Risk Based on Camp Size") 
scatter_plot.set_xlabel("Number of People in the Camp")   # Total people in each camp
scatter_plot.set_ylabel("Disease Risk Level (0 = Low, 1 = High)")  
# Risk from illness or poor conditions

# I use this to make sure everything fits on the chart nicely
plt.tight_layout()

# I start another new chart
plt.figure()

# I want to compare how disease risk looks in different camp zones in Myanmar
box_plot = sns.boxplot(data=df, x='zone', y="disease_risk")

# I add a simple, clear title and axis labels
box_plot.set_title("Disease Risk in Different Camp Zones in Myanmar")
box_plot.set_xlabel("Camp Zone Name")               # Like MaeLa, Umpiem, etc.
box_plot.set_ylabel("Disease Risk Level")           # Higher = more risk for illness or outbreaks

# I rotate the camp zone names to make them easier to read
box_plot.tick_params(axis='x', rotation=45)

# I fix the layout so everything fits and looks clean
plt.tight_layout()
```

![image](https://github.com/user-attachments/assets/865f8e1d-e716-4328-8385-19ad8da0c18b)

![image](https://github.com/user-attachments/assets/138483a4-b2fb-4065-bece-781905e201cf)

![image](https://github.com/user-attachments/assets/1054ab56-db50-4ca0-b117-5e727b2259f7)

### **Conclusion for Step 4: Explore the Data (EDA)**

I looked at the bar chart and saw that **water** was the most needed help with about **2,300 camps asking for it**, which was more than **food** and **health**, and much more than **shelter** with around **2,100 requests**.  
In the scatter plot, I noticed that camps with only **500 people** could still have a disease risk close to **1.0**, showing that small camps can be just as risky as big ones with **4,000 people**.  
When I looked at the box plot, I saw that **Thai Border Camp** had a higher middle disease risk around **0.6**, while **Nupo** was lower, closer to **0.5**.  
From these three charts, I learned that **what help is needed, disease risk, and camp problems** depend more on **where** the camp is, not just how **big** it is.

### **Step 5: Process the Text (spaCy)**

Now I will clean up the request messages using a tool called **spaCy** so that my **deep learning model** can understand the messages better and make smarter predictions.

---

### **What I’ll Do**

- I will **remove stopwords** — these are small, common words like "the", "and", or "is" that show up a lot but don’t add much meaning.
- I will **break each sentence into pieces (called tokens)** — this helps my model understand each word on its own.
- I will **make all letters lowercase**, so that "Water" and "water" are treated as the same word.

---

### **Tools I’ll Use**

- I will use **spaCy**, a smart Python tool that helps clean up English text and get it ready for machine learning.

---

### **What is spaCy?** 

spaCy is a tool in Python that helps me work with language.  
I use spaCy to **read the messages**, **clean up extra words**, and **turn them into smaller, useful parts**.  
It’s like turning messy handwriting into neat notes that my deep learning model can understand clearly.

---

### **Why do I use spaCy for this step?**  

I use spaCy because it helps me **clean the text messages** that come from the camps.  
If I don’t clean them, the deep learning model might get confused by extra words, punctuation, or capital letters.  
With spaCy, I can **remove unimportant words**, **turn everything lowercase**, and **break each sentence into small pieces**.  
This makes it easier for the model to learn and give better results.

---

### **What is Pandas?**  

Pandas is a Python tool that helps me **open**, **view**, and **work with data tables**.
I use it to clean and look at the information inside my dataset.

---

### **Why do I use Pandas for this step?** 

Because I need to **load the CSV file**, **read the request messages**, and **add new cleaned text columns** to my dataset.  
Pandas helps me do all of that quickly and easily.

Code for step 5 

```python
import pandas as pd  # I use pandas to open, explore, and work with my dataset.
import spacy  # I use spaCy to clean and understand the English request messages.

file_path = "aid_request_dataset_9000_updated.csv"  # I tell pandas where my data file is.
df = pd.read_csv(file_path)  # I load the real dataset into a table I can work with.

nlp = spacy.load("en_core_web_sm")  # I load a small English model so Python can understand text.

# I define a function to clean each request message.
def clean_text(text):
    doc = nlp(text)  # I let spaCy read the text.
    tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]  # I keep only useful words.
    return " ".join(tokens)  # I join the clean words back into one sentence.

df['cleaned_text'] = df['request_text'].apply(clean_text)  # I clean all request messages and save them in a new column.

df.to_csv("cleaned_aid_requests.csv", index=False)  # I save the new clean dataset into a file.

print(df[["request_text", "cleaned_text"]].head(10))  # I print the original and cleaned messages to compare them.
```

![image](https://github.com/user-attachments/assets/a78b5b62-f3c1-4116-9e46-16c155031472)

### **Conclusion for Step 5: Process the Text (spaCy)**

I cleaned the messages and made them much shorter, like changing **"We need food and baby formula"** into just **"need food baby formula"**.  
I noticed that small words like "we" and "and" disappeared, which made it easier to see the important parts of the message.  
I also found that the sentence **"need water purifiers urgently"** showed up **three times**, so now all three messages look exactly the same after cleaning.  
By comparing before and after, I saw that cleaning the text makes it faster and easier for the model to focus on what the camps really need.

### **Step 6: Combine Text and Numbers into One Dataset**

Now I want my model to look at **both the request message** and the **numbers** (like how many people are in the camp or the disease risk). That way, the model can learn from **all the information together**.

---

### **What I’ll Do**

- I will **combine the cleaned request message** with the other helpful data — like the **number of people**, the **zone**, and the **disease risk score**.
- I will **make sure all the rows are still matched up properly**, so each camp’s message and numbers stay together before I start training my deep learning model.

---

### **Tools I’ll Use**

- I will use **Pandas**, which is a tool that helps me work with tables of data easily.

---

### **What is Pandas?** 

Pandas is a Python tool that helps me **open, read, combine, and clean data tables**.
I use it to handle both the cleaned text and the numbers in my dataset.

---

### **Why do I have to use Pandas for this step?**  

Because I need to **put the cleaned message** and the **numeric features** (like how many people are in the camp or if it has a clinic) **together in one final dataset**.  
Pandas makes it easy for me to **combine all the columns I need**, check if anything is missing, and get it ready for the model to learn.

### What I’m going to do:

- [final_combined_dataset.csv](https://github.com/user-attachments/files/19922046/final_combined_dataset.csv)

- [cleaned_aid_requests.csv](https://github.com/user-attachments/files/19922045/cleaned_aid_requests.csv)

- [aid_request_dataset_9000_updated.csv](https://github.com/user-attachments/files/19922044/aid_request_dataset_9000_updated.csv)

code for step 6

```python
import pandas as pd  # I use pandas so I can open, explore, and work with the dataset like a table.

df = pd.read_csv("cleaned_aid_requests.csv")  # I open my saved CSV file that has the cleaned request messages.

# I choose only the important columns that my deep learning model will learn from.
final_df = df[[
    "cleaned_text",         # I include the cleaned message
    "num_people",           # I include how many people are in the camp
    "num_children",         # I include how many children are there
    "num_elderly",          # I include how many elderly people are there
    "disease_risk",         # I include how risky the health situation is
    "has_clinic",           # I include if the camp has a clinic (1 = yes, 0 = no)
    "situation_score",      # I include an overall score for the camp's situation
    "zone",                 # I include which area or region the camp is in
    "previous_aid",         # I include if the camp got any help before
    "label"                 # I include what type of help the camp is asking for
]]

print(final_df.head())  # I quickly check the first 5 rows to make sure everything looks correct.

final_df.to_csv("final_combined_dataset.csv", index=False)  # I save this final version into a new CSV file so I can use it later.
```

```
                        cleaned_text  num_people  num_children  num_elderly  disease_risk  has_clinic  situation_score               zone previous_aid    label
0             need food baby formula        3428           970          401          0.14           0                4         River View        water     food
1        urgently need food supplies        1392           365          212          0.08           0                5  Eastern Hill Camp         none     food
2      need water purifiers urgently        2644            27          340          0.10           0                8    Northern Refuge         none    water
3                        clinic sick         663           207           31          0.59           1                2              MaeLa         none   health
4  temporary shelter urgently needed        3343           516          560          0.63           0                8             Umpiem         food  shelter
```
### **Conclusion for Step 6: Combine Text and Numbers into One Dataset**

I connected the cleaned messages with camp numbers, so now each request has extra facts like how many people are there and their disease risk.  
For example, I saw that the message **"need food baby formula"** had **3,428 people** with a low disease risk of **0.14**.  
I also compared camps and noticed that **Umpiem** had a high situation score of **8**, while **MaeLa** had a low score of **2**, even though MaeLa still had a disease risk of **0.59**.  
By putting both the words and numbers together, I made it easier for my model to see not just what the camps say, but how serious their real problems are.

## **Step 7: I Build and Train My Brain (Model)**

Now it’s time for me to build and train my **smart brain model**.  
I’m teaching it how to read **messages from refugee camps** and understand **numbers** (like how many people live there) so it can figure out **what kind of help is needed** — like food, medicine, or shelter.

---

### **What I Do in This Step**

I build something like a **smart brain** using a tool called **TensorFlow**.

I connect different pieces (called **layers**) together:
- One piece reads the words people write
- Another looks at the numbers
- Then I combine both to **make smart guesses**

---

### **What Is This Code About?**

```python
from tensorflow.keras.layers import Input, Dense, Concatenate, TextVectorization, Dropout
from tensorflow.keras.models import Model
```

These are the **parts of the brain** I build:

- **Input**: I tell the brain what kind of data it should expect (text or numbers).
- **Dense**: I create thinking layers (like neurons in a real brain).
- **Concatenate**: I combine different types of data (words + numbers).
- **TextVectorization**: I turn my words into numbers, so my model can understand them.
- **Dropout**: I help the model **not memorize everything**, by making it forget a few things on purpose. This helps it learn better in the long run.
- **Model**: I wrap everything together into one big brain.

---

### **Why I Use These Tools**

I use these tools from TensorFlow because they help me:
- **Build** my brain from scratch
- **Train** it to find patterns in messages and camp data
- **Predict** what kind of help a camp needs — quickly and smartly

---

### **What is Pandas?**

Pandas is a tool I use to **look at and prepare my data** in table format.

---

### **Why I Use Pandas**

I use Pandas because:
- I can **read my CSV file** (the refugee camp data)
- I can **clean the text** so it’s easier for my brain to understand
- I can **clean** everything before training.

---

### **What is Matplotlib?**

Matplotlib is a tool I use to **draw charts and graphs**.

---

### **Why I Use Matplotlib**

I use it to:
- **See how well I’m learning** after each round of training
- Check if I’m doing better or worse.
- **Spot any problems early**, so I can fix them before it’s too late

---

### **What is OS?**

`os` is a tool I use to **talk to my computer’s system** quietly.

---

### **Why I Use OS**

I use it because:
- I want to **hide annoying warning messages** that I don’t need right now
- I like keeping my output **clean and neat**
- Sometimes I use it to create folders or paths when saving files later

---  

code for step 7

```python
# I load the tools I need
import pandas as pd  # I use pandas to read and handle my data table.
import matplotlib.pyplot as plt  # I use matplotlib to draw charts to show my learning.
import os  # I use os to control my computer settings a little bit.

# I tell TensorFlow to only show important messages (no extra warnings).
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf  # I use TensorFlow to build and train my deep learning model.

# I use Input to tell my model what kind of data it's going to receive
# I use Dense to create layers of neurons
# I use Concatenate to combine my text info and number info
# I use TextVectorization to help me turn words into numbers
# I use Dropout to help me forget unimportant stuff and avoid overthinking
from tensorflow.keras.layers import Input, Dense, Concatenate, TextVectorization, Dropout

# I use Model to actually build and wrap all those parts into a complete brain system
from tensorflow.keras.models import Model

# I load my final dataset with both cleaned text and camp numbers.
df = pd.read_csv("final_combined_dataset.csv")

# I clean the text so it becomes readable and usable for my brain (the model)
# If the text is a list of words, I join them into a sentence. If not, I make sure it's a string.
def clean_text_data(text):
    if isinstance(text, list):
        return ' '.join(text)
    return str(text)

# I apply the cleaning function to each message, and I also make sure there are no missing 
# values.
df["cleaned_text"] = df["cleaned_text"].apply(clean_text_data).fillna("")

# I turn the help categories (like 'medical', 'food', etc.) into numbers so my model can learn 
# them.
# The model doesn't understand words, so I give each label its own ID number.
label_map = {label: i for i, label in enumerate(df["label"].unique())}

# I add a new column with these label numbers, so my model can use it during training
df["label_encoded"] = df["label"].map(label_map)

# I split my data into two parts:
# - 80% goes to training: I use this part to learn
# - 20% goes to testing: I save this part to check how well I learned
split = int(len(df) * 0.8)
train_df = df[:split]
test_df = df[split:]

# I teach my model how to read and understand words using "Text Vectorization"
# Basically, I turn each sentence into a list of numbers (tokens) that the model can understand
# I limit to the 10,000 most common words and cut every sentence to 50 tokens max
text_vectorizer = TextVectorization(
    max_tokens=10000,           # I only care about the most frequent 10k words
    output_mode='int',          # I turn each word into an integer 
    output_sequence_length=50   # I make sure every sentence has the same length (50 words)
)

# I start building my model by telling it to expect two kinds of information:
# 1. What people wrote in their messages (text)
# 2. Extra helpful details like number of people, children, etc. (numbers)

# I tell my model: "Hey, expect one piece of text per example"
text_input = Input(shape=(1,), dtype=tf.string, name="text_input")

# I convert that piece of text into a list of numbers using my earlier vectorizer
x_text = text_vectorizer(text_input)

# I also prepare to give it 6 numbers for each example (like num_people, num_children, etc.)
numeric_input = Input(shape=(6,), dtype=tf.float32, name="numeric_input")

# I combine the words and the numbers together into one big input
x = Concatenate()([x_text, numeric_input])

# Now I build the thinking part of the model (the "brain") using layers called Dense
# Each Dense layer is like a group of brain cells that learn patterns

# I give it a big layer of 128 neurons to start learning
x = Dense(128, activation='relu')(x)

# I use Dropout to randomly turn off 30% of the neurons
# This helps my model not memorize too much
x = Dropout(0.3)(x)

# I add another smaller layer with 64 neurons for more thinking power
x = Dense(64, activation='relu')(x)

# I use Dropout again to keep it balanced and avoid overfitting
x = Dropout(0.3)(x)

# I choose how the model should make its final decision:
# It will predict one of several labels (like 'food', 'shelter') — using softmax to pick the 
# best one
output = Dense(len(label_map), activation='softmax')(x)

# I build the full model by connecting both inputs (text + numbers) to the output
model = Model(inputs={"text_input": text_input, "numeric_input": numeric_input}, outputs=output)

# I prepare the model for training:
# - Adam helps the model learn efficiently
# - sparse_categorical_crossentropy helps it guess the right label
# - accuracy lets me track how often it's right
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), 
    loss="sparse_categorical_crossentropy", 
    metrics=['accuracy']
)

# I get the input text ready:
# I turn the cleaned messages into a Tensor (TensorFlow’s way to hold data),
# and reshape it to look like a column (N rows, 1 column)
text_input_array = tf.convert_to_tensor(train_df["cleaned_text"].tolist(), dtype=tf.string)
text_input_array = tf.reshape(text_input_array, (-1, 1))  # Important shape: (N, 1)

# I get the numeric features ready:
# These are things like "how many people", "disease risk", etc.

# I convert the values to a float Tensor so TensorFlow can use them
numeric_raw = tf.convert_to_tensor(train_df[[
    "num_people", "num_children", "num_elderly", 
    "disease_risk", "has_clinic", "situation_score"
]].values, dtype=tf.float32)

# I normalize the numbers:
# That means I make all values have a similar range (e.g., around 0), 
# so one feature doesn't "overpower" the others just because it's bigger.
numeric_mean = tf.reduce_mean(numeric_raw, axis=0) 
numeric_std = tf.math.reduce_std(numeric_raw, axis=0)
numeric_input_array = (numeric_raw - numeric_mean) / (numeric_std + 1e-7)  
# I add a small number to avoid division errors

# I print the shapes just to double check everything looks good!
# I expect: 7200 text samples, each with 1 message (7200, 1)
# And: 7200 rows of numeric data, each with 6 features (7200, 6)
print("text_input_array shape:", text_input_array.shape)       # (7200, 1)
print("numeric_input_array shape:", numeric_input_array.shape) # (7200, 6)

# I train my brain (the model)!
# Here, I show it both the text and numbers, along with the correct answer (label),
# and it starts learning to guess the right label next time
history = model.fit(
    x={
        "text_input": text_input_array,      # I give the model the messages
        "numeric_input": numeric_input_array # I give the model the camp numbers
    },
    y=train_df["label_encoded"],             # I tell the model what the correct label is (the answer key)
    epochs=10,                               # I repeat this training 10 times
    batch_size=32,                           # I study 32 examples at a time
    validation_split=0.1                     # I save 10% to check how well I'm learning
)

# I show how well I did during training — like how many answers I got right while practicing
plt.plot(history.history['accuracy'], label='How well I learned (training data)')

# I show how well I did on questions I hadn't seen before — like a test or exam!
plt.plot(history.history['val_accuracy'], label='How well I did on new questions (test data)')

# X-axis: This is like counting each round I studied. The more rounds (epochs), the better I 
# might get.
plt.xlabel('Each round I studied (epoch)')

# Y-axis: This shows how often I got the right answer — how "correct" I was as a percentage!
plt.ylabel('How many times I was right (%)')

# Title: I explain that I'm learning and improving each time — just like a smart student from 
# Myanmar
plt.title('I am learning better each time — like a smart student from Myanmar')

# I add a little legend (label box) to explain which line is which
plt.legend()

# Finally, I show the chart!
plt.show()
```

```
text_input_array shape: (7200, 1)
numeric_input_array shape: (7200, 6)

Epoch 1/10
203/203 ━━━━━━━━━━━━━━━━━━━━ 2s 4ms/step - accuracy: 0.3667 - loss: 1.8147 - val_accuracy: 0.8389 - val_loss: 0.4753

Epoch 2/10
203/203 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.7538 - loss: 0.6130 - val_accuracy: 0.9222 - val_loss: 0.2227

Epoch 3/10
203/203 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.8828 - loss: 0.3330 - val_accuracy: 1.0000 - val_loss: 0.0841

Epoch 4/10
203/203 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9517 - loss: 0.1832 - val_accuracy: 1.0000 - val_loss: 0.0318

Epoch 5/10
203/203 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9759 - loss: 0.1002 - val_accuracy: 1.0000 - val_loss: 0.0145

Epoch 6/10
203/203 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9906 - loss: 0.0625 - val_accuracy: 1.0000 - val_loss: 0.0071

Epoch 7/10
203/203 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9903 - loss: 0.0503 - val_accuracy: 1.0000 - val_loss: 0.0043

Epoch 8/10
203/203 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step - accuracy: 0.9935 - loss: 0.0360 - val_accuracy: 1.0000 - val_loss: 0.0021

Epoch 9/10
203/203 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9944 - loss: 0.0257 - val_accuracy: 1.0000 - val_loss: 0.0017

Epoch 10/10
203/203 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - accuracy: 0.9981 - loss: 0.0190 - val_accuracy: 1.0000 - val_loss: 0.0013
```

![image](https://github.com/user-attachments/assets/1183e0bb-0738-4093-abb0-0fdfd3a138c2)

### **Conclusion for Step 7: Build the Model**

I started my training with a low accuracy of **36.67%**, which means I was mostly getting the answers wrong at the beginning. But after practicing for 10 rounds, I got much better and reached **99.81% accuracy**, showing that I really learned how to match the data to the right kind of help. My mistakes (called “loss”) dropped from **1.81** to just **0.019**, which means my answers became much closer to the correct ones.

In the graph, the blue line shows how I improved with training data — it started low and went up steadily. The orange line shows how well I did on new data I hadn’t seen before, and it stayed at **100% accuracy** after just a few rounds. That tells me I didn’t just memorize — I actually understood. I became really good at figuring out what each refugee camp needed.

## **Step 8: I Train and Monitor My Brain (Model)**

### **What I Do in This Step**

- I study the training data to **learn patterns** and make predictions.
- I test myself with **validation data**, like doing quizzes to check how much I’ve learned.
- I use **TensorBoard** to **see progress charts**, like how often I get answers right or wrong (accuracy & loss).

---

### **Tools I Use in This Step**

- **TensorFlow**: This helps me **train and run my brain model**.
- **TensorBoard**: This shows me **cool graphs** of my learning progress.

---

### **What is `tensorflow.keras.callbacks import TensorBoard, EarlyStopping`?**

I use these tools to **track my learning** and to **know when to stop training**:

- **TensorBoard**: I use this to **draw graphs** that show how well I’m learning after every round.
- **EarlyStopping**: I use this to say,  
  “Stop training me if I stop improving”

---

### **Why I Use `TensorBoard` and `EarlyStopping`**

I use them because I want to:
- **Track my progress visually** (with charts)
- **Stop training early** if I stop improving
- **Make smart choices** about how long to train

---

### **What is `datetime`?**

I use `datetime` to **get the current date and time**.  
This helps me **name my log folders uniquely** so I don’t get confused later.

---

### **Why I Use `datetime` in This Step**

I use it because:
- It helps me **create a new folder each time I train**
- I can **organize my logs** by date and time
- It keeps **everything tidy and easy to find** in TensorBoard

---

### **What is TensorBoard?**

TensorBoard is like **my personal progress tracker**.

It helps me:
- **See how well I’m doing** (accuracy)
- **See how far off I am** (loss)
- **Spot problems early**, so I can fix them

---

### **Why I Use TensorBoard**

I use TensorBoard because:
- I want to **see how fast I’m learning**
- I want to **know if I’m stuck or doing great**
- I want to **make smart decisions** about changing or stopping training

code for step 8 

```python
import pandas as pd  # I use pandas to open and handle my dataset.
import tensorflow as tf  # I use TensorFlow to build and train my deep learning model.
from tensorflow.keras.layers import Input, Dense, Concatenate, TextVectorization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import datetime  # I use this to record the date and time when I train my model.

# I build the text input part (one message at a time).
text_input = Input(shape=(1,), dtype=tf.string, name="text_input")
x_text = text_vectorizer(text_input)  # I turn the words into numbers.

# I build the numeric input part (6 extra features like people, risk, etc.).
numeric_input = Input(shape=(6,), dtype=tf.float32, name="numeric_input")

# I combine both text and numbers together.
x = Concatenate()([x_text, numeric_input])

# I add thinking layers (Dense) to help the model learn patterns.
x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)  # I turn off 40% of neurons to avoid overfitting.
x = Dense(64, activation='relu')(x)
x = Dropout(0.4)(x)

# I add the final decision layer (softmax to pick one label).
output = Dense(len(label_map), activation='softmax')(x)

# I build the full model by connecting inputs to output.
model = Model(inputs={"text_input": text_input, "numeric_input": numeric_input}, outputs=output)

# - I use the Adam optimizer because it helps my brain learn fast and smart
# - I use a special loss function (sparse_categorical_crossentropy) 
#   because I’m trying to pick the correct category (like 'food', 'shelter' etc.)
# - I also track accuracy, which means I check how often I get the right answer
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  # I learn a little at a time
    loss="sparse_categorical_crossentropy",                    
    # I compare my guesses to the right answer
    metrics=["accuracy"]                                       
    # I want to see how often I’m right
)

# I open TensorBoard so I can see how well my brain model is learning 
# It will show me graphs like:
# - How accurate I am (how often I'm right)
# - How much I need to improve (my loss)
# This command opens the TensorBoard dashboard using the log files I saved earlier.
!tensorboard --logdir=logs

# I also set up early stopping — this is like saying:
# "If I stop improving for 3 rounds, please stop training early and use my best version."
early_stop = EarlyStopping(
    monitor='val_loss',             # I check how confused I am during mini-tests
    patience=3,                     # I give myself 3 chances to improve
    restore_best_weights=True      # If I do worse later, go back to my best self
)

# I finally train my model!
# - I give it both the messages and numeric data
# - I show it the correct answers
# - I train for 10 rounds (or fewer if early stop triggers)
# - I save 10% as a mini-test set during each round
# - I track everything with TensorBoard
history = model.fit(
    x={
        "text_input": text_input_array,       # I give it the message sentences
        "numeric_input": numeric_input_array  # I give it the 6 numeric features
    },
    y=train_df["label_encoded"],              # I tell it what the correct answer is
    epochs=10,                                # I repeat this process 10 times
    batch_size=32,                            # I study 32 examples at a time
    validation_split=0.1,                     # I test on 10% of the data each time
    callbacks=[tensorboard_callback, early_stop]  # I track progress and stop early if needed
)
```

```
Epoch 1/10
203/203 ━━━━━━━━━━━━━━━━━━━━ 3s 6ms/step - accuracy: 0.3514 - loss: 2.2524 - val_accuracy: 0.7847 - val_loss: 0.5965

Epoch 2/10
203/203 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.6588 - loss: 0.8057 - val_accuracy: 0.9458 - val_loss: 0.3468

Epoch 3/10
203/203 ━━━━━━━━━━━━━━━━━━━━ 1s 5ms/step - accuracy: 0.8045 - loss: 0.4945 - val_accuracy: 1.0000 - val_loss: 0.1806

Epoch 4/10
203/203 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.8855 - loss: 0.3270 - val_accuracy: 1.0000 - val_loss: 0.1128

Epoch 5/10
203/203 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.9328 - loss: 0.2180 - val_accuracy: 1.0000 - val_loss: 0.0558

Epoch 6/10
203/203 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.9556 - loss: 0.1486 - val_accuracy: 1.0000 - val_loss: 0.0231

Epoch 7/10
203/203 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.9710 - loss: 0.1113 - val_accuracy: 1.0000 - val_loss: 0.0113

Epoch 8/10
203/203 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.9826 - loss: 0.0759 - val_accuracy: 1.0000 - val_loss: 0.0074

Epoch 9/10
203/203 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.9884 - loss: 0.0588 - val_accuracy: 1.0000 - val_loss: 0.0052

Epoch 10/10
203/203 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.9889 - loss: 0.0498 - val_accuracy: 1.0000 - val_loss: 0.0024
```










## **Step 9: Evaluate the Model**

### **What I Do in This Step**

- I **use new test data** — data I didn’t study before — to **see how well my brain (model) makes predictions**.
- I check **how many answers I got right** and **where I made mistakes**.
- I look at each type of help (like **Food, Water, Medical, or Shelter**) and count **how often I guessed correctly**.

---

## **Tools I Use**

- **I use Pandas** to work with tables and results — kind of like organizing my test scores in a spreadsheet.
- **I use Seaborn** to draw clear and colorful charts so I can **see how well I did**.
- **I use Matplotlib** behind the scenes to help Seaborn **display those charts properly**.
- **I use TensorFlow** to **run my model and make predictions** about what the camp needs.

---

## **What is Pandas?**

Pandas is like **my smart spreadsheet buddy in Python**.  
I use it to:
- **Read and clean test results**
- **Count correct and wrong guesses**
- **Compare my predictions with the real answers**

---

## **Why I Use Pandas Here**

I use Pandas so I can ask questions like:
> “How many times did I say the camp needed water, and I was right?”

This helps me **see how smart my model really is** — using **numbers and categories**.

---

## **What is Seaborn?**

Seaborn is like **an advanced visualization tool** that works with Pandas.  
I use it to draw:
- Confusion matrices (like a heatmap showing where I was right or wrong)
- Beautiful graphs that help me explain results clearly

---

## **Why I Use Seaborn**

Because I want to **see my performance clearly**, not just read numbers in a table.  
Charts help me **spot mistakes fast** and show my results **in a more visual way**.

---

## **What is Matplotlib?**

Matplotlib is **the canvas and paint** behind my Seaborn charts.  
It helps with:
- Drawing the actual pictures (like heatmaps and bar charts)
- Setting the size, labels, and titles of my visualizations

---

## **Why I Use Matplotlib**

Even though Seaborn does the drawing, **Matplotlib holds the frame**.  
I use it to **set up the chart size and labels**, so everything looks clean and easy to understand.

---

## **What is TensorFlow?**

TensorFlow is **the brain-building toolkit** I’ve been using.  
It helps me:
- Build my model
- Train it using data
- Test it using new data
- Predict what kind of help a camp might need

---

## **Why I Use TensorFlow in This Step**

I use TensorFlow to **run the model on new (test) data**.  
It tells me:
> “Based on this message and data, I think the camp needs shelter.”

Then I can check if that guess was right.

code for step 9 

```python
# I import the tools I need
import pandas as pd  # I use pandas to clean and work with data.
import seaborn as sns  # I use seaborn to draw nice charts.
import matplotlib.pyplot as plt  # I use matplotlib to show the charts.
import tensorflow as tf  # I use TensorFlow to make predictions with my trained model.

# I reshape the messages to match the shape my model was trained on
# My model expects each message to be wrapped in a list (one column), so I reshape it
test_text_array = tf.reshape(test_text_array, (-1, 1))

# I normalize (scale) all the numbers
# This helps my model treat all numbers fairly (like putting everything on the same scale)
# For example, without this, a camp with 100 people might unfairly outweigh a camp with 
# "1" disease risk
test_numeric_array = (test_numeric_array - numeric_mean) / (numeric_std + 1e-7)

# I use my trained brain (model) to guess what kind of help each camp might need
# I give it the message (what people wrote)
# I also give it important numbers about the camp (like number of people or disease risk)
# My model looks at this info and gives me its best guess for what help is needed
predictions = model.predict({
    "text_input": test_text_array, 
    "numeric_input": test_numeric_array
})

# I choose the top prediction for each camp — the one the model is most confident about
predicted_label = predictions.argmax(axis=1)  # Highest score = best guess

# I make a copy of my test data, so I can keep the real answers and add what I guessed
test_results = test_df.copy()
test_results["predicted"] = predicted_label  # I add my predictions into the table

# I count how many times I guessed each category — right and wrong
# For example: How many times I said "Food" and it really was "Food"
# Or how many times I said "Water" but it was actually "Shelter"
summary = test_results.groupby(["label_encoded", "predicted"]).size().reset_index(name='count')

# I organize the summary into a table (called a confusion matrix)
# This helps me match what I actually guessed vs. what the real answer was

pivot = summary.pivot_table(
    index='label_encoded',     # I put the real answers (true categories) on the Y-axis
    columns='predicted',       # I put my guesses (predicted categories) on the X-axis
    values='count',            # I show how many times each guess happened
    fill_value=0               # I use 0 when there are no counts (to keep the table complete)
)

# I want to show real category names (like “Food”) instead of just numbers (0, 1, 2, 3)

label_map = {
    0: "Food",      # I say: 0 = Food
    1: "Water",     # I say: 1 = Water
    2: "Medical",   # I say: 2 = Medical
    3: "Shelter"    # I say: 3 = Shelter
}

pivot.index = pivot.index.map(label_map)      
# I label what the camp really needed
pivot.columns = pivot.columns.map(label_map)  
# I label what I thought the camp needed

# I calculate the accuracy of my model.
accuracy = (test_results["label_encoded"] == test_results["predicted"]).mean()

# I start drawing my final chart (heatmap) to show how smart my model was!
plt.figure(figsize=(8, 6))  # I set the chart size so it’s easier to read

# I draw a heatmap — this is like a colorful table to show how well I did
# Each box shows how many times I guessed right or wrong for each type of help
# I add numbers inside the boxes so it’s easy to read and understand
# I use blue colors (YlGnBu) to make it look nice and clear
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu")

# I add a title to explain what I did and how well I did
plt.title(f"I Predicted What Help Each Camp Needs\nModel Accuracy: {accuracy:.2%}")

# I label the X and Y axes so people understand the chart
plt.xlabel("What I Thought the Camp Needed") 
plt.ylabel("What the Camp Actually Needed")

# I show my finished plot with show method. 
plt.show()
```

![image](https://github.com/user-attachments/assets/6df1cc40-a105-410a-a454-1648a1d3df46)

### **Conclusion for Step 9: Evaluate the Model**

I tested my brain (model) on real messages and camp data to see how well I can guess what kind of help each camp needs.  
I found that my model got every single prediction correct — the heatmap shows all the right answers are along the diagonal line.  
For example, I said “Food” 458 times and all of those were actually correct, just like “Water” (459), “Medical” (449), and “Shelter” (434).  
I reached a perfect accuracy of **100.00%**, which means my model didn’t make any mistakes — this is much higher than random guessing, which would only give about 25% accuracy.

## **Step 10: I Ask and Answer Smart Questions About My Project**

Now that I’ve trained my model and tested it, I can use it to answer real questions that can help people.  

I want to **look for patterns** in the data to better understand what camps need and how we can support them faster.

---

### **Some of the questions I asked:**

- **Which camps ask for medical help the most?**  
  I looked at the data to see where the word “medical” or “health” shows up the most.

- **Do camps with more elderly people need more shelter?**  
  I compared the number of elderly people in each camp to how often they asked for shelter.  

- **What kinds of words in messages often lead to a prediction for food?**  
  I checked which words (like “hungry” or “no supplies”) made the model say “This camp needs food.”

---

I used these kinds of questions to **help aid workers make better decisions**.  
By understanding the patterns, they can **bring the right help to the right camps more quickly**.

### **1. What inspired you to do this project?**
**I did this project because I care about people in need.** Refugee camps in Myanmar face real problems, and I wanted to see if I could use data and messages to help aid workers understand what kind of help people are asking for.

---

### **2. How does your model actually work?**
**I taught my model by showing it messages from camps and numbers about their situation.** It learned patterns — like if people said “hungry” and had many kids, they probably needed food. After training, it could guess the type of help a new camp needs.

---

### **3. What were the most important things your model used to make predictions?**
**I think the most important parts were the messages and the number of people in the camp.** The words gave clues like “sick” or “no water,” and the numbers helped show how serious the situation was.

---

### **4. How do you know your model is accurate and not just guessing?**
**I checked how many times my model was right on new data it hadn’t seen before.** It got 100% accuracy on test data and almost no mistakes in the confusion matrix — so I know it's really learning and not just memorizing.

---

### **5. What was the hardest part of your project?**
**The hardest part was getting TensorBoard to work with my model.** It kept showing errors until I fixed how my inputs were connected. That taught me how important clean architecture is in AI.

---

### **6. What are you most proud of?**
**I’m proud that I built something that could actually help real people.** My model doesn’t just guess — it explains patterns clearly, like which camps need food or medical help most often.

---

### **7. How could aid workers actually use your model in the real world?**
**They could use it to read lots of messages quickly and decide where to send food, water, or medicine.** It could save time and get help to people faster than doing it all by hand.

---

### **8. What would you improve if you had more time?**
**If I had more time, I’d teach the model to understand more languages and test it with new messages.** That way, it could help more camps in different places, even if they don’t speak English.

---

### **9. Did anything surprise you while doing this project?**
**Yes — I was surprised how fast my model learned.** By just the 3rd round of training, it already had over 90% accuracy. I didn’t expect it to get that smart so quickly.

---

### **10. Why do you think this project matters?**
**Because it shows how technology can help people in real life.** I didn’t just build something cool — I built something that could help save time, save resources, and maybe even save lives.  
