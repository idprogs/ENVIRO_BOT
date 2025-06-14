# Code by Ian Drumm
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from enum import Enum
from pydantic import BaseModel, Field
import random

# Enums for Classifications
class ViewpointEnum(str, Enum):
    LeftWing = "Left Wing"
    Neutral = "Neutral"
    RightWing = "Right Wing"

class EmotionEnum(str, Enum):
    Happy = "Happy"
    Sad = "Sad"
    Angry = "Angry"
    Surprised = "Surprised"
    Fearful = "Fearful"
    Disgusted = "Disgusted"
    Excited = "Excited"
    Anxious = "Anxious"
    Neutral = "Neutral"
    Content = "Content"
    Proud = "Proud"
    Love = "Love"
    Amused = "Amused"
    Disappointed = "Disappointed"
    Frustrated = "Frustrated"

class SentimentEnum(str, Enum):
    Supportive = "Supportive"
    Critical = "Critical"
    Sceptical = "Sceptical"

class StyleEnum(str, Enum):
    Humorous = "Humorous"
    Sarcastic = "Sarcastic"
    Serious = "Serious"

class TrainingOrTestDataEnum(str, Enum):
    Training = "Training"
    Test = "Test"

# Data Models Using Enums
class ViewpointDataModel(BaseModel):
    Viewpoint: ViewpointEnum = Field(description="the political viewpoint expressed in the comment")

class EmotionDataModel(BaseModel):
    Emotion: EmotionEnum = Field(description="the emotion conveyed in the comment")

class SentimentDataModel(BaseModel):
    Sentiment: SentimentEnum = Field(description="the sentiment conveyed in the comment")

class StyleDataModel(BaseModel):
    Style: StyleEnum = Field(description="the style of the comment")

class TrainingOrTestDataModel(BaseModel):
    Training_or_Test_Data: TrainingOrTestDataEnum = Field(description="whether the data should be used for training or testing, only needed for evaluation metrics")

# Generalized Classification Function
def classify_comment(item, llm, pydantic_model):
    # Get the field name from the Pydantic model
    field_name = next(iter(pydantic_model.model_fields))
    
    # Get the field (ModelField object)
    field = pydantic_model.model_fields[field_name]
    
    # Extract the description for the classification name
    classification_name = field.description  # Adjusted for Pydantic v2.x
    
    # Get the Enum class from the field annotation
    field_type = field.annotation
    if issubclass(field_type, Enum):
        # Get the options from the Enum members
        options = [member.value for member in field_type]
    else:
        raise ValueError("Field type must be an Enum for this function to work.")
    
    # Create the prompt template
    options_str = "', '".join(options)
    prompt_template = (
        f"Given the following Reddit post and comment, classify the {classification_name} of the Comment as one of the following:\n"
        f"'{options_str}'.\n\n"
        "Post: {post}\n"
        "Comment: {comment}\n\n"
        "Please output only the classification."
    )
    
    # Format the prompt with actual data
    prompt_text = prompt_template.format(
        post=item['Post'],
        comment=item['Comment']
    )
    
    # Invoke the LLM with the formatted prompt
    res = llm.invoke(prompt_text)
    
    # Process the response
    classification_result = res.strip().title().replace("-", " ")
    
    # Validate the output using the Pydantic model
    try:
        parsed_output = pydantic_model(**{field_name: classification_result})
        
        # Get the Enum member
        classification_enum = getattr(parsed_output, field_name)
        
        # Return the string value of the classification
        return classification_enum.value
    except Exception as e:
        print(f"Failed to parse the output: {e}")
        return None

# Specific Classification Functions
def get_viewpoint(item=None, llm=None):
    return classify_comment(item=item, llm=llm, pydantic_model=ViewpointDataModel)

def get_emotion(item=None, llm=None):
    return classify_comment(item=item, llm=llm, pydantic_model=EmotionDataModel)

def get_sentiment(item=None, llm=None):
    return classify_comment(item=item, llm=llm, pydantic_model=SentimentDataModel)

def get_style(item=None, llm=None):
    return classify_comment(item=item, llm=llm, pydantic_model=StyleDataModel)

def get_training_or_test_data(item=None):
    if random.random()<=0.75:
        classification="Training"
    else:
        classification="Test"  
    return classification