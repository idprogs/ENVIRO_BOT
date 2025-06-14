# Code by Ian Drumm
# This module provides a TextEvaluation class for calculating various text similarity
# and quality metrics between original (reference) texts and generated texts.
# Metrics include BLEU, ROUGE, METEOR, BERTScore, embedding similarity,
# perplexity, sentiment intensity, and emotion similarity.

import mauve
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider
from nltk.translate import meteor_score
from bert_score import score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from nrclex import NRCLex
from scipy.spatial.distance import cosine
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from sentence_transformers import SentenceTransformer, util
import evaluate

class TextEvaluation:
    """
    A class to evaluate text generation quality using various metrics.
    It compares generated text against reference (original) text.
    """
    def __init__(self):
        # NLTK resources (punkt, wordnet, omw-1.4) should be downloaded once separately if not present.
        # Example:
        # nltk.download('punkt')
        # nltk.download('wordnet')
        # nltk.download('omw-1.4')
        self.smoothie = SmoothingFunction().method4
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        self.analyzer = SentimentIntensityAnalyzer()

    def calculate_bleu(self, reference, generated):
        """Calculates BLEU score between a reference and a generated sentence."""
        reference = [nltk.word_tokenize(reference)]
        generated = nltk.word_tokenize(generated)
        return sentence_bleu(reference, generated, smoothing_function=self.smoothie)

    def calculate_rouge(self, reference, generated):
        scores = self.rouge_scorer.score(reference, generated)
        return scores

    def calculate_cider(self, references, generated):
        cider_scorer = Cider()
        # Ensure each reference and generated text is in a list
        gts = {i: [ref] for i, ref in enumerate(references)}
        res = {i: [gen] for i, gen in enumerate(generated)}
        score, _ = cider_scorer.compute_score(gts, res)
        return score

    def calculate_meteor(self, reference, generated):
        """Calculates METEOR score between a reference and a generated sentence."""
        reference = nltk.word_tokenize(reference)
        generated = nltk.word_tokenize(generated)
        return meteor_score.single_meteor_score(reference, generated)

    def calculate_bert(self, references, generated):
        """Calculates BERTScore (Precision, Recall, F1) between reference and generated texts."""
        # Note: 'references' and 'generated' are single strings here, but bertscore expects lists.
        # Load BERTScore
        bertscore = evaluate.load("bertscore")
        
        # Calculate BERTScore
        results = bertscore.compute(predictions=[generated], references=[references], lang="en")
        print(results)
        
        # Extract the mean of precision, recall, and F1 scores
        P = sum(results["precision"]) / len(results["precision"])
        R = sum(results["recall"]) / len(results["recall"])
        F1 = sum(results["f1"]) / len(results["f1"])
        
        return P, R, F1
    
    def calculate_bert_df(self, references, generated):
        """Calculates BERTScore and returns results in a DataFrame."""
        P, R, F1 = bert_score(generated, references, lang="en", rescale_with_baseline=True)
        df = pd.DataFrame({
            'Real Comment': references,
            'Bot Commment': generated,
            'P': P.mean().item(),
            'R': R.mean().item(),
            'F1': F1.mean().item()
        })

        return df
    
    def calculate_embedding_similarity(self, reference, generated):
        """Calculates cosine similarity between sentence embeddings of reference and generated text."""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode([reference, generated], convert_to_tensor=True)
        cosine_score = util.pytorch_cos_sim(embeddings[0], embeddings[1])
        return cosine_score.item()

    def calculate_mauve_single_pair(self, real_comment, generated_comment, model_name='facebook/bart-large-cnn'):
        """Calculates MAUVE score for a single pair of real and generated comments."""
        # Tokenizer and model for embeddings
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Calculate MAUVE score for the single pair of comments
        mauve_score = mauve.compute_mauve(
            p_text=[real_comment],
            q_text=[generated_comment],
            device_id=0  # Use GPU if available, set to -1 for CPU
        )

        # Return the MAUVE score for this pair
        return mauve_score.mauve

    def calculate_emotion_similarity(self, reference, generated):
        """Calculates cosine similarity between emotion vectors of reference and generated text using NRCLex."""
        # Initialize NRCLex objects
        ref_emotion = NRCLex(reference)
        gen_emotion = NRCLex(generated)

        # Define the emotions of interest
        emotions_of_interest = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']

        # Get raw emotion scores
        ref_scores = ref_emotion.raw_emotion_scores
        gen_scores = gen_emotion.raw_emotion_scores

        # Create emotion vectors for both texts
        ref_vector = [ref_scores.get(emotion, 0) for emotion in emotions_of_interest]
        gen_vector = [gen_scores.get(emotion, 0) for emotion in emotions_of_interest]

        # Normalize emotion vectors
        ref_total = sum(ref_vector)
        gen_total = sum(gen_vector)
        ref_vector_normalized = [x / ref_total if ref_total else 0 for x in ref_vector]
        gen_vector_normalized = [x / gen_total if gen_total else 0 for x in gen_vector]

        # Handle the case where vectors are all zeros
        if not any(ref_vector_normalized) or not any(gen_vector_normalized):
            similarity = 0.0
        else:
            # Compute cosine similarity between the normalized emotion vectors
            similarity = 1 - cosine(ref_vector_normalized, gen_vector_normalized)

        # Return the similarity score and emotion vectors
        return similarity, ref_vector_normalized, gen_vector_normalized


    def evaluate_emotion_intensity(self, real_comments, generated_comments):
        """
        Evaluates sentiment intensity of real and generated comments, distinguishing between strong emotions and neutral ones.

        Args:
            real_comments (list): List of original real comments.
            generated_comments (list): List of generated comments.

        Returns:
            pd.DataFrame: A DataFrame with sentiment scores for comparison, including emotion intensity.
        """
        
        # Helper function to classify intensity
        def classify_intensity(compound_score):
            if compound_score >= 0.6:
                return 'Strong Positive'
            elif compound_score >= 0.2:
                return 'Mild Positive'
            elif compound_score > -0.2:
                return 'Neutral'
            elif compound_score > -0.6:
                return 'Mild Negative'
            else:
                return 'Strong Negative'

        # Analyze sentiment for real comments
        real_results = [self.analyzer.polarity_scores(comment) for comment in real_comments]
        real_compound_scores = [result['compound'] for result in real_results]
        real_intensities = [classify_intensity(score) for score in real_compound_scores]

        # Analyze sentiment for generated comments
        generated_results = [self.analyzer.polarity_scores(comment) for comment in generated_comments]
        generated_compound_scores = [result['compound'] for result in generated_results]
        generated_intensities = [classify_intensity(score) for score in generated_compound_scores]

        # Create a DataFrame to store the comparison
        df = pd.DataFrame({
            'Real Comment': real_comments,
            'Real Sentiment Compound': real_compound_scores,
            'Real Emotion Intensity': real_intensities,
            'Generated Comment': generated_comments,
            'Generated Sentiment Compound': generated_compound_scores,
            'Generated Emotion Intensity': generated_intensities
        })

        return df

    # Note: The `evaluate_emotion_intensity` method above returns a DataFrame.
    # The `calculate_emotion_intensity` method below returns raw scores for a single pair.

    def calculate_emotion_intensity(self, reference, generated):
        # Initialize VADER SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        
        # Analyze the reference text
        ref_scores = analyzer.polarity_scores(reference)
        
        # Analyze the generated text
        gen_scores = analyzer.polarity_scores(generated)
        
        return ref_scores, gen_scores

    def calculate_perplexity(self, reference, generated):
        """Calculates perplexity for reference and generated texts using GPT-2."""
        # Load pre-trained model and tokenizer (GPT-2)
        model_name = 'gpt2'  # You can use 'gpt2-medium', 'gpt2-large', etc.
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        
        model.eval()  # Set the model to evaluation mode

        # Function to compute perplexity
        def compute_perplexity(text):
            encodings = tokenizer(text, return_tensors='pt')
            input_ids = encodings.input_ids
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss
                perplexity = torch.exp(loss)
            return perplexity.item()
        
        # Calculate perplexity for both texts
        ref_perplexity = compute_perplexity(reference)
        gen_perplexity = compute_perplexity(generated)

        return ref_perplexity, gen_perplexity



    def evaluate(self,original_data, generated_data):
        """
        Evaluates a list of original data against a list of generated data
        using multiple metrics and returns a list of dictionaries with the results.
        Also writes generated comments to 'evalstr.txt'.
        """
        results = []

        # file_path = "./evalstr.txt" # File to log generated comments
        # with open(file_path, "w") as file:
        #     file.write("\n")

        for original, generated in zip(original_data, generated_data):
            
            print("Comparing original and generated comments to a given Reddit post\n", original)
            comment_bleu = self.calculate_bleu(original, generated)
            comment_rouge = self.calculate_rouge(original, generated)
            comment_meteor = self.calculate_meteor(original, generated)
            p, r, f = self.calculate_bert(original, generated)
            es = self.calculate_embedding_similarity(original, generated)
            ref_perplexity, gen_perplexity = self.calculate_perplexity(original, generated)

            ref_scores, gen_scores = self.calculate_emotion_intensity(original, generated)
            similarity_score, ref_vector, gen_vector = self.calculate_emotion_similarity(original, generated)

            # MAUVE and CIDEr can be computationally intensive, uncomment if needed
            # comment_mauve = self.calculate_mauve_single_pair(original, generated) # Requires mauve library
            # comment_cider = self.calculate_cider([original], [generated]) # Requires pycocoevalcap
            results.append({
                'Original': original,
                'Generated': generated,
                'Comment_BLEU': comment_bleu,
                'Comment_ROUGE': comment_rouge,
                'Comment_METEOR': comment_meteor,
                'BERT_P': p,
                'BERT_R': r,
                'BERT_F': f,
                'embedding_similarity': es,
                'real_perplexity' : ref_perplexity,
                'gen_perplexity' : gen_perplexity,
                'real_sentiment_intensity' : ref_scores['compound'],
                'gen_sentiment_intensity' : gen_scores['compound'],
                'emotional_similarity': similarity_score,
                'real_anger': ref_vector[0],
                'real_anticipation': ref_vector[1],
                'real_disgust': ref_vector[2],
                'real_fear': ref_vector[3],
                'real_joy': ref_vector[4],
                'real_sadness': ref_vector[5],
                'real_surprise': ref_vector[6],
                'real_trust': ref_vector[7],
                'gen_anger': gen_vector[0],
                'gen_anticipation': gen_vector[1],
                'gen_disgust': gen_vector[2],
                'gen_fear': gen_vector[3],
                'gen_joy': gen_vector[4],
                'gen_sadness': gen_vector[5],
                'gen_surprise': gen_vector[6],
                'gen_trust': gen_vector[7]
            })

            # with open(file_path, "a") as file:
            #     file.write("["+generated+"]"+"\n")

        return results
