### How to Run

To install dependencies:
```
cat requirements.txt | sed -e '/^\s*#.*$/d' -e '/^\s*$/d' | xargs -n 1 pip install
```
or
```
poetry install
```

To run the Python script:
```
python aspect_based_sentiment.py
```

### Contributors

1. R. Mills
   - SN: 7168755
   - Email: rm718@uowmail.edu.au
   - Contribution: - Full

2. K. Goel
   - SN: 7836685
   - Email: kg956@uowmail.edu.au
   - Contribution: - Full

3. B. Sensha Shrestha
   - SN: 8447196
   - Email: bss541@uowmail.edu.au
   - Contribution: - Full

4. M. Faruk
   - SN: 7056849
   - Email: mzf395@uowmail.edu.au
   - Contribution: - Full



### Aspect-based sentiment analysis: A study of the IMDB review database" 

This application explores the aspect-based sentiment analysis (ABSA) to movie reviews using advanced deep learning techniques, particularly Convolutional Neural Networks (CNN) and Long Short-Term Memory networks (LSTM). The authors, from the University of Wollongong, systematically analyzed sentiments within IMDB movie reviews, achieving accuracy rates of 89% for CNN and 87% for LSTM models. They also used Latent Dirichlet Allocation (LDA) for aspect-specific sentiment analysis related to different elements of movies, achieving a combined accuracy of 78%.

The study underscores the challenges and nuances in analyzing movie reviews, highlighting the effectiveness of ABSA in handling mixed sentiments within a single statement. It covers the methods used for aspect extraction and sentiment classification, including subword tokenization and model management. The document also includes experimental comparisons between CNN and LSTM, exploring the impact of different tokenization techniques, particularly the use of BERT tokenization, which initially decreased accuracy but was later improved with adjustments.

The findings suggest that while both CNN and LSTM are effective for general sentiment analysis, the integration of advanced models like BERT requires careful optimization to improve performance in practical applications. The study also discusses the potential of ABSA to enhance the depth and accuracy of sentiment analysis by focusing on specific aspects of content.