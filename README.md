# Subtheme Sentiments 

Idea is to develop an approach that given a sample will identify the sub-themes along with their respective sentiments.

<center>
<img src="https://i.ibb.co/yRJJ5wH/Screenshot-2020-10-03-042959.jpg" alt="Screenshot-2020-10-03-042959" border="0">
</center>

## Approach
### Data Exploration
During [Data Exploration](https://github.com/skshashankkumar41/Subtheme-Sentiments/blob/master/output/Data-Exploration.pdf) I came to know that there are around 10k data points and around 90 unique labels but most of them are noisy and are present in very low frequency. So, after doing some preprocessing and undersampling some more frequently occurring labels at the end we have 23 unique labels and around 6k data points. Look [Data Exploration](https://github.com/skshashankkumar41/Subtheme-Sentiments/blob/master/output/Data-Exploration.pdf) for more details.

### My Approach
I considered this problem as a Multi-Label classification and used pre-trained BERT models with fine-tuning to train. 

I chose Pretrained BERT models to leverage the information of Language models and as the data mostly consist of reviews, Language models would work fine, and also It is very easy to Implement. I have used Binary Cross Entropy with Logits as Loss Function.

I have tried both bert-base-uncased and bert-large-uncased pre-trained models to train the data. For more details check [Model Analysis](https://github.com/skshashankkumar41/Subtheme-Sentiments/blob/master/output/Model-Analysis.pdf), bert-large-uncased is performing slightly better but due to its larger size, In this project, I stick with the bert-base-uncased. You can download the trained model from [here](https://drive.google.com/file/d/1kcs0WctkGAqLrzSI1QhsnmK05AG5gopd/view?usp=sharing).

### Performance Metrics 
**Micro f1 score:**
Calculate metrics globally by counting the total true positives, false negatives, and false positives. This is a better metric when we have a class imbalance.

**Macro f1 score:**
Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.

[https://www.kaggle.com/wiki/MeanFScore](https://www.kaggle.com/wiki/MeanFScore)

[http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)

**Hamming loss:** The Hamming loss is the fraction of labels that are incorrectly predicted.

[https://www.kaggle.com/wiki/HammingLoss](https://www.kaggle.com/wiki/HammingLoss)

## Results 
After 5 Epochs model started overfitting. More Details in [Models Analysis](https://github.com/skshashankkumar41/Subtheme-Sentiments/blob/master/output/Model-Analysis.pdf) 
<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky"><span style="font-weight:bold">Metric</span></th>
    <th class="tg-0pky"><span style="font-weight:bold">Training</span></th>
    <th class="tg-0pky"><span style="font-weight:bold">Validation</span></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky"><span style="font-weight:bold">BCE Loss</span></td>
    <td class="tg-0pky">0.019</td>
    <td class="tg-0pky">0.025</td>
  </tr>
  <tr>
    <td class="tg-0pky"><span style="font-weight:bold">F1-Micro-Score</span></td>
    <td class="tg-0pky">0.821</td>
    <td class="tg-0pky">0.737</td>
  </tr>
  <tr>
    <td class="tg-0pky"><span style="font-weight:bold">F1-Macro-Score</span></td>
    <td class="tg-0pky">0.618</td>
    <td class="tg-0pky">0.536</td>
  </tr>
  <tr>
    <td class="tg-0pky"><span style="font-weight:bold">Hamming Loss</span></td>
    <td class="tg-0pky">0.031</td>
    <td class="tg-0pky">0.046</td>
  </tr>
</tbody>
</table>

## Shortcomings and Improvements 
1. As in [Data Exploration](https://github.com/skshashankkumar41/Subtheme-Sentiments/blob/master/output/Data-Exploration.pdf), we combined labels to single label based on the sentiment which have a frequency less than 100 due to which we are ignoring some labels, we can improve this by oversampling those labels by using combinations of co-occurring labels.
2. By experimenting with layers on top of pre-trained BERT could also improve results.
3. By doing some Hyper-Parameter tuning of Batch Sizes, Learning Rate, we could improve results.
4. I have used BCE Loss, some other loss functions could also improve results.

## Usage
Clone the repository and run the following commands from the repository directory.
#### Install project dependencies from requirements.txt
```bash
pip install -r requirements.txt
```

#### Preprocessing Data and Saving Train and Validation Data Pickel File
```python
python preprocess.py
```
#### Training, Evaluating and Saving Model 
```python
python train.py
```
### Inference
```python
python inference.py --text "Your Review Text"

Example -> python inference.py --text "Good prices. easy to arrange local fitting"
Output -> ['ease of booking positive', 'location positive', 'value for money positive']
```

## Files 
<b>[config.py](https://github.com/skshashankkumar41/Subtheme-Sentiments/blob/master/config.py)</b>
<br>
This file contains all the configuration for preprocessing, training, validation, and inference of the model.
<br>

<b>[preprocess.py](https://github.com/skshashankkumar41/Subtheme-Sentiments/blob/master/preprocess.py)
</b>
<br>
This file preprocesses the original data, converts the data to a multi-label classification problem, and also stores the train and validation pickle data. All the methods for preprocessing are commented pretty well in the file itself.
<br>

<b>[dataset.py](https://github.com/skshashankkumar41/Subtheme-Sentiments/blob/master/dataset.py)</b>
<br>
This file creates the custom pytorch dataset using bert tokenizer with all the features required by bert model.
<br>

<b>[dataloader.py](https://github.com/skshashankkumar41/Subtheme-Sentiments/blob/master/dataloader.py)</b>
<br>
This file creates the dataset loader for both train and validation datasets in batches for training.

<b>[model.py](https://github.com/skshashankkumar41/Subtheme-Sentiments/blob/master/model.py)</b><br>
This file creates the custom bert model for multi-label classification, it uses hugging face transformers library to load pre-trained bert.
<br>

<b>[train.py](https://github.com/skshashankkumar41/Subtheme-Sentiments/blob/master/train.py)</b><br>
This file creates the training and validation functions to train and validate the model, Evaluation metrics are also defined in this file itself.
<br>

<b>[validate.py](https://github.com/skshashankkumar41/Subtheme-Sentiments/blob/master/validate.py)</b><br>
This file contains the validation function that requires data loader and model to validate the dataset.
<br>

<b>[utils.py](https://github.com/skshashankkumar41/Subtheme-Sentiments/blob/master/utils.py)</b><br>
This file has some utility functions to save models, print metrics, etc.
<br>

<b>[inference.py](https://github.com/skshashankkumar41/Subtheme-Sentiments/blob/master/inference.py)</b>
<br>
This file contains the function for inference, we can give the reviews directly and it will predict labels using the trained bert model.
<br>


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update the tests as appropriate.

