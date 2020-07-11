# Transfer Learning for Adversarial NMT
While Neural Machine Translation (NMT) systems have been able to provide near-human level translation accuracies, they rely on large parallel datasets, and perform poorly on low-resource languages where there is insufficient data either on the source or target side. An effective method for improving NMT on low-resource languages is to employ transfer learning, where a model trained on a high-resource language pair is used to initialize training for the low-resource language pair. Most of the research in this field is fairly recent, with some works investigating the effects of parameter freezing and language similarity.

Here, I study the effect of employing transfer learning methods on an adversarial machine translation model (RNN-GAN). Apart from directly initializing the parent GAN model to train the low-resource language pair, the effect of freezing parameters from the parent model during transfer learning has also been tested. The results of these experiments show a consistent increase in the BLEU scores of the child model upon transfer from a parent model, and give rise to several avenues of future work.

## Datasets
All the datasets have been obtained from the Web Inventory of Transcribed and Translated Talks (WIT3), each translated between a different language and English.
1. High-resource languages: German-English, Russian-English, and Czech-English (>100k sentences each)
2. Low-resource language: Slovenian-English (~15k sentences)

## Usage

To preprocess the dataset(s): (The steps below are used to download and process the German-English dataset. These steps have to be repeated for each dataset used in the experiment.)
1. `git clone https://github.com/nazim1021/fairseq.git`
2. `cd examples/translation/; bash prepare-iwslt14.sh; cd ../..` (make the dataset name changes in bash script)
3. `TEXT=examples/translation/iwslt14.tokenized.de-en`
4. `python preprocess.py --source-lang de --target-lang en --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test --destdir data-bin/iwslt14.tokenized.de-en`

### Training and testing:
1. To "pre-train" the model on the high-resource language: 
```
python joint_train.py --data data-bin/iwslt14.tokenized.de-en/  --src_lang de --trg_lang en --learning_rate 1e-3 --joint-batch-size 64 --gpuid 0 --clip-norm 1.0 --epochs 10
```
This will save the model in checkpoints folder.

2. This model is used to initialize training on the low-resource language:
```
python joint_train_load.py --data data-bin/iwslt14.tokenized.sl-en/  --src_lang sl --trg_lang en --learning_rate 1e-3 --joint-batch-size 64 --gpuid 0 --clip-norm 1.0 --epochs 10
```
Use `joint_train_load_paramfreeze.py` for transfer learning with parameter freezing.

3. Generate predictions:
```
python generate.py --data data-bin/iwslt14.tokenized.sl-en/ --src_lang sl --trg_lang en --batch-size 64 --gpuid 0
```
This generates `predictions.txt` and `real.txt`. 

4. The `mosesedecoder` toolkit is used to determine BLEU  score for evaluation:

a. Postprocess the real and predictions text files
```
bash postprocess.sh < real.txt > real_processed.txt
bash postprocess.sh < predictions.txt > predictions_processed.txt
```
b. Run BLEU evaluation
```
perl scripts/multi-bleu.perl real_processed.txt < predictions_processed.txt
```

## References
1. Code adapted from: https://github.com/nazim1021/neural-machine-translation-using-gan
2. Barret Zoph, Deniz Yuret, Jonathan May, and Kevin Knight. Transfer learning for low-resource neural machine translation, 2016.
3. Toan Q. Nguyen and David Chiang. Transfer learning across low-resource, related languages for neural machine translation, 2017.
4. Tom Kocmi and Ondˇrej Bojar. Trivial transfer learning for low-resource neural machine translation, 2018.
5. https://github.com/pytorch/fairseq
6. https://github.com/moses-smt/mosesdecoder/tree/master/scripts
7. WIT3. https://wit3.fbk.eu.
