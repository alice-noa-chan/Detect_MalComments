# Detect_MalComments
Machine learning model for detecting toxic comments, aimed at improving online communication safety.

# Prepare the dataset
1. Place the file `ko_comments.tsv` in the same folder as the `train.py` file.
   * The `ko_comments.tsv` file should have a 'content' column and a 'label' column. The 'content' is the comment text, and the 'label' indicates if the comment is normal (0) or offensive (1), represented as numbers.

2. Install the necessary packages.
```shell
pip install torch scikit-learn transformers tqdm pandas numpy re emoji soynlp imbalanced-learn
```

   * `train.py` will train using a GPU if it is available. If your environment is not set up for GPU usage, please install PyTorch with cuda from the [PyTorch Official Website](https://pytorch.org/get-started/locally) in the `Compute Platform` section. Training on a CPU will be very slow.

3. Execute the `train.py` file.

# Testing a trained model
You can use the trained model to check for malicious comments. In the `test.py` file, enter the desired value for the `text` variable and run it to get the result.

# Used Model
Training is conducted using the [KcELECTRA model (KcELECTRA-base-v2022)](https://github.com/Beomi/KcELECTRA).

# Example
You can try it out on the [Alice Site - Malicious Comment Detection](https://alice.uiharu.dev/chat) page. Currently, the site is only available in Korean, and the model is also only available in Korean.

# How good is the AI at detection?
In the National Institute of Korean Language's "Modu's Corpus" site, under the Artificial Intelligence AI Speech Evaluation section, at the [Hate Speech Detection (Pilot Operation Task)](https://corpus.korean.go.kr/taskOrdtm/taskList.do?taskOrdtmId=56&clCd=ING_TASK&subMenuId=sub01) competition, a `micro_F1` score of 88.8827586 was achieved.