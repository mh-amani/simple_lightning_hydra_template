# simple_hydra_template (replace with your project name)


<!-- to do list -->
- [ ] thing 1
- [ ] thing 2
- [ ] thing 3



## 🚀  Quickstart

```bash
# clone project
git clone ...
cd ...
# create conda env
conda create -n env_name python=3.11 -y
conda activate env_name
# install dependencies
pip install -r requirements.txt

```




## training scripts

To debug on parity dataset:
```bash
python ./src/train.py experiment=parity_rnn trainer=cpu tags=["parity"]
```


