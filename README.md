# ProtMSM: A Length-Aware Gated Hybrid Network Combining Multi-Scale Convolution and Mamba for Robust Protein Function Prediction

##  Installation
```bash
git clone [https://github.com/Rin-ovo/ProtMSM.git](https://github.com/Rin-ovo/ProtMSM.git)
cd ProtMSM
pip install -r requirements.txt

##  Run
After setting up the running environment, please execute the files in the following order：
1.data_split.py
2.seq_embedding.py
3.train.py
When training/predicting datasets from different branches, please modify the ontology entry in configuration_model.py.

## It is recommended to adopt the following directory structure.
ProtMSM/
├── data/
│   ├── bp/                  
│   │   ├── bp_go_ic.txt       
│   ├── mf/                
│   │   ├── mf_go_ic.txt
│   │   └── ...
│   └── cc/                  
│       ├── cc_go_ic.txt
│       └── ...
├── ProtMSM_model.py         
├── configuration_model.py  
└── ...
If you wish to use your own dataset, please store it in the standard directory format.
