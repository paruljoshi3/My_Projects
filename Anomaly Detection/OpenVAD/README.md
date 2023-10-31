Modules - 
1. maf 
    - contains GCN and EDl layers (model2.py)
        - layers.py (GCN layers)
    - contains data processing functions for graph classification tasks (utility.py)
    - contains training loop and evaluation metrics for GCN and EDL layers in (train2.py)
    - Custom dataset classes
    - contains modules to train the Masked autoregressive model
        - made.py
        - maf.py
        - maf_layer.py
    - VAE.py 
        - contains the integrated model for MAF with a Transpose Convolution layer
        - training loop for NFs
    - GCN-EDL.py 
        - trains only GCN and EDL Layers to obtain a probabilistic score of anomaly and normal snippets
        - trains on triplet loss (maximise the distance bw anomaly and normal snippets) and MIL loss
    - GCN-NF-EDL.py 
        - final training loop, considers only MIL loss
    - openvadmodel.py
        - complete model class for data pipeline
2. video_features
    - contains feature extractor using i3d model
    - i3dmodel.py  
        - complete model class for data pipeline
3. test.py 
    - integrates i3d feature extractor and openVAD model for data pipeline
4. Results.ipynb 
    - plots the learning curve for GCN and EDL model training 
5. .txt files -
    - records of the training losses and AUC-PR score
6. ckpt 
    - trained weights for EDL layer
    

    