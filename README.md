# miner
mechanistic inference of node-edge relationships

# tutorial information
Several notebooks with instructions and code for performing miner analysis are provided in the miner/src directory. 
We recommend starting with the "download_data" and "download_network" notebooks to load all of the example data.
The "miner_generate_network" notebook will perform the essential network inference steps.
The analyses corresponding to figures in the MINER paper are in tutorial notebooks with the figure names.
Other notebooks offer useful tools for analyzing or interpreting the network.

# what you need to get started
Before the miner analysis can be performed, the gene expression data to be analyzed must be added to the miner/data/expression directory. You can normalize the data (e.g., TMM normalization), then Z-score it and use it directly in the miner workflow. Otherwise, there is a pre-processing function (miner.preprocess) that can be used if your expression data is of the form log2(TPM+1) or log2(FPKM+1).

If survival analysis is desired, a survival file must added to the miner/data/survival directory

If causal analysis is desired, a mutation file must added to the miner/data/mutations directory

# where to put your data
miner/data has folders for expression, survival, and mutations data. Follow the download_data and download_network tutorial notebooks to generate example files. When using your own data, apply the same data formatting and update the tutorial code with the names of your files (and filepaths). The notebooks expect that the miner3 repository has been downloaded to your desktop.
   
Note that the gene names will be converted to Ensembl Gene ID format

# common mistakes to avoid
1. miner does not yet support expression data in counts format. Ensure that data is in log2(TPM+1) or log2(FPKM+1) format if using miner.preprocess().
2. mechanistic inference includes a step that enforces a minimum correlation coefficient. If your results seem too sparse, try decreasing the minCorrelation parameter (default is R>0.2).
3. Some tutorial notebooks require survival analysis using lifelines. If you have not yet installed it, try "pip install lifelines" in your terminal.
4. GSEA requires gseapy. If you have not yet installed it, try "pip install gseapy" in your terminal
5. This code has been written in python3. Make sure that you are not using python2. We recommend installing anaconda3 if you are new to python.
