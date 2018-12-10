Currently the system runs with the VCTK data set. I'll tell you how to set up a different data set later in this file.


Step 1: Download the VCTK data set, and extract the contents into the the root folder
    At this point, the root folder should contain a folder named VCTK-Corpus


Step 2: Go into VCTKProcessor, and run the script called "run.sh"
    After this, you should have many npy and some pkl files in the VCTKProcessor/data

    metadata.pkl contains the metadata describing the files here.
        It's a list of tuples (folder_name, list_of_file_names)
        It's ordered so that element 0 describes speech_0.npy, sizes_0.npy ...etc
        Each element of list_of_file_names is the name of one utterance file. 
        These are in the order they appear in speech_[k].npy, sizes_[k].npy ..etc.
    speech_[k].npy contains the _magnitude_ spectrograms for all the utterances associated with one speaker, concatenated along the time axis
    sizes_[k].npy contains the number of frames in each utterance associated with one speaker, in the order they appear in speech_[k].npy
    conv_options.pkl contains the information about how the frames were produced from the samples. We can load this file to see how the conversion was done. It also specifies how the log features should be computed, even though they aren't computed here yet.


Step 3: Go into SpeakerFeatures, and run the script called train.sh
    This will train a speaker verification network, and produce a file called "data/centers.pth" containing the d-vector for each speaker.
    The format of this file should be a PyTorch tensor with shape (num_speakers, d-vector dimensionality).
    You can skip this step if you already computed some d-vectors for the speakers in the VCTK data set, or if you're changing a data set and have a different way of getting d-vectors / face-vectors. As long as you have the "data/centers.pth" file.


Step 4: Go into SpeakerTransfer, and run the script called train.sh
    This will first train a describer network + a reconstructor network (i.e. a denoising autoencoder that also predicts the d-vectors)
    Then it will pretrain the latent forger, initializing it to be similar to an identity function.
    Finally, it will run the remainder of the training process for the latent forger.


To test the set of networks, run playgroound.sh. This will put you into a shell where the describer, reconstructor and latent forger are all loaded. Then you can do other stuff inside the shell, like load audio files and perform speaker forgery.


To set up a different data set, you need to create your own version of VCTKProcessor/process.py and VCTKProcessor/loader.py.
    The former is intended to read the raw data inside the data set, and produce spectrograms that are saved in VCTKProcessor/data
    The latter is used by other scripts in the code base as a data loader for the processed data. The processed data doesn't fit in memory, so what the data loader does is access a random subset of speakers, and load a random span of utterances from each speaker (of fixed length) into memory. Then the data from this loaded subset is iterated until it is used up, and at that point a new random batch of data is loaded into memory.
    You don't have to use the same file structure or loading algorithm, but I'm just telling you my current strategy as a reference.
