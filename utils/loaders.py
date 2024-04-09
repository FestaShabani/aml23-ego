import glob
from abc import ABC
import pandas as pd
from .epic_record import EpicVideoRecord
import torch.utils.data as data
from PIL import Image
import os
import os.path
from utils.logger import logger
import numpy as np 
import math

class EpicKitchensDataset(data.Dataset, ABC):
    def __init__(self, split, modalities, mode, dataset_conf, num_frames_per_clip, num_clips, dense_sampling,
                 transform=None, load_feat=False, additional_info=False, **kwargs):
        """
        split: str (D1, D2 or D3)
        modalities: list(str, str, ...)
        mode: str (train, test/val)
        dataset_conf must contain the following:
            - annotations_path: str
            - stride: int
        dataset_conf[modality] for the modalities used must contain:
            - data_path: str
            - tmpl: str
            - features_name: str (in case you are loading features for a predefined modality)
            - (Event only) rgb4e: int
        num_frames_per_clip: dict(modality: int)
        num_clips: int
        dense_sampling: dict(modality: bool)
        additional_info: bool, set to True if you want to receive also the uid and the video name from the get function
            notice, this may be useful to do some proper visualizations!
        """
        self.modalities = modalities  # considered modalities (ex. [RGB, Flow, Spec, Event])
        self.mode = mode  # 'train', 'val' or 'test'
        self.dataset_conf = dataset_conf
        self.num_frames_per_clip = num_frames_per_clip
        self.dense_sampling = dense_sampling
        self.num_clips = num_clips
        self.stride = self.dataset_conf.stride
        self.additional_info = additional_info

        #this was used to create train_val folder in github, which contains D1_test.pkl, D1_train.pkl ..(same for D2,D3)
        if self.mode == "train":
            pickle_name = split + "_train.pkl"
        elif kwargs.get('save', None) is not None:
            pickle_name = split + "_" + kwargs["save"] + ".pkl"
        else:
            pickle_name = split + "_test.pkl"

                     
    	"""
        #self.list_file is a DataFrame that contains metadata about the video samples belonging to a specific split of the dataset, such as D1.
        #This DataFrame could be like this:
        #UID	Label	Untrimmed Video Name	Start Timestamp	End Timestamp	...
        #1	        0	video1.mp4	                    0:05	    0:10	...
        #2	        1	video2.mp4	                    0:00	    0:07	...

        self.video_list  iterates over the rows of the DataFrame self.list_file.
        For each row in self.list_file, tup represents a tuple containing the index and the row data.
        Each tuple tup is passed to the EpicVideoRecord constructor along with self.dataset_conf.
        In the end, self.video_list contains a list of EpicVideoRecord objects, each representing metadata about a single video sample in the dataset. 
        example: self.video_list = [
        EpicVideoRecord(index=0, row_data={'UID': 1, 'Label': 0, 'Untrimmed Video Name': 'video1.mp4', 'Start Timestamp': '0:05', 'End Timestamp': '0:10', ...}),
        EpicVideoRecord(index=1, row_data={'UID': 2, 'Label': 1, 'Untrimmed Video Name': 'video2.mp4', 'Start Timestamp': '0:00', 'End Timestamp': '0:07', ...}),
        ...]
        transform represents a pipeline of transformations such as (an example)
        transform = transforms.Compose([
        transforms.Resize((224, 224)),    # Resize images to 224x224 pixels
        transforms.RandomHorizontalFlip(),# Randomly flip images horizontally
        transforms.ToTensor(),            # Convert images to PyTorch Tensors
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize images
        ])
        load_feat - bool of whether the dataset class loads raw frames(False) or pre-extracted features from the dataset(True).
        This part could be as below:  model_features = pd.DataFrame...['features'])
            uid features_RGB     features_Flow
        0    1  [0.1, 0.2, 0.3]  [0.2, 0.3, 0.4]
        1    2  [0.4, 0.5, 0.6]  [0.5, 0.6, 0.7]
        2    3  [0.7, 0.8, 0.9]  [0.8, 0.9, 1.0]

        Then model_features = pd.DataFrame...['features']) [["uid", "features_" + m]] prvoided m is RGB could be 
            uid features_RGB
        0    1  [0.1, 0.2, 0.3]
        1    2  [0.4, 0.5, 0.6]
        2    3  [0.7, 0.8, 0.9]
        """             
                     
        self.list_file = pd.read_pickle(os.path.join(self.dataset_conf.annotations_path, pickle_name)) #this annotations_path=train_val is in I3D_save_feat.yaml, which will be the name of our folder that contains  the above pkl file. So this path would be train_val/D1_test  
        logger.info(f"Dataloader for {split}-{self.mode} with {len(self.list_file)} samples generated") # example Dataloader for D1_test with 435 samples generated
        self.video_list = [EpicVideoRecord(tup, self.dataset_conf) for tup in self.list_file.iterrows()]
        self.transform = transform  # pipeline of transforms
        self.load_feat = load_feat

        if self.load_feat: #if True the dataset class will load pre-extracted features from the dataset instead of raw frames. The dataset class will load these pre-extracted features and pass them directly to the neural network model without the need for additional preprocessing.
            self.model_features = None # model_features will hold the pre-extracted features.
            for m in self.modalities: # iterates over the modalities to load the pre-extracted features.
                # load features for each modality
                model_features = pd.DataFrame(pd.read_pickle(os.path.join("saved_features", #it constructs a DataFrame from reading the pickle file saved_features/I3D_features_D1_test.pkl .. (ex I3D_features_16_dense_D1_test.pkl)
                                                                          self.dataset_conf[m].features_name + "_" +
                                                                          pickle_name))['features'])[["uid", "features_" + m]]
                if self.model_features is None:
                    self.model_features = model_features
                else: #just merges if it is not non with inner join with rows matching uid
                    self.model_features = pd.merge(self.model_features, model_features, how="inner", on="uid")

            self.model_features = pd.merge(self.model_features, self.list_file, how="inner", on="uid")

    def _get_train_indices(self, record, modality='RGB'):
        ##################################################################
        # TODO: implement sampling for training mode                     #
        # Give the record and the modality, this function should return  #
        # a list of integers representing the frames to be selected from #
        # the video clip.                                                #
        # Remember that the returned array should have size              #
        #           num_clip x num_frames_per_clip                       #
        ##################################################################
        """
        Samples frame indices for training, supporting both dense and uniform sampling.
        
        Args:
            record (EpicVideoRecord): An object containing metadata for a video clip.
            modality (str): The type of modality (e.g., RGB) to consider.
            
        Returns:
            np.ndarray: Indices of frames to sample.
        """
        # Initialize variables for clarity
        num_frames = record.num_frames[modality]
        num_samples = self.num_frames_per_clip[modality] * self.num_clips

        if self.dense_sampling[modality]:
            # Dense sampling logic
            center_frames = np.linspace(0, num_frames, self.num_clips + 2, dtype=np.int32)[1:-1]
            indices = []
            for center in center_frames:
                start = max(0, center - math.ceil(self.num_frames_per_clip[modality] / 2 * self.stride))
                end = min(num_frames, center + math.ceil(self.num_frames_per_clip[modality] / 2 * self.stride))
                indices.extend(range(start, end, self.stride))

            # Handle case where we have fewer frames than needed
            if len(indices) < num_samples:
                indices += [indices[-1]] * (num_samples - len(indices))
        else:
            # Uniform sampling logic
            if num_frames >= num_samples:
                stride = max(1, num_frames // num_samples)
                indices = np.arange(0, stride * num_samples, stride)
            else:
                # When there are fewer frames than needed, repeat the last frame index
                indices = np.arange(0, num_frames)
                additional_indices = [num_frames - 1] * (num_samples - len(indices))
                indices = np.concatenate((indices, additional_indices))

        # Ensure indices are within bounds
        indices = np.clip(indices, 0, num_frames - 1)

        return indices.astype(int)

    def _get_val_indices(self, record, modality):
        ##################################################################
        # TODO: implement sampling for testing mode                      #
        # Give the record and the modality, this function should return  #
        # a list of integers representing the frames to be selected from #
        # the video clip.                                                #
        # Remember that the returned array should have size              #
        #           num_clip x num_frames_per_clip                       #
        ##################################################################
        """
        Samples frame indices for validation/testing, supporting both dense and uniform sampling.
        
        Args:
            record (EpicVideoRecord): An object containing metadata for a video clip.
            modality (str): The type of modality (e.g., RGB) to consider.
            
        Returns:
            np.ndarray: Indices of frames to sample.
        """
        # Initialize variables for clarity
        num_frames = record.num_frames[modality]
        num_samples = self.num_frames_per_clip[modality] * self.num_clips

        if self.dense_sampling[modality]:
            # Dense sampling logic
            center_frames = np.linspace(0, num_frames, self.num_clips + 2, dtype=np.int32)[1:-1]
            indices = []
            for center in center_frames:
                start = max(0, center - math.ceil(self.num_frames_per_clip[modality] / 2 * self.stride))
                end = min(num_frames, center + math.ceil(self.num_frames_per_clip[modality] / 2 * self.stride))
                indices.extend(range(start, end, self.stride))

            # Handle case where we have fewer frames than needed
            if len(indices) < num_samples:
                indices += [indices[-1]] * (num_samples - len(indices))
        else:
            # Uniform sampling logic
            if num_frames >= num_samples:
                stride = max(1, num_frames // num_samples)
                indices = np.arange(0, stride * num_samples, stride)
            else:
                # When there are fewer frames than needed, repeat the last frame index
                indices = np.arange(0, num_frames)
                additional_indices = [num_frames - 1] * (num_samples - len(indices))
                indices = np.concatenate((indices, additional_indices))

        # Ensure indices are within bounds
        indices = np.clip(indices, 0, num_frames - 1)

        return indices.astype(int)

    
    def __getitem__(self, index):
        """
        record =  Given an index, it retrieves the corresponding EpicVideoRecord object from the self.video_list.
        example: self.video_list = [
        EpicVideoRecord(index=0, row_data={'UID': 1, 'Label': 0, 'Untrimmed Video Name': 'video1.mp4', 'Start Timestamp': '0:05', 'End Timestamp': '0:10', ...}),
        EpicVideoRecord(index=1, row_data={'UID': 2, 'Label': 1, 'Untrimmed Video Name': 'video2.mp4', 'Start Timestamp': '0:00', 'End Timestamp': '0:07', ...}),
        ...]
        model_features=
            uid features_RGB     features_Flow
        0    1  [0.1, 0.2, 0.3]  [0.2, 0.3, 0.4]
        1    2  [0.4, 0.5, 0.6]  [0.5, 0.6, 0.7]
        2    3  [0.7, 0.8, 0.9]  [0.8, 0.9, 1.0]

                      uid features_RGB   features_Flow
        sample_row  = 1  [0.1, 0.2, 0.3] [0.2, 0.3, 0.4]
        sample[features_RGB]  =    [0.1, 0.2, 0.3]

        returns either sample, record.label, record.untrimmed_video_name, record.uid = 
        sample = {
        "RGB": [0.1, 0.2, 0.3],
        "Flow": [0.4, 0.5, 0.6]
        }
        record.label = 0
        record.untrimmed_video_name = "video1.mp4"
        record.uid = 1

        (or just the first two)
        """
        
        frames = {}
        label = None
        # record is a row of the pkl file containing one sample/action
        # notice that it is already converted into a EpicVideoRecord object so that here you can access
        # all the properties of the sample easily
        record = self.video_list[index]

        if self.load_feat: # it loads pre-extracted features for the sample 
            sample = {}
            sample_row = self.model_features[self.model_features["uid"] == int(record.uid)] #filters the DataFrame self.model_features to retrieve the row corresponding to the UID of the record above.
            assert len(sample_row) == 1
            for m in self.modalities:
                sample[m] = sample_row["features_" + m].values[0]
            if self.additional_info:
                return sample, record.label, record.untrimmed_video_name, record.uid
            else:
                return sample, record.label

        """ 
        (training) index is the position of a specific frame.
        segment_indices  will hold the sampled indices for each modality.
        the indices for sampling frames are selected from within the range of possible frame indices in the video clip.
        sample_{num_frames}:  represents the total number of frames in the video clip that are available for sampling.
        then the start_index of the sample is added as an offset
        For example, in the end:
        segment_indices = {         #a dictionary that stores the sampled frame indices for each modality within a specific
            'RGB': [10, 15, 20, 25, 30], 
            'Flow': [5, 10, 15, 20, 25]
        }
        
        """
        segment_indices = {}
        # notice that all indexes are sampled in the[0, sample_{num_frames}] range, then the start_index of the sample
        # is added as an offset
        for modality in self.modalities:
            if self.mode == "train":
                # here the training indexes are obtained with some randomization
                segment_indices[modality] = self._get_train_indices(record, modality)
            else:
                # here the testing indexes are obtained with no randomization, i.e., centered
                segment_indices[modality] = self._get_val_indices(record, modality)

        for m in self.modalities:
            img, label = self.get(m, record, segment_indices[m])
            frames[m] = img

        if self.additional_info:
            return frames, label, record.untrimmed_video_name, record.uid
        else:
            return frames, label

    def get(self, modality, record, indices):
    """
    images is a list that will store sampled frames
    p - frame index
    frame = the RGB image (as a list) at that specific index p
    Finally images contains the list of of all RGB frames (img) of those specific indices of that specific record in that specific modality
    Then all those images are transformed (for ex resized, normalized etc).
    get returns these transformed images and the label of that specific video clip (record).
    """

        
        images = list() 
        for frame_index in indices:
            p = int(frame_index)
            # here the frame is loaded in memory
            frame = self._load_data(modality, record, p) 
            images.extend(frame)
        # finally, all the transformations are applied
        process_data = self.transform[modality](images)
        return process_data, record.label

    def _load_data(self, modality, record, idx):
        """
        load_data is responsible for loading a single frame from a specified modality from the dataset.
                record - can be like record = {
                                                'uid': 12345,
                                                'label': 1,
                                                'untrimmed_video_name': 'video1.mp4',
                                                'start_frame': 0,
                                                'num_frames': {
                                                    'RGB': 300,
                                                    'Flow': 300,
                                                    'Spec': 300,
                                                    'Event': 300
                                                },
                                # Other metadata attributes as needed such as start timestamp and end timestamp
                                            }
        idx - index of the frame within the video CLIP
        dataset.RGB.data_path=../ek_data/frames
        
        data_path = ../ek_data/frames
        tmpl=   "img_{:010d}.jpg"

        We have a video clip represented by record, which starts at record.start_frame and contains multiple frames. 
        The idx represents the index of the frame within this specific video clip.
        To find the index of the frame within the entire untrimmed video (so, not in the specificc clip).
        Suppose: 
        record.start_frame = 100: the video clip starts at frame index 100.
        idx = 20: the 20th frame within this video clip.
        idx_untrimmed = 100 + 20 = 120
        The path of the image: something like ../ek_data/frames/img_0000000005.jpg
        It opens the image and returns the RGB version of it in img

        Explanation for the case when idx_untrimmed > max_idx_video:
            Sometimes, due to various reasons such as missing frames or incomplete data, the frame indices in the dataset may not be contiguous.
            In such cases, the maximum frame index found provides an upper bound for the available frames, allowing the code to handle situations where the dataset does not contain frames for every possible index
            If the requested frame index exceeds the maximum available frame index, the code falls back to loading the image for the maximum available frame index. This ensures that even if the requested frame is not available, the code can still provide an image for the nearest available frame.
            return [img] #returns a list with a single image
        """
      
        
        data_path = self.dataset_conf[modality].data_path
        tmpl = self.dataset_conf[modality].tmpl

        if modality == 'RGB' or modality == 'RGBDiff':
            # here the offset for the starting index of the sample is added

            idx_untrimmed = record.start_frame + idx
            try:
                img = Image.open(os.path.join(data_path, record.untrimmed_video_name, record.untrimmed_video_name, tmpl.format(idx_untrimmed))) \
                    .convert('RGB')
            except FileNotFoundError:
                print("Img not found")
                max_idx_video = int(sorted(glob.glob(os.path.join(data_path,
                                                                  record.untrimmed_video_name,
                                                                  "img_*")))[-1].split("_")[-1].split(".")[0])
                if idx_untrimmed > max_idx_video:
                    img = Image.open(os.path.join(data_path, record.untrimmed_video_name, tmpl.format(max_idx_video))) \
                        .convert('RGB')
                else:
                    raise FileNotFoundError
            return [img] 
        
        else:
            raise NotImplementedError("Modality not implemented")

    def __len__(self):
        return len(self.video_list)
